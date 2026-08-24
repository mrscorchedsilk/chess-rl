#include "mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace chess_rl_native {
namespace {

constexpr int HISTORY_STEPS = 8;
constexpr int NUM_PLANES = 12 * HISTORY_STEPS + 8;  // 104
constexpr int SQUARES = 64;

// Terminal game value from the side to move's perspective (matches
// mcts.py._terminal_value: the mated side to move -> -1, draws -> 0; a side
// to move that somehow "won" -> +1, defensively).
float terminal_value_from_outcome(const Outcome& outcome, const std::string& side) {
    if (outcome.winner.empty()) return 0.0f;
    return outcome.winner == side ? 1.0f : -1.0f;
}

}  // namespace

MCTS::MCTS(double c_puct, double virtual_loss, int num_simulations,
           double dirichlet_alpha, double dirichlet_epsilon, std::uint64_t seed)
    : c_puct_(c_puct),
      virtual_loss_(virtual_loss),
      num_simulations_(num_simulations),
      dirichlet_alpha_(dirichlet_alpha),
      dirichlet_epsilon_(dirichlet_epsilon),
      rng_(seed) {
    if (num_simulations_ < 0)
        throw std::invalid_argument("num_simulations must be >= 0");
    if (virtual_loss_ < 0.0)
        throw std::invalid_argument("virtual_loss must be >= 0");
    if (c_puct_ < 0.0)
        throw std::invalid_argument("c_puct must be >= 0");
    if (dirichlet_epsilon_ < 0.0 || dirichlet_epsilon_ > 1.0)
        throw std::invalid_argument("dirichlet_epsilon must be in [0, 1]");
    if (dirichlet_epsilon_ > 0.0 && dirichlet_alpha_ <= 0.0)
        throw std::invalid_argument("dirichlet_alpha must be > 0 when dirichlet_epsilon > 0");
}

void MCTS::set_root(const std::string& start_fen,
                    const std::vector<std::string>& history_moves) {
    pos_ = Position::from_uci_history(start_fen, history_moves);
    pool_.clear();
    root_ = pool_.add_node(/*parent=*/-1, /*move_index=*/-1, /*prior=*/0.0f);
    sims_run_ = 0;
    complete_ = pos_->outcome(/*claim_draw=*/true).has_value();
    noise_applied_ = false;
    pending_.clear();
    pending_offsets_.clear();
}

int MCTS::select_child(int node) const {
    const float sqrt_np = std::sqrt(static_cast<float>(pool_.n(node)));
    const int offset = pool_.child_offset(node);
    const int count = pool_.child_count(node);
    int best = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < count; ++k) {
        const int child = pool_.child(node, k);
        const std::int32_t n = pool_.n(child);
        // W is stored from the child's own (opponent's) perspective, so
        // negate it to get the value from `node`'s perspective.
        const float q = (n == 0) ? 0.0f : -pool_.w(child) / static_cast<float>(n);
        const float score =
            q + static_cast<float>(c_puct_) * pool_.prior(child) * sqrt_np /
                    (1.0f + static_cast<float>(n));
        if (score > best_score) {
            best_score = score;
            best = child;
        }
    }
    return best;
}

void MCTS::apply_virtual_loss(const std::vector<int>& path) {
    const std::int32_t vl = static_cast<std::int32_t>(virtual_loss_);
    for (const int node : path) {
        pool_.n(node) += vl;
        pool_.w(node) += static_cast<float>(virtual_loss_);
    }
}

void MCTS::backprop(const std::vector<int>& path, float value) {
    const std::int32_t vl = static_cast<std::int32_t>(virtual_loss_);
    const float vl_f = static_cast<float>(virtual_loss_);
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        const int node = *it;
        pool_.n(node) -= vl;  // undo virtual visits
        pool_.w(node) -= vl_f;  // undo virtual value
        pool_.n(node) += 1;     // real visit
        pool_.w(node) += value; // real value
        value = -value;
    }
}

void MCTS::apply_dirichlet_noise(int node) {
    if (dirichlet_epsilon_ <= 0.0) return;
    const int count = pool_.child_count(node);
    if (count == 0) return;
    std::vector<double> eta(static_cast<std::size_t>(count));
    std::gamma_distribution<double> gamma(dirichlet_alpha_, 1.0);
    double total = 0.0;
    for (int k = 0; k < count; ++k) {
        eta[static_cast<std::size_t>(k)] = gamma(rng_);
        total += eta[static_cast<std::size_t>(k)];
    }
    if (total <= 0.0) return;
    const float eps = static_cast<float>(dirichlet_epsilon_);
    for (int k = 0; k < count; ++k) {
        const int child = pool_.child(node, k);
        const float e = static_cast<float>(eta[static_cast<std::size_t>(k)] / total);
        pool_.prior(child) = (1.0f - eps) * pool_.prior(child) + eps * e;
    }
}

MCTS::GatherResult MCTS::gather_leaves(int max_batch) {
    GatherResult result;
    result.legal_offsets.push_back(0);
    if (complete_) {
        pending_.clear();
        pending_offsets_.clear();
        return result;
    }
    if (max_batch <= 0)
        throw std::invalid_argument("max_batch must be positive");

    pending_.clear();
    pending_offsets_.clear();

    // Root not yet evaluated: produce exactly the root leaf (mirrors
    // mcts.py's single root expansion forward pass and does not consume a
    // simulation). Otherwise run up to max_batch descents, bounded by the
    // remaining simulation budget.
    const int n = pool_.has_children(root_)
                      ? std::min(max_batch, num_simulations_ - sims_run_)
                      : 1;

    for (int i = 0; i < n; ++i) {
        // ---- descend to a leaf (one mutable board, no copies) ----
        std::vector<int> path;
        path.push_back(root_);
        int node = root_;
        while (pool_.has_children(node) && !pool_.is_terminal(node)) {
            const int child = select_child(node);
            pos_->push_uci(pos_->index_to_move(pool_.move_index(child)));
            node = child;
            path.push_back(node);
        }

        // ---- resolve the leaf: terminal / needs-eval ----
        bool is_terminal = pool_.is_terminal(node);
        if (!is_terminal) {
            // Claimable draws (threefold repetition, fifty-move claim) count
            // as terminal here so search, self-play and arena all agree.
            if (const auto outcome = pos_->outcome(/*claim_draw=*/true)) {
                pool_.set_terminal(node,
                                   terminal_value_from_outcome(*outcome, pos_->side_to_move()));
                is_terminal = true;
            }
        }
        const float leaf_value = is_terminal ? pool_.terminal_value(node) : 0.0f;

        // ---- unwind the mutable board back to the root ----
        for (std::size_t k = 1; k < path.size(); ++k) pos_->pop();

        // ---- virtual loss so concurrent simulations diverge ----
        // Applied AFTER the descent, exactly like mcts.py: the first sim of a
        // batch descends with no virtual loss anywhere, later sims see it.
        apply_virtual_loss(path);

        if (is_terminal) {
            backprop(path, leaf_value);
        } else {
            PendingLeaf leaf;
            leaf.node_id = node;
            leaf.path = std::move(path);
            pending_.push_back(std::move(leaf));
        }
    }

    if (pool_.has_children(root_)) sims_run_ += n;
    if (sims_run_ >= num_simulations_) complete_ = true;

    // ---- materialise leaves: replay paths, encode planes, collect legal
    //      move indices (CSR).  Fill pending_ in place (apply_evaluations
    //      reads pending_[t].legal), then copy into the returned result. ----
    for (std::size_t i = 0; i < pending_.size(); ++i) {
        PendingLeaf& leaf = pending_[i];
        for (std::size_t k = 1; k < leaf.path.size(); ++k) {
            pos_->push_uci(pos_->index_to_move(pool_.move_index(leaf.path[k])));
        }
        leaf.planes.resize(NUM_PLANES * SQUARES);
        pos_->encode_planes_u8(leaf.planes.data(), HISTORY_STEPS);
        leaf.legal = pos_->legal_move_indices();
        for (std::size_t k = 1; k < leaf.path.size(); ++k) pos_->pop();

        result.legal_indices.insert(result.legal_indices.end(),
                                    leaf.legal.begin(), leaf.legal.end());
        result.legal_offsets.push_back(
            static_cast<std::int32_t>(result.legal_indices.size()));
    }
    result.leaves = pending_;

    pending_offsets_ = result.legal_offsets;  // for apply_evaluations validation
    return result;
}

void MCTS::apply_evaluations(const std::vector<int>& tokens,
                             const std::vector<std::int32_t>& legal_offsets,
                             const std::vector<float>& legal_logits,
                             const std::vector<float>& values) {
    if (pending_.empty())
        throw std::invalid_argument(
            "apply_evaluations called without a pending gather_leaves");

    const std::size_t B = pending_.size();
    if (tokens.size() != B)
        throw std::invalid_argument(
            "tokens length must equal the number of leaves from gather_leaves");
    if (legal_offsets.size() != B + 1)
        throw std::invalid_argument("legal_offsets must have length B + 1");
    const std::size_t K = static_cast<std::size_t>(legal_offsets.back());
    if (legal_logits.size() != K)
        throw std::invalid_argument(
            "legal_logits length must equal legal_offsets[-1]");
    if (values.size() != B)
        throw std::invalid_argument(
            "values length must equal the number of leaves");

    // Validate every token and its CSR row boundaries BEFORE mutating any
    // state, so a malformed call leaves the search untouched.
    for (const int t : tokens) {
        if (t < 0 || static_cast<std::size_t>(t) >= B)
            throw std::invalid_argument("token out of range");
        if (legal_offsets[t] != pending_offsets_[t] ||
            legal_offsets[t + 1] != pending_offsets_[t + 1])
            throw std::invalid_argument(
                "legal_offsets row boundaries do not match gather_leaves");
    }

    const float vl_f = static_cast<float>(virtual_loss_);
    for (const int t : tokens) {
        const PendingLeaf& leaf = pending_[static_cast<std::size_t>(t)];
        const int row_begin = legal_offsets[t];
        const int row_len = legal_offsets[t + 1] - legal_offsets[t];
        if (row_len <= 0)
            throw std::invalid_argument(
                "a non-terminal leaf must have at least one legal move");

        // ---- softmax over the legal row (max-subtracted for stability) ----
        float max_logit = legal_logits[static_cast<std::size_t>(row_begin)];
        for (int k = 1; k < row_len; ++k)
            max_logit = std::max(
                max_logit,
                legal_logits[static_cast<std::size_t>(row_begin) + k]);
        std::vector<float> priors(static_cast<std::size_t>(row_len));
        double sum = 0.0;
        for (int k = 0; k < row_len; ++k) {
            priors[static_cast<std::size_t>(k)] = std::exp(
                legal_logits[static_cast<std::size_t>(row_begin) + k] - max_logit);
            sum += priors[static_cast<std::size_t>(k)];
        }
        if (sum > 0.0) {
            for (int k = 0; k < row_len; ++k)
                priors[static_cast<std::size_t>(k)] =
                    static_cast<float>(priors[static_cast<std::size_t>(k)] / sum);
        } else {
            const float uniform = 1.0f / static_cast<float>(row_len);
            for (int k = 0; k < row_len; ++k)
                priors[static_cast<std::size_t>(k)] = uniform;
        }

        // Defensive re-normalisation (mcts.py._make_children).
        double total = 0.0;
        for (int k = 0; k < row_len; ++k)
            total += priors[static_cast<std::size_t>(k)];
        if (total > 0.0) {
            for (int k = 0; k < row_len; ++k)
                priors[static_cast<std::size_t>(k)] =
                    static_cast<float>(priors[static_cast<std::size_t>(k)] / total);
        }

        // ---- expand: one child per legal move index, in CSR order ----
        std::vector<std::int32_t> kids(static_cast<std::size_t>(row_len));
        for (int k = 0; k < row_len; ++k) {
            kids[static_cast<std::size_t>(k)] = pool_.add_node(
                leaf.node_id, leaf.legal[static_cast<std::size_t>(k)],
                priors[static_cast<std::size_t>(k)]);
        }
        pool_.set_children(leaf.node_id, kids);

        // ---- root-only Dirichlet noise, once, right after root expansion ----
        if (leaf.node_id == root_ && !noise_applied_) {
            apply_dirichlet_noise(root_);
            noise_applied_ = true;
        }

        backprop(leaf.path, values[static_cast<std::size_t>(t)]);
    }

    pending_.clear();
    pending_offsets_.clear();
}

std::vector<std::pair<std::string, double>> MCTS::policy(double temperature) const {
    std::vector<std::pair<std::string, int>> pairs;
    if (pool_.has_children(root_)) {
        const int offset = pool_.child_offset(root_);
        const int count = pool_.child_count(root_);
        for (int k = 0; k < count; ++k) {
            const int child = pool_.child(root_, k);
            pairs.emplace_back(pos_->index_to_move(pool_.move_index(child)),
                               static_cast<int>(pool_.n(child)));
        }
    } else if (!complete_) {
        // Root never expanded (e.g. num_simulations == 0): uniform fallback
        // over the legal moves, matching mcts.py's zero-count branch.
        for (const std::string& uci : pos_->legal_moves_uci()) {
            pairs.emplace_back(uci, 0);
        }
    }
    // complete_ && !has_children: terminal root -> empty policy.

    std::vector<std::pair<std::string, double>> out;
    if (pairs.empty()) return out;

    if (temperature == 0.0) {
        // One-hot on the most visited move; ties go to the first child in
        // CSR order (np.argmax semantics).
        int best_count = pairs[0].second;
        std::size_t best = 0;
        for (std::size_t i = 1; i < pairs.size(); ++i) {
            if (pairs[i].second > best_count) {
                best_count = pairs[i].second;
                best = i;
            }
        }
        for (std::size_t i = 0; i < pairs.size(); ++i)
            out.emplace_back(pairs[i].first, i == best ? 1.0 : 0.0);
    } else {
        long long total = 0;
        for (const auto& p : pairs) total += p.second;
        if (total <= 0) {  // no visits at all -> uniform
            const double uniform = 1.0 / static_cast<double>(pairs.size());
            for (const auto& p : pairs) out.emplace_back(p.first, uniform);
        } else {
            const double inv_temp = 1.0 / temperature;
            std::vector<double> probs(pairs.size());
            double sum = 0.0;
            for (std::size_t i = 0; i < pairs.size(); ++i) {
                probs[i] = std::pow(static_cast<double>(pairs[i].second), inv_temp);
                sum += probs[i];
            }
            for (std::size_t i = 0; i < pairs.size(); ++i)
                out.emplace_back(pairs[i].first, probs[i] / sum);
        }
    }

    std::sort(out.begin(), out.end(),
              [](const std::pair<std::string, double>& a,
                 const std::pair<std::string, double>& b) { return a.first < b.first; });
    return out;
}

float MCTS::root_value() const {
    if (root_ < 0) return 0.0f;
    const std::int32_t n = pool_.n(root_);
    if (n <= 0) return 0.0f;
    return pool_.w(root_) / static_cast<float>(n);
}

std::vector<std::pair<std::string, int>> MCTS::root_visit_counts() const {
    std::vector<std::pair<std::string, int>> out;
    if (!pool_.has_children(root_)) return out;
    const int offset = pool_.child_offset(root_);
    const int count = pool_.child_count(root_);
    for (int k = 0; k < count; ++k) {
        const int child = pool_.child(root_, k);
        out.emplace_back(pos_->index_to_move(pool_.move_index(child)),
                         static_cast<int>(pool_.n(child)));
    }
    std::sort(out.begin(), out.end(),
              [](const std::pair<std::string, int>& a,
                 const std::pair<std::string, int>& b) { return a.first < b.first; });
    return out;
}

}  // namespace chess_rl_native
