#include "actor.h"

#include <algorithm>
#include <stdexcept>

namespace chess_rl_native {
namespace {

constexpr int HISTORY_STEPS = 8;
constexpr int NUM_PLANES = 12 * HISTORY_STEPS + 8;  // 104
constexpr int SQUARES = 64;
constexpr int POLICY_BINS = POLICY_SIZE;  // 4672

const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Deterministic per-game seed derivation (splitmix64 finaliser).
std::uint64_t derive_seed(std::uint64_t base, std::uint64_t index) {
    std::uint64_t x = base + index + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

}  // namespace

Actor::GameState::GameState(double c_puct, double virtual_loss,
                            int num_simulations, double dirichlet_alpha,
                            double dirichlet_epsilon, std::uint64_t seed)
    : mcts(c_puct, virtual_loss, num_simulations, dirichlet_alpha,
           dirichlet_epsilon, seed),
      rng(seed) {}

Actor::Actor(int games, double c_puct, double virtual_loss,
             int num_simulations, double temperature,
             int temperature_threshold, int max_game_length,
             std::uint64_t seed, int num_threads)
    : temperature_(temperature),
      temperature_threshold_(temperature_threshold),
      max_game_length_(max_game_length),
      num_threads_(ThreadPool::clamp_threads(
          std::min(num_threads > 0 ? num_threads : games, games))) {
    if (games <= 0) throw std::invalid_argument("games must be positive");
    if (num_simulations < 0)
        throw std::invalid_argument("num_simulations must be >= 0");
    if (c_puct < 0.0) throw std::invalid_argument("c_puct must be >= 0");
    if (virtual_loss < 0.0)
        throw std::invalid_argument("virtual_loss must be >= 0");
    if (temperature < 0.0)
        throw std::invalid_argument("temperature must be >= 0");
    if (temperature_threshold < 0)
        throw std::invalid_argument("temperature_threshold must be >= 0");
    if (max_game_length <= 0)
        throw std::invalid_argument("max_game_length must be positive");

    games_.reserve(static_cast<std::size_t>(games));
    for (int g = 0; g < games; ++g) {
        const std::uint64_t game_seed = derive_seed(seed, static_cast<std::uint64_t>(g));
        // Dirichlet noise matches mcts.py defaults; the per-game seed keeps
        // every game's noise stream deterministic given the actor seed.
        games_.emplace_back(c_puct, virtual_loss, num_simulations,
                            /*dirichlet_alpha=*/0.3,
                            /*dirichlet_epsilon=*/0.25, game_seed);
        GameState& game = games_.back();
        game.pos = Position::from_uci_history(START_FEN, {});
        game.mcts.set_root(START_FEN, {});
    }

    // One persistent pool for the actor's whole lifetime.  Created after the
    // games so a constructor throw above never leaves workers running.
    if (num_threads_ > 1)
        pool_ = std::make_unique<ThreadPool>(num_threads_);
}

void Actor::parallel_for(const std::function<void(int)>& fn) {
    const int n = static_cast<int>(games_.size());
    if (!pool_) {
        for (int g = 0; g < n; ++g) fn(g);
        return;
    }
    pool_->parallel_for(n, fn);
}

void Actor::set_teacher(int weight_version, int generation) {
    teacher_weight_version_ = weight_version;
    teacher_generation_ = generation;
}

Actor::GatherResult Actor::gather_leaves(int max_batch) {
    if (max_batch <= 0)
        throw std::invalid_argument("max_batch must be positive");

    GatherResult result;
    result.legal_offsets.push_back(0);
    leaf_game_.clear();
    game_base_.assign(games_.size() + 1, 0);

    // `max_batch` is the TOTAL merged batch budget (the GPU runtime's fixed
    // buckets cap it at 256).  Each in-play game gets an equal share so the
    // merged total can never exceed max_batch.  Each game's MCTS is gathered
    // EXACTLY ONCE per round (its pending list is consumed by the matching
    // apply_evaluations), and its leaves stay contiguous so the game_base_
    // routing in apply_evaluations holds.
    int in_play = 0;
    for (const GameState& game : games_)
        if (!game.finished && !game.mcts.is_complete()) ++in_play;
    if (in_play == 0) {
        game_base_.assign(games_.size() + 1, 0);
        return result;
    }
    const int per_game = std::max(1, max_batch / in_play);

    // Gather each in-play game's leaves in parallel (games are fully
    // independent: each owns its MCTS, board and RNG), then merge serially in
    // game order so the merged CSR and leaf->game routing stay deterministic.
    const std::size_t G = games_.size();
    std::vector<MCTS::GatherResult> local(G);
    parallel_for([&](int g) {
        GameState& game = games_[static_cast<std::size_t>(g)];
        if (game.finished || game.mcts.is_complete()) return;
        local[static_cast<std::size_t>(g)] =
            game.mcts.gather_leaves(per_game);
    });

    for (std::size_t g = 0; g < G; ++g) {
        game_base_[g] = static_cast<int>(result.leaves.size());
        MCTS::GatherResult& gr = local[g];
        for (auto& leaf : gr.leaves) {
            leaf_game_.push_back(static_cast<int>(g));
            result.leaves.push_back(std::move(leaf));
        }
        result.legal_indices.insert(result.legal_indices.end(),
                                    gr.legal_indices.begin(),
                                    gr.legal_indices.end());
        // gr.legal_offsets[0] == 0 and each entry is cumulative within the
        // game's own block; shift by the merged total so far so the merged
        // CSR keeps accumulating across games.
        const std::int32_t block_base = result.legal_offsets.back();
        for (std::size_t i = 1; i < gr.legal_offsets.size(); ++i)
            result.legal_offsets.push_back(block_base + gr.legal_offsets[i]);
    }
    game_base_[G] = static_cast<int>(result.leaves.size());
    return result;
}

void Actor::apply_evaluations(const std::vector<int>& tokens,
                              const std::vector<std::int32_t>& legal_offsets,
                              const std::vector<float>& legal_logits,
                              const std::vector<float>& values) {
    const std::size_t B = leaf_game_.size();
    if (B == 0)
        throw std::invalid_argument(
            "apply_evaluations called without a pending gather_leaves");
    if (tokens.size() != B)
        throw std::invalid_argument(
            "tokens length must equal the number of leaves from gather_leaves");
    if (legal_offsets.size() != B + 1)
        throw std::invalid_argument("legal_offsets must have length B + 1");
    if (legal_offsets.empty() || legal_offsets[0] != 0)
        throw std::invalid_argument("legal_offsets must start with 0");
    const std::size_t K = static_cast<std::size_t>(legal_offsets.back());
    if (legal_logits.size() != K)
        throw std::invalid_argument(
            "legal_logits length must equal legal_offsets[-1]");
    if (values.size() != B)
        throw std::invalid_argument(
            "values length must equal the number of leaves");

    // Validate every token and the CSR monotonicity BEFORE routing, so a
    // malformed call leaves every game's search untouched.
    for (const int t : tokens) {
        if (t < 0 || static_cast<std::size_t>(t) >= B)
            throw std::invalid_argument("token out of range");
    }
    for (std::size_t i = 0; i + 1 < legal_offsets.size(); ++i) {
        if (legal_offsets[i] > legal_offsets[i + 1])
            throw std::invalid_argument("legal_offsets must be non-decreasing");
    }

    // Per-game local token lists (merged position -> local leaf index).
    std::vector<std::vector<int>> game_tokens(games_.size());
    for (const int t : tokens) {
        const int g = leaf_game_[static_cast<std::size_t>(t)];
        game_tokens[static_cast<std::size_t>(g)].push_back(
            t - game_base_[static_cast<std::size_t>(g)]);
    }

    // Build each game's own CSR slice (serial, cheap), then apply evaluations
    // in parallel (each game's MCTS.apply_evaluations is independent).
    struct GameSlice {
        std::vector<int> tokens;
        std::vector<std::int32_t> offsets;
        std::vector<float> logits;
        std::vector<float> values;
    };
    const std::size_t G = games_.size();
    std::vector<GameSlice> slices(G);
    for (std::size_t g = 0; g < G; ++g) {
        const int base = game_base_[g];
        const int len = game_base_[g + 1] - base;
        GameSlice& s = slices[g];
        if (len == 0) continue;
        s.tokens = game_tokens[g];

        // Reconstruct the game's own CSR row pointers: its gather's offsets
        // are the merged block shifted by the block's cumulative offset.
        s.offsets.reserve(static_cast<std::size_t>(len) + 1);
        for (int i = 0; i <= len; ++i) {
            s.offsets.push_back(
                legal_offsets[static_cast<std::size_t>(base + i)] -
                legal_offsets[static_cast<std::size_t>(base)]);
        }
        const std::int32_t row_begin = legal_offsets[static_cast<std::size_t>(base)];
        const std::int32_t row_end =
            legal_offsets[static_cast<std::size_t>(base + len)];
        s.logits.assign(legal_logits.begin() + row_begin,
                        legal_logits.begin() + row_end);
        s.values.assign(values.begin() + base, values.begin() + base + len);
    }

    parallel_for([&](int g) {
        GameSlice& s = slices[static_cast<std::size_t>(g)];
        if (s.offsets.empty()) return;
        games_[static_cast<std::size_t>(g)].mcts.apply_evaluations(
            s.tokens, s.offsets, s.logits, s.values);
    });

    leaf_game_.clear();
    game_base_.clear();
}

std::string Actor::sample_move(
    const std::vector<std::pair<std::string, double>>& policy,
    double temperature, std::mt19937& rng) {
    if (policy.empty())
        throw std::runtime_error("cannot sample a move from an empty policy");
    if (temperature == 0.0) {
        // MCTS::policy(0.0) is one-hot on the most visited move.
        for (const auto& entry : policy) {
            if (entry.second > 0.5) return entry.first;
        }
        throw std::runtime_error("policy(0.0) did not produce a one-hot move");
    }
    std::vector<double> probs;
    probs.reserve(policy.size());
    for (const auto& entry : policy) probs.push_back(entry.second);
    double total = 0.0;
    for (const double p : probs) total += p;
    if (total <= 0.0)
        throw std::runtime_error("policy has no probability mass");
    for (double& p : probs) p /= total;  // defensive re-normalisation
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return policy[static_cast<std::size_t>(dist(rng))].first;
}

void Actor::finalise(int game_index, GameState& game, float white_result,
                     std::string termination) {
    Game done;
    done.teacher_generation = teacher_generation_;
    done.teacher_weight_version = teacher_weight_version_;
    done.result_termination = std::move(termination);
    done.game_index = game_index;
    done.examples.reserve(game.examples.size());
    for (Example& ex : game.examples) {
        // z-value convention (selfplay.py): from the side-to-move's view.
        ex.z = (ex.side == "w") ? white_result : -white_result;
        done.examples.push_back(std::move(ex));
    }
    game.examples.clear();
    game.finished = true;
    std::lock_guard<std::mutex> lock(finished_mutex_);
    finished_.push_back(std::move(done));
}

void Actor::advance() {
    parallel_for([&](int g) {
        GameState& game = games_[static_cast<std::size_t>(g)];
        if (game.finished || !game.mcts.is_complete()) return;

        const int ply = static_cast<int>(game.pos->history_uci().size());
        const double temp =
            (ply < temperature_threshold_) ? temperature_ : 0.0;
        const auto policy = game.mcts.policy(temp);

        if (policy.empty()) {
            // Search completed with no moves: the root position is terminal
            // (checkmate / stalemate / claimable draw).
            const auto outcome = game.pos->outcome(/*claim_draw=*/true);
            if (!outcome)
                throw std::runtime_error(
                    "complete search with an empty policy for a "
                    "non-terminal position");
            const float white_result =
                outcome->winner.empty() ? 0.0f
                : (outcome->winner == "w" ? 1.0f : -1.0f);
            finalise(g, game, white_result, outcome->termination);
            return;
        }

        // Record the training example BEFORE the move is played.
        Example ex;
        ex.state.resize(static_cast<std::size_t>(NUM_PLANES * SQUARES));
        game.pos->encode_planes(ex.state.data(), HISTORY_STEPS);
        ex.pi.assign(static_cast<std::size_t>(POLICY_BINS), 0.0f);
        for (const auto& entry : policy) {
            ex.pi[static_cast<std::size_t>(move_to_index(entry.first))] +=
                static_cast<float>(entry.second);
        }
        ex.side = game.pos->side_to_move();
        game.examples.push_back(std::move(ex));

        const std::string move_uci = sample_move(policy, temp, game.rng);
        game.pos->push_uci(move_uci);

        // Terminal check mirrors selfplay.py: outcome(claim_draw=true) covers
        // checkmate, stalemate, insufficient material, the automatic rules and
        // the claimable draws; the length cap is a draw.
        const int new_ply = static_cast<int>(game.pos->history_uci().size());
        if (new_ply >= max_game_length_) {
            finalise(g, game, 0.0f, "length_cap");
            return;
        }
        if (const auto outcome = game.pos->outcome(/*claim_draw=*/true)) {
            const float white_result =
                outcome->winner.empty() ? 0.0f
                : (outcome->winner == "w" ? 1.0f : -1.0f);
            finalise(g, game, white_result, outcome->termination);
            return;
        }

        // Fresh search on the new position: rebuild from the standard start
        // FEN plus the full history (from_uci_history replays the moves, so
        // passing the CURRENT fen would replay them on top of itself).
        game.mcts.set_root(START_FEN, game.pos->history_uci());
    });
}

std::vector<Game> Actor::finished_games() {
    std::vector<Game> out;
    out.swap(finished_);
    // Parallel finalise() pushes games in completion order; sort back to game
    // order so same-seed runs are byte-identical (determinism contract).
    std::sort(out.begin(), out.end(), [](const Game& a, const Game& b) {
        return a.game_index < b.game_index;
    });
    return out;
}

bool Actor::is_done() const {
    for (const GameState& game : games_) {
        if (!game.finished) return false;
    }
    return true;
}

int Actor::games_remaining() const {
    int count = 0;
    for (const GameState& game : games_) {
        if (!game.finished) ++count;
    }
    return count;
}

}  // namespace chess_rl_native
