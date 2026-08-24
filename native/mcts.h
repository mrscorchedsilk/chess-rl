#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "node_pool.h"
#include "position.h"

namespace chess_rl_native {

// Cache-friendly PUCT MCTS core (AlphaZero style), semantics-for-semantics
// equivalent to mcts.py:
//
//  - ONE mutable Position for the whole search (no board copies while
//    descending; leaves are materialised only when gather_leaves encodes
//    them),
//  - batched leaf evaluation via gather_leaves / apply_evaluations with
//    virtual loss so in-flight simulations diverge,
//  - compact SoA node pool (see node_pool.h): no per-node heap allocation,
//    no std::map,
//  - root-only Dirichlet noise from a std::mt19937 seeded at construction.
//
// Simulation budget: the root is evaluated by the first gather_leaves call
// (returned as the single leaf of that gather, mirroring mcts.py's root
// expansion forward pass); afterwards exactly `num_simulations` descents are
// run, and is_complete() turns true once they are exhausted. A terminal root
// (checkmate / stalemate / ...) completes immediately with an empty policy.
class MCTS {
  public:
    MCTS(double c_puct, double virtual_loss, int num_simulations,
         double dirichlet_alpha, double dirichlet_epsilon, std::uint64_t seed);

    // Starts a fresh search from the given position.
    void set_root(const std::string& start_fen,
                  const std::vector<std::string>& history_moves);

    struct PendingLeaf {
        int node_id;                       // pool id of the leaf node
        std::vector<int> path;             // node ids root -> leaf, inclusive
        std::vector<std::int32_t> legal;   // legal move indices at the leaf
        // Compact 104 * 64 encoded planes (see Position::encode_planes_u8).
        // Binary planes hold 0/1; the halfmove plane holds the raw clock and
        // is divided by Position::HALFMOVE_SCALE by the consumer.
        std::vector<std::uint8_t> planes;
    };

    struct GatherResult {
        std::vector<PendingLeaf> leaves;
        // CSR over leaves: legal_offsets[i]..legal_offsets[i+1] indexes
        // legal_indices for leaf i; legal_offsets has leaves.size() + 1
        // entries and always starts with 0 (even when leaves is empty).
        std::vector<std::int32_t> legal_offsets;
        std::vector<std::int32_t> legal_indices;
    };

    // Runs up to `max_batch` simulations (bounded by the remaining budget).
    // If the root is not yet expanded, exactly one leaf is produced: the root
    // itself. Returns empty once the search is complete.
    [[nodiscard]] GatherResult gather_leaves(int max_batch);

    // Applies network logits/values to the leaves gathered by the previous
    // gather_leaves call: expands each leaf (softmax priors over its legal
    // row, then mcts.py's defensive re-normalisation), applies root-only
    // Dirichlet noise right after the root expansion, and backprops the value
    // along the leaf's path (removing virtual loss first, flipping the sign
    // at every level). `tokens` must be the token list returned by
    // gather_leaves (each token is an index into that gather's pending list).
    // Every array must have the exact sizes returned by gather_leaves;
    // anything else throws std::invalid_argument before any state changes.
    void apply_evaluations(const std::vector<int>& tokens,
                           const std::vector<std::int32_t>& legal_offsets,
                           const std::vector<float>& legal_logits,
                           const std::vector<float>& values);

    [[nodiscard]] bool is_complete() const { return complete_; }

    // Visit-count distribution over the root's legal moves, temperature
    // adjusted: temperature == 0.0 -> one-hot on the most visited move;
    // otherwise counts^(1/temperature), normalised (uniform when no visits).
    // Sorted by UCI. Empty for a terminal root.
    [[nodiscard]] std::vector<std::pair<std::string, double>> policy(
        double temperature) const;

    // Diagnostic: [(uci, visit_count)] over the root's legal moves, sorted by
    // UCI. Not part of the pinned contract; used by the native tests.
    [[nodiscard]] std::vector<std::pair<std::string, int>> root_visit_counts() const;

  private:
    // Id of the child of `node` maximising
    //   -W/N + c_puct * P * sqrt(N_parent) / (1 + N)
    // (W is child-perspective, hence the negation). Ties go to the first
    // child in CSR order (ascending action index) via strict `>`.
    [[nodiscard]] int select_child(int node) const;

    void apply_virtual_loss(const std::vector<int>& path);
    void backprop(const std::vector<int>& path, float value);
    void apply_dirichlet_noise(int node);

    double c_puct_;
    double virtual_loss_;
    int num_simulations_;
    double dirichlet_alpha_;
    double dirichlet_epsilon_;
    std::mt19937 rng_;

    NodePool pool_;
    // Single mutable board; parked at the root between calls. Held in an
    // optional because Position's default constructor is private (MCTS is
    // not a friend) — assignment from from_uci_history works fine.
    std::optional<Position> pos_;
    int root_ = -1;
    int sims_run_ = 0;        // descent simulations consumed
    bool complete_ = false;
    bool noise_applied_ = false;

    // State of the last gather_leaves, consumed by the next apply_evaluations.
    std::vector<PendingLeaf> pending_;
    std::vector<std::int32_t> pending_offsets_;  // size pending_.size() + 1
};

}  // namespace chess_rl_native
