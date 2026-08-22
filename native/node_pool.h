#pragma once

#include <cstdint>
#include <vector>

namespace chess_rl_native {

// Compact structure-of-arrays node pool for the MCTS tree.
//
// Append-only: node ids are stable indices into flat vectors, so a node is
// never moved or invalidated once created. There is no per-node heap
// allocation and no std::map: the children of a node occupy the contiguous
// range [child_offset, child_offset + child_count) of one flat `children_`
// vector, which keeps selection cache-friendly.
//
// All values follow mcts.py conventions: W is stored from the node's own
// side-to-move perspective, N counts visits (real + virtual), and priors are
// normalised over the node's legal moves.
class NodePool {
  public:
    // Creates a node and returns its id. `parent` == -1 for the root,
    // `move_index` == -1 for the root.
    int add_node(int parent, std::int32_t move_index, float prior);

    // Drops every node (used by MCTS::set_root for a fresh search).
    void clear();

    [[nodiscard]] int size() const { return static_cast<int>(parent_.size()); }

    [[nodiscard]] int parent(int id) const { return parent_[id]; }
    [[nodiscard]] std::int32_t move_index(int id) const { return move_index_[id]; }
    [[nodiscard]] float prior(int id) const { return prior_[id]; }
    float& prior(int id) { return prior_[id]; }

    [[nodiscard]] float w(int id) const { return w_[id]; }
    float& w(int id) { return w_[id]; }
    [[nodiscard]] std::int32_t n(int id) const { return n_[id]; }
    std::int32_t& n(int id) { return n_[id]; }

    [[nodiscard]] bool is_terminal(int id) const { return terminal_[id] != 0; }
    [[nodiscard]] float terminal_value(int id) const { return terminal_value_[id]; }
    void set_terminal(int id, float value) {
        terminal_[id] = 1;
        terminal_value_[id] = value;
    }

    [[nodiscard]] bool has_children(int id) const { return child_count_[id] > 0; }
    [[nodiscard]] int child_offset(int id) const { return child_offset_[id]; }
    [[nodiscard]] int child_count(int id) const { return child_count_[id]; }
    [[nodiscard]] std::int32_t child(int id, int k) const {
        return children_[static_cast<std::size_t>(child_offset_[id]) + k];
    }

    // (Re)assigns the child range of `id`. The first expansion appends a new
    // range; a re-expansion of the same node (two simulations reaching the
    // same unexpanded leaf in one batch) overwrites the existing range in
    // place so the flat layout stays compact.
    void set_children(int id, const std::vector<std::int32_t>& kids);

  private:
    std::vector<int> parent_;
    std::vector<std::int32_t> move_index_;
    std::vector<float> prior_;
    std::vector<float> w_;
    std::vector<std::int32_t> n_;
    std::vector<std::int8_t> terminal_;          // 0 = unknown, 1 = terminal
    std::vector<float> terminal_value_;          // meaningful only when terminal
    std::vector<int> child_offset_;              // -1 until first expansion
    std::vector<int> child_count_;
    std::vector<std::int32_t> children_;         // flat, per-node contiguous runs
};

}  // namespace chess_rl_native
