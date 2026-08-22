#include "node_pool.h"

#include <algorithm>

namespace chess_rl_native {

int NodePool::add_node(int parent, std::int32_t move_index, float prior) {
    const int id = static_cast<int>(parent_.size());
    parent_.push_back(parent);
    move_index_.push_back(move_index);
    prior_.push_back(prior);
    w_.push_back(0.0f);
    n_.push_back(0);
    terminal_.push_back(0);
    terminal_value_.push_back(0.0f);
    child_offset_.push_back(-1);
    child_count_.push_back(0);
    return id;
}

void NodePool::clear() {
    parent_.clear();
    move_index_.clear();
    prior_.clear();
    w_.clear();
    n_.clear();
    terminal_.clear();
    terminal_value_.clear();
    child_offset_.clear();
    child_count_.clear();
    children_.clear();
}

void NodePool::set_children(int id, const std::vector<std::int32_t>& kids) {
    if (child_offset_[id] < 0) {
        child_offset_[id] = static_cast<int>(children_.size());
        children_.insert(children_.end(), kids.begin(), kids.end());
    } else {
        std::copy(kids.begin(), kids.end(),
                  children_.begin() + child_offset_[id]);
    }
    child_count_[id] = static_cast<int>(kids.size());
}

}  // namespace chess_rl_native
