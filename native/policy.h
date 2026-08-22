#pragma once

#include <string>
#include <string_view>

namespace chess_rl_native {

// AlphaZero 73-plane action map constants (see encoding.py).
inline constexpr int POLICY_PLANES = 73;
inline constexpr int POLICY_SIZE = 64 * POLICY_PLANES;  // 4672
inline constexpr int QUEEN_PLANES = 56;
inline constexpr int KNIGHT_PLANES = 8;
inline constexpr int UNDERPROMOTION_PLANES = 9;

// Map a UCI move string to its flat action index = from_square * 73 + plane.
// Pure function of the UCI string (from/to/promotion); a queen promotion rides
// the queen-like plane. Throws std::invalid_argument on a malformed UCI move.
int move_to_index(std::string_view uci);

// Invert an action index for the given side to move ("w" or "b").
//
// This is a *board-free* approximation: the 4672 index carries from-square +
// plane but not the piece type, so a queen-like plane landing on the back rank
// cannot be told from a queen promotion without the board. It infers "from is
// a pawn" from the from-square's rank (rank 7 for white, rank 2 for black),
// which is exact for every pawn promotion and every unambiguous move but wrong
// for a non-pawn piece stepping onto the back rank. Prefer
// Position::index_to_move(index), which consults the board.
std::string index_to_move(int index, std::string_view side_to_move);

// Exact inverse geometry shared by the board-free heuristic and the
// board-aware Position method. `from_is_pawn` tells whether the from square
// holds a pawn of the side to move (used only for the queen-promotion decision
// on the back rank).
std::string index_to_move_impl(int index, bool white, bool from_is_pawn);

}  // namespace chess_rl_native
