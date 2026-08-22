#include "policy.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace chess_rl_native {
namespace {

struct Direction {
    int dr;
    int df;
};

// (dr, df) deltas, rank 0 = white's back rank; the order IS the plane order
// (mirrors encoding.py QUEEN_DIRECTIONS).
constexpr Direction QUEEN_DIRECTIONS[8] = {
    {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1},
};

// Knight offsets, the order IS the plane order (mirrors encoding.py
// KNIGHT_OFFSETS).
constexpr Direction KNIGHT_OFFSETS[8] = {
    {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
};

// Underpromotion pieces in plane order: N=0, B=1, R=2.
constexpr char UNDER_PROMOTION_CHARS[3] = {'n', 'b', 'r'};

int sign(int value) { return (value > 0) - (value < 0); }

// Parse a UCI move string into from/to squares (0..63, a1=0..h8=63) and the
// promotion char ('q','r','b','n' or '\0'). Returns false if malformed.
bool parse_uci(std::string_view uci, int& from, int& to, char& promo) {
    if (uci.size() != 4 && uci.size() != 5) return false;
    for (int i = 0; i < 4; ++i) {
        const char c = uci[i];
        if (i % 2 == 0) {
            if (c < 'a' || c > 'h') return false;
        } else {
            if (c < '1' || c > '8') return false;
        }
    }
    promo = '\0';
    if (uci.size() == 5) {
        const char p = uci[4];
        if (p != 'q' && p != 'r' && p != 'b' && p != 'n') return false;
        promo = p;
    }
    from = (uci[1] - '1') * 8 + (uci[0] - 'a');
    to = (uci[3] - '1') * 8 + (uci[2] - 'a');
    return true;
}

int queen_direction_index(int dr, int df) {
    const int drs = sign(dr);
    const int dfs = sign(df);
    for (int i = 0; i < 8; ++i) {
        if (QUEEN_DIRECTIONS[i].dr == drs && QUEEN_DIRECTIONS[i].df == dfs) return i;
    }
    return -1;
}

}  // namespace

int move_to_index(std::string_view uci) {
    int from = 0;
    int to = 0;
    char promo = '\0';
    if (!parse_uci(uci, from, to, promo)) throw std::invalid_argument("malformed UCI move");

    const int fr = from / 8;
    const int ff = from % 8;
    const int tr = to / 8;
    const int tf = to % 8;
    const int dr = tr - fr;
    const int df = tf - ff;

    if (promo != '\0' && promo != 'q') {
        // Underpromotion: piece (N=0, B=1, R=2) * 3 + (df + 1).
        const int piece = (promo == 'n') ? 0 : (promo == 'b') ? 1 : 2;
        const int plane = QUEEN_PLANES + KNIGHT_PLANES + piece * 3 + (df + 1);
        return from * POLICY_PLANES + plane;
    }

    for (int ki = 0; ki < 8; ++ki) {
        if (KNIGHT_OFFSETS[ki].dr == dr && KNIGHT_OFFSETS[ki].df == df) {
            return from * POLICY_PLANES + QUEEN_PLANES + ki;
        }
    }

    const int dist = std::max(std::abs(dr), std::abs(df));
    const int di = queen_direction_index(dr, df);
    if (di < 0) throw std::invalid_argument("UCI move is not a valid queen-like move");
    const int plane = di * 7 + (dist - 1);
    return from * POLICY_PLANES + plane;
}

std::string index_to_move_impl(int index, bool white, bool from_is_pawn) {
    if (index < 0 || index >= POLICY_SIZE) throw std::invalid_argument("action index out of range");

    const int from = index / POLICY_PLANES;
    const int plane = index % POLICY_PLANES;
    const int fr = from / 8;
    const int ff = from % 8;

    int to = 0;
    char promo = '\0';
    if (plane < QUEEN_PLANES) {
        const int di = plane / 7;
        const int dist = plane % 7 + 1;
        const int dr = QUEEN_DIRECTIONS[di].dr;
        const int df = QUEEN_DIRECTIONS[di].df;
        to = from + (dr * 8 + df) * dist;
        const int tr = to / 8;
        // A PAWN of the side to move landing on its promotion back rank is a
        // queen promotion (queen promotions ride the queen-like plane).
        if (from_is_pawn && (white ? tr == 7 : tr == 0)) {
            promo = 'q';
        }
    } else if (plane < QUEEN_PLANES + KNIGHT_PLANES) {
        const int dr = KNIGHT_OFFSETS[plane - QUEEN_PLANES].dr;
        const int df = KNIGHT_OFFSETS[plane - QUEEN_PLANES].df;
        to = from + dr * 8 + df;
    } else {
        const int p = plane - QUEEN_PLANES - KNIGHT_PLANES;
        const int piece = p / 3;
        const int df = p % 3 - 1;
        const int tr = white ? 7 : 0;
        const int tf = ff + df;
        to = tr * 8 + tf;
        promo = UNDER_PROMOTION_CHARS[piece];
    }

    std::string result;
    result.reserve(5);
    result += static_cast<char>('a' + ff);
    result += static_cast<char>('1' + fr);
    result += static_cast<char>('a' + (to % 8));
    result += static_cast<char>('1' + (to / 8));
    if (promo != '\0') result += promo;
    return result;
}

std::string index_to_move(int index, std::string_view side_to_move) {
    const bool white = (side_to_move == "w");
    if (!white && side_to_move != "b") throw std::invalid_argument("invalid side to move");
    if (index < 0 || index >= POLICY_SIZE) throw std::invalid_argument("action index out of range");
    // Board-free heuristic: infer "from is a pawn" from the from-square's rank.
    const int fr = (index / POLICY_PLANES) / 8;
    const bool from_is_pawn = white ? (fr == 6) : (fr == 1);
    return index_to_move_impl(index, white, from_is_pawn);
}

}  // namespace chess_rl_native
