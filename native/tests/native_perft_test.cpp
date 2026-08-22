#include "position.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr auto START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr auto KIWIPETE_FEN =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_perft(const char* name, const char* fen, const std::vector<std::uint64_t>& expected,
                 int first_depth) {
    for (int depth = first_depth; depth <= 5; ++depth) {
        const auto actual = chess_rl_native::perft(fen, depth);
        require(actual == expected[static_cast<std::size_t>(depth)],
                std::string(name) + " perft depth " + std::to_string(depth) + ": expected " +
                    std::to_string(expected[static_cast<std::size_t>(depth)]) + ", got " + std::to_string(actual));
    }
}
}  // namespace

int main() {
    try {
        check_perft("start", START_FEN, {1, 20, 400, 8902, 197281, 4865609}, 0);
        check_perft("kiwipete", KIWIPETE_FEN, {0, 48, 2039, 97862, 4085603, 193690690}, 1);

        auto position = chess_rl_native::Position::from_fen(START_FEN);
        require(position.fen() == START_FEN, "start FEN round trip failed");
        require(position.side_to_move() == "w", "side-to-move adapter failed");
        require(position.halfmove_clock() == 0, "halfmove adapter failed");
        require(position.fullmove_number() == 1, "fullmove adapter failed");
        require(position.castling_rights() == "KQkq", "castling adapter failed");
        require(position.ep_square() == "-", "en-passant adapter failed");
        require(position.legal_moves_uci().size() == 20, "legal move adapter failed");
        position.push_uci("e2e4");
        require(position.side_to_move() == "b", "push side-to-move failed");
        position.pop();
        require(position.fen() == START_FEN, "pop did not restore start position");

        bool rejected = false;
        try {
            position.push_uci("e2e5");
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "illegal UCI move was accepted");
        std::cout << "native perft and Position assertions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
