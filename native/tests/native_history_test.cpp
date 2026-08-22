#include "position.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr auto START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr auto MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 150 3";

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
    try {
        const std::vector<std::string> cycle{"g1f3", "g8f6", "f3g1", "f6g8"};
        auto twofold = chess_rl_native::Position::from_uci_history(START_FEN, cycle);
        require(twofold.history_uci() == cycle, "UCI history was not preserved");
        require(twofold.history_fens(8).size() == 5, "history FEN count is wrong");
        require(twofold.is_repetition(1), "current position must count as one occurrence");
        require(twofold.is_repetition(2), "twofold repetition was not detected");
        require(!twofold.is_repetition(3), "twofold must not be threefold");

        const std::vector<std::string> threefold_moves{
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"};
        auto threefold = chess_rl_native::Position::from_uci_history(START_FEN, threefold_moves);
        const auto threefold_outcome = threefold.outcome(true);
        require(threefold_outcome.has_value(), "claimable threefold must have an outcome");
        require(threefold_outcome->winner.empty(), "threefold winner must be a draw");
        require(threefold_outcome->termination == "threefold_repetition",
                "threefold termination is wrong");

        auto mate = chess_rl_native::Position::from_fen(MATE_FEN);
        const auto mate_outcome = mate.outcome(false);
        require(mate_outcome.has_value(), "checkmate must have an outcome");
        require(mate_outcome->winner == "black", "checkmate winner is wrong");
        require(mate_outcome->termination == "checkmate",
                "checkmate must outrank seventy-five moves");

        bool rejected = false;
        try {
            chess_rl_native::Position::from_uci_history(START_FEN, {"e2e5"});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "illegal complete history was accepted");
        std::cout << "native history and outcome assertions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
