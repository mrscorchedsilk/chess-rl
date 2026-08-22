#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "chess.hpp"

namespace chess_rl_native {

struct Outcome {
    // Empty denotes a draw. Python bindings expose it as None.
    std::string winner;
    std::string termination;
};

class Position {
  public:
    static Position from_fen(std::string_view fen);
    static Position from_uci_history(std::string_view start_fen, const std::vector<std::string>& moves);

    [[nodiscard]] std::string fen() const;
    [[nodiscard]] std::vector<std::string> legal_moves_uci() const;
    [[nodiscard]] std::vector<std::string> history_uci() const;
    [[nodiscard]] std::vector<std::string> history_fens(int max_steps = 8) const;
    [[nodiscard]] bool is_repetition(int count) const;
    [[nodiscard]] std::optional<Outcome> outcome(bool claim_draw = true) const;
    void push_uci(std::string_view uci);
    void pop();

    [[nodiscard]] std::string side_to_move() const;
    [[nodiscard]] std::uint32_t halfmove_clock() const noexcept;
    [[nodiscard]] std::uint32_t fullmove_number() const noexcept;
    [[nodiscard]] std::string castling_rights() const;
    [[nodiscard]] std::string ep_square() const;

  private:
    Position() = default;

    chess::Board board_;
    std::vector<chess::Move> history_;
    std::vector<std::string> history_uci_;

    friend std::uint64_t perft(std::string_view fen, int depth);
};

std::uint64_t perft(std::string_view fen, int depth);

}  // namespace chess_rl_native
