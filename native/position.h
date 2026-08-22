#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "chess.hpp"

namespace chess_rl_native {

class Position {
  public:
    static Position from_fen(std::string_view fen);

    [[nodiscard]] std::string fen() const;
    [[nodiscard]] std::vector<std::string> legal_moves_uci() const;
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

    friend std::uint64_t perft(std::string_view fen, int depth);
};

std::uint64_t perft(std::string_view fen, int depth);

}  // namespace chess_rl_native
