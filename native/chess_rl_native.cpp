#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chess.hpp"
#include "position.h"

#if __cplusplus < 201703L
#error "chess_rl_native requires C++17 or newer"
#endif

namespace py = pybind11;

namespace {
py::object outcome_to_python(const std::optional<chess_rl_native::Outcome>& outcome) {
    if (!outcome) return py::none();
    py::dict result;
    if (outcome->winner.empty())
        result["winner"] = py::none();
    else
        result["winner"] = py::str(outcome->winner);
    result["termination"] = outcome->termination;
    return std::move(result);
}

py::dict build_info() {
    py::dict info;
    info["cxx_standard"] = "c++17";
    info["cplusplus"] = static_cast<long long>(__cplusplus);
    info["compiler"] = __VERSION__;
    info["native_abi_version"] = CHESS_RL_NATIVE_ABI_VERSION;
    info["chess_library_commit"] = CHESS_RL_NATIVE_CHESS_LIBRARY_COMMIT;
    info["chess_library_header_sha256"] = CHESS_RL_NATIVE_CHESS_LIBRARY_HEADER_SHA256;
    return info;
}
}  // namespace

PYBIND11_MODULE(_chess_rl_native, module) {
    module.doc() = "Native chess position adapter and perft for chess-rl";
    module.def("native_abi_version", [] { return CHESS_RL_NATIVE_ABI_VERSION; });
    module.def("chess_library_commit", [] { return CHESS_RL_NATIVE_CHESS_LIBRARY_COMMIT; });
    module.def("chess_library_header_sha256", [] { return CHESS_RL_NATIVE_CHESS_LIBRARY_HEADER_SHA256; });
    module.def("build_info", &build_info);

    py::class_<chess_rl_native::Position>(module, "Position")
        .def_static("from_fen", &chess_rl_native::Position::from_fen, py::arg("fen"))
        .def_static("from_uci_history", &chess_rl_native::Position::from_uci_history,
                    py::arg("start_fen"), py::arg("moves"))
        .def("fen", &chess_rl_native::Position::fen)
        .def("legal_moves_uci", &chess_rl_native::Position::legal_moves_uci)
        .def("history_uci", &chess_rl_native::Position::history_uci)
        .def("history_fens", &chess_rl_native::Position::history_fens, py::arg("max_steps") = 8)
        .def("is_repetition", &chess_rl_native::Position::is_repetition, py::arg("count"))
        .def("outcome", [](const chess_rl_native::Position& position, bool claim_draw) {
            return outcome_to_python(position.outcome(claim_draw));
        }, py::arg("claim_draw") = true)
        .def("push_uci", &chess_rl_native::Position::push_uci, py::arg("uci"))
        .def("pop", &chess_rl_native::Position::pop)
        .def("side_to_move", &chess_rl_native::Position::side_to_move)
        .def("halfmove_clock", &chess_rl_native::Position::halfmove_clock)
        .def("fullmove_number", &chess_rl_native::Position::fullmove_number)
        .def("castling_rights", &chess_rl_native::Position::castling_rights)
        .def("ep_square", &chess_rl_native::Position::ep_square);

    module.def("perft", &chess_rl_native::perft, py::arg("fen"), py::arg("depth"));
}
