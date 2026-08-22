#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

#include "chess.hpp"
#include "policy.h"
#include "position.h"

#if __cplusplus < 201703L
#error "chess_rl_native requires C++17 or newer"
#endif

namespace py = pybind11;

namespace chess_rl_native {
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

// {uci: prob} -> 4672-length float32 policy vector (mass preserved exactly).
py::array_t<float> policy_to_vector(const py::dict& pi) {
    py::array_t<float> vec(POLICY_SIZE);
    auto* data = vec.mutable_data();
    std::fill(data, data + POLICY_SIZE, 0.0f);
    for (auto item : pi) {
        const std::string uci = py::str(item.first);
        const float prob = item.second.cast<float>();
        data[move_to_index(uci)] += prob;
    }
    return vec;
}

// Encode a Position into a (12*history_steps + 8, 8, 8) float32 numpy array.
py::array_t<float> encode_position(const Position& position, int history_steps) {
    if (history_steps < 1) throw std::invalid_argument("history_steps must be positive");
    const int num_planes = 12 * history_steps + 8;
    py::array_t<float> result({num_planes, 8, 8});
    position.encode_planes(result.mutable_data(), history_steps);
    return result;
}

py::array_t<float> encode_fen(const std::string& fen, int history_steps) {
    return encode_position(Position::from_fen(fen), history_steps);
}

// Dense 4672-length 0/1 legal-move mask (parity/testing only; the hot path
// uses Position::legal_move_indices() to avoid the dense allocation).
py::array_t<float> legal_move_mask(const Position& position) {
    py::array_t<float> mask(POLICY_SIZE);
    auto* data = mask.mutable_data();
    std::fill(data, data + POLICY_SIZE, 0.0f);
    for (const int idx : position.legal_move_indices()) data[idx] = 1.0f;
    return mask;
}

}  // namespace
}  // namespace chess_rl_native

PYBIND11_MODULE(_chess_rl_native, module) {
    using namespace chess_rl_native;
    module.doc() = "Native chess position adapter, policy map, encoder and perft";
    module.def("native_abi_version", [] { return CHESS_RL_NATIVE_ABI_VERSION; });
    module.def("chess_library_commit", [] { return CHESS_RL_NATIVE_CHESS_LIBRARY_COMMIT; });
    module.def("chess_library_header_sha256", [] { return CHESS_RL_NATIVE_CHESS_LIBRARY_HEADER_SHA256; });
    module.def("build_info", &build_info);

    // Policy / action-map constants (AlphaZero 73-plane policy).
    module.attr("POLICY_PLANES") = POLICY_PLANES;
    module.attr("POLICY_SIZE") = POLICY_SIZE;
    module.attr("QUEEN_PLANES") = QUEEN_PLANES;
    module.attr("KNIGHT_PLANES") = KNIGHT_PLANES;
    module.attr("UNDERPROMOTION_PLANES") = UNDERPROMOTION_PLANES;

    module.def("move_to_index", &move_to_index, py::arg("uci"));
    module.def("index_to_move", &index_to_move, py::arg("index"),
               py::arg("side_to_move"));
    module.def("policy_to_vector", &policy_to_vector, py::arg("pi"));
    module.def("encode_fen", &encode_fen, py::arg("fen"),
               py::arg("history_steps") = 8);

    py::class_<Position>(module, "Position")
        .def_static("from_fen", &Position::from_fen, py::arg("fen"))
        .def_static("from_uci_history", &Position::from_uci_history,
                    py::arg("start_fen"), py::arg("moves"))
        .def("fen", &Position::fen)
        .def("legal_moves_uci", &Position::legal_moves_uci)
        .def("history_uci", &Position::history_uci)
        .def("history_fens", &Position::history_fens, py::arg("max_steps") = 8)
        .def("is_repetition", &Position::is_repetition, py::arg("count"))
        .def("outcome", [](const Position& position, bool claim_draw) {
            return outcome_to_python(position.outcome(claim_draw));
        }, py::arg("claim_draw") = true)
        .def("push_uci", &Position::push_uci, py::arg("uci"))
        .def("pop", &Position::pop)
        .def("side_to_move", &Position::side_to_move)
        .def("halfmove_clock", &Position::halfmove_clock)
        .def("fullmove_number", &Position::fullmove_number)
        .def("castling_rights", &Position::castling_rights)
        .def("ep_square", &Position::ep_square)
        .def("raw_ep_square", &Position::raw_ep_square)
        .def("legal_move_indices", &Position::legal_move_indices)
        .def("index_to_move", &Position::index_to_move, py::arg("index"))
        .def("legal_move_mask", &legal_move_mask)
        .def("encode", &encode_position, py::arg("history_steps") = 8);

    module.def("perft", &perft, py::arg("fen"), py::arg("depth"));
}
