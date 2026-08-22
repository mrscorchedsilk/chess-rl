#include <pybind11/pybind11.h>

#include "chess.hpp"

#if __cplusplus < 201703L
#error "chess_rl_native requires C++17 or newer"
#endif

namespace py = pybind11;

namespace {
// This is intentionally compile-time only in Task 2.  It proves that the
// pinned amalgamated header is included without starting the rules API.
using PinnedChessBoard = chess::Board;
static_assert(sizeof(PinnedChessBoard) > 0, "pinned chess.hpp did not compile");

py::dict build_info() {
    py::dict info;
    info["cxx_standard"] = "c++17";
    info["compiler"] = __VERSION__;
    info["native_abi_version"] = CHESS_RL_NATIVE_ABI_VERSION;
    info["chess_library_commit"] = CHESS_RL_NATIVE_CHESS_LIBRARY_COMMIT;
    info["chess_library_header_sha256"] =
        CHESS_RL_NATIVE_CHESS_LIBRARY_HEADER_SHA256;
    return info;
}
}  // namespace

PYBIND11_MODULE(_chess_rl_native, module) {
    module.doc() = "Private native foundation for chess-rl";
    module.def("native_abi_version", [] { return CHESS_RL_NATIVE_ABI_VERSION; });
    module.def("chess_library_commit",
               [] { return CHESS_RL_NATIVE_CHESS_LIBRARY_COMMIT; });
    module.def("chess_library_header_sha256",
               [] { return CHESS_RL_NATIVE_CHESS_LIBRARY_HEADER_SHA256; });
    module.def("build_info", &build_info);
}
