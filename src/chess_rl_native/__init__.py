"""Stable Python facade for the private chess-rl native extension."""

from . import _chess_rl_native as _native

__version__ = "0.1.0"

Position = _native.Position
perft = _native.perft


def native_abi_version() -> str:
    return _native.native_abi_version()


def chess_library_commit() -> str:
    return _native.chess_library_commit()


def chess_library_header_sha256() -> str:
    return _native.chess_library_header_sha256()


def build_info() -> dict[str, str]:
    return dict(_native.build_info())


__all__ = [
    "__version__",
    "Position",
    "perft",
    "native_abi_version",
    "chess_library_commit",
    "chess_library_header_sha256",
    "build_info",
]
