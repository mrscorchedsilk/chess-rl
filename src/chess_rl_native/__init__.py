"""Stable Python facade for the private chess-rl native extension."""

from . import _chess_rl_native as _native

__version__ = "0.1.0"

Position = _native.Position
perft = _native.perft

# Native MCTS core (Task 5): batched PUCT search, semantics-matched to mcts.py.
MCTS = _native.MCTS

# Native multi-game self-play Actor (Task 6): per-game MCTS cores, merged
# gather_leaves batches, temperature-sampled moves, z-labelled examples.
Actor = _native.Actor

# Policy / action-map constants (AlphaZero 73-plane policy).
POLICY_PLANES = _native.POLICY_PLANES
POLICY_SIZE = _native.POLICY_SIZE
QUEEN_PLANES = _native.QUEEN_PLANES
KNIGHT_PLANES = _native.KNIGHT_PLANES
UNDERPROMOTION_PLANES = _native.UNDERPROMOTION_PLANES

# Action map.
move_to_index = _native.move_to_index
index_to_move = _native.index_to_move
policy_to_vector = _native.policy_to_vector

# Encoder.
encode_fen = _native.encode_fen


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
    "MCTS",
    "Actor",
    "POLICY_PLANES",
    "POLICY_SIZE",
    "QUEEN_PLANES",
    "KNIGHT_PLANES",
    "UNDERPROMOTION_PLANES",
    "move_to_index",
    "index_to_move",
    "policy_to_vector",
    "encode_fen",
    "native_abi_version",
    "chess_library_commit",
    "chess_library_header_sha256",
    "build_info",
]
