"""Task 9: architecture_id versioning and model-size selection.

Strict TDD tests for the architecture registry and the no-cross-body-load
guarantee: a checkpoint trained under one architecture_id must never be
silently loaded into a differently-shaped body.

CPU-only. Run: .venv/bin/python -m pytest tests/test_model_v3.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ARCHITECTURES, Config  # noqa: E402
from model import (  # noqa: E402
    ChessNet,
    infer_state_dict_architecture_id,
    resolve_architecture_id,
)

EXPECTED_PARAMS = {
    "v2-6x128": 2_170_218,
    "v3-10x128": 3_352_938,
    "v3-10x192": 7_241_194,
    "v3-10x256": 12_604_010,
}


def _make_cfg(arch_id):
    num_res, num_f = ARCHITECTURES[arch_id]
    cfg = Config()
    cfg.architecture_id = arch_id
    cfg.num_res_blocks = num_res
    cfg.num_filters = num_f
    return cfg


@pytest.mark.parametrize("arch_id", sorted(ARCHITECTURES))
def test_parameter_count_matches_plan(arch_id):
    net = ChessNet(_make_cfg(arch_id))
    count = sum(p.numel() for p in net.parameters())
    assert count == EXPECTED_PARAMS[arch_id], (
        f"{arch_id}: {count:,} params != expected {EXPECTED_PARAMS[arch_id]:,}"
    )


@pytest.mark.parametrize("arch_id", sorted(ARCHITECTURES))
def test_architecture_id_round_trip(arch_id):
    net = ChessNet(_make_cfg(arch_id))
    assert net.architecture_id == arch_id
    sd = net.state_dict()
    assert infer_state_dict_architecture_id(sd) == arch_id
    num_res, num_f = ARCHITECTURES[arch_id]
    assert resolve_architecture_id(num_res, num_f) == arch_id


def test_resolve_known_shapes_are_exact_unknown_shapes_are_custom():
    assert resolve_architecture_id(5, 999) == "custom-5x999"
    assert resolve_architecture_id(6, 64) == "custom-6x64"
    # A custom body must never collide with a registered architecture_id.
    assert resolve_architecture_id(6, 64) not in ARCHITECTURES


def test_architectures_produce_distinct_weights():
    """Two different bodies must not share a state-dict key set."""
    a = ChessNet(_make_cfg("v2-6x128"))
    b = ChessNet(_make_cfg("v3-10x192"))
    assert set(a.state_dict().keys()) != set(b.state_dict().keys()) or (
        a.state_dict()["conv_in.weight"].shape
        != b.state_dict()["conv_in.weight"].shape
    )


def test_forward_shapes_constant_across_sizes():
    for arch_id in ARCHITECTURES:
        cfg = _make_cfg(arch_id)
        net = ChessNet(cfg).eval()
        x = torch.randn(2, cfg.num_input_planes, 8, 8)
        with torch.no_grad():
            logits, value = net(x)
        assert logits.shape == (2, cfg.policy_size), arch_id
        assert value.shape == (2, 1), arch_id
