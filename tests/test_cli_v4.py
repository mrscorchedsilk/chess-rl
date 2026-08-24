"""CLI surface for launching a new-architecture lineage.

There was no way to select an architecture from the command line, so a v4
lineage could not be started without editing config.py.  The important
correctness point: --architecture must set the BODY (num_res_blocks /
num_filters) from the registry, not just the label — setting the label alone
would leave the class defaults in place and silently train a 6x128 body under
a "v4-20x256" name.

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_cli_v4.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import ARCHITECTURES   # noqa: E402
import train                       # noqa: E402


def parse(*argv):
    return train.build_parser().parse_args(list(argv))


def build_cfg(*argv):
    """Run main()'s override block without dispatching a training run."""
    captured = {}

    def fake_run_native(cfg=None, resume=False, warm_start_checkpoint=None):
        captured["cfg"] = cfg
        captured["resume"] = resume

    original = train.run_native
    train.run_native = fake_run_native
    try:
        train.main(["--selfplay-backend", "native", *argv])
    finally:
        train.run_native = original
    return captured["cfg"]


# --------------------------------------------------------------------------- #
#  architecture selection                                                     #
# --------------------------------------------------------------------------- #

def test_v4_is_selectable():
    assert "v4-20x256" in ARCHITECTURES
    args = parse("--architecture", "v4-20x256")
    assert args.architecture == "v4-20x256"


def test_unknown_architecture_is_rejected():
    with pytest.raises(SystemExit):
        parse("--architecture", "v9-99x999")


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_architecture_sets_the_body_not_just_the_label(arch):
    cfg = build_cfg("--architecture", arch)
    blocks, filters = ARCHITECTURES[arch]
    assert cfg.architecture_id == arch
    assert (cfg.num_res_blocks, cfg.num_filters) == (blocks, filters)


def test_v4_body_is_twenty_by_two_fifty_six():
    cfg = build_cfg("--architecture", "v4-20x256")
    assert (cfg.num_res_blocks, cfg.num_filters) == (20, 256)


# --------------------------------------------------------------------------- #
#  the other knobs                                                            #
# --------------------------------------------------------------------------- #

def test_throughput_and_cadence_knobs_are_independent():
    cfg = build_cfg("--games-in-flight", "96", "--games-per-iteration", "20")
    assert cfg.selfplay_games_in_flight == 96
    assert cfg.games_per_iteration == 20


def test_training_and_replay_knobs_apply():
    cfg = build_cfg("--train-epoch-size", "16384", "--replay-size", "500000",
                    "--shards", "4")
    assert cfg.train_epoch_size == 16384
    assert cfg.replay_buffer_size == 500_000
    assert cfg.selfplay_shards == 4


def test_arena_knobs_apply():
    cfg = build_cfg("--arena-games", "200", "--arena-simulations", "200")
    assert cfg.arena_games == 200
    assert cfg.arena_simulations == 200


def test_feature_flags_are_off_unless_passed():
    cfg = build_cfg("--architecture", "v4-20x256")
    assert cfg.moves_left_head is False
    assert cfg.resign_enabled is False


def test_feature_flags_can_be_enabled():
    cfg = build_cfg("--moves-left-head", "--resign")
    assert cfg.moves_left_head is True
    assert cfg.resign_enabled is True


def test_checkpoint_dir_is_honoured(tmp_path):
    target = str(tmp_path / "v4")
    cfg = build_cfg("--architecture", "v4-20x256", "--checkpoint-dir", target)
    assert cfg.checkpoint_dir == target


def test_no_overrides_leaves_cfg_none():
    """Unchanged behaviour: with no flags the run uses its own default Config."""
    assert build_cfg() is None
