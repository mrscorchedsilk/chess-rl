#!/usr/bin/env python3
"""Clean, fail-closed native foundation build/perft acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R "
    "w KQkq - 0 1"
)
EXPECTED_COMMIT = "53e6a841dcda7059a2af363d85f785ef1817304a"
EXPECTED_HEADER_SHA256 = "f2c8e2e929641e2c71cbe9d8abd718cf3cac46c2a34531215ebd733905e98d7f"


def run(argv, *, cwd, env=None, timeout=300):
    started = time.perf_counter()
    completed = subprocess.run(
        [str(value) for value in argv],
        cwd=str(cwd),
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
        shell=False,
    )
    record = {
        "argv": [str(value) for value in argv],
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "duration_s": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(repo: Path, python: Path):
    cmake = shutil.which("cmake") or str(python.with_name("cmake"))
    ctest = shutil.which("ctest") or str(python.with_name("ctest"))
    commands = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="chess-native-gate-") as temp_name:
        temp = Path(temp_name)
        build_dir = temp / "cmake-build"
        wheel_dir = temp / "wheel"
        venv_dir = temp / "venv"
        wheel_dir.mkdir()

        pybind = run(
            [python, "-m", "pybind11", "--cmakedir"], cwd=repo, timeout=30,
        )
        commands.append(pybind)
        pybind_dir = pybind["stdout"].strip()
        commands.append(run([
            cmake, "-S", repo, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind_dir}",
            f"-DPython_EXECUTABLE={python}",
        ], cwd=repo))
        commands.append(run([cmake, "--build", build_dir, "--parallel"], cwd=repo))
        commands.append(run([
            ctest, "--test-dir", build_dir, "--output-on-failure",
        ], cwd=repo))

        env = os.environ.copy()
        env["SOURCE_DATE_EPOCH"] = "946684800"
        commands.append(run([
            python, "-m", "build", "--wheel", "--outdir", wheel_dir,
            "-C", f"build-dir={temp / 'wheel-build'}",
        ], cwd=repo, env=env, timeout=600))
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected exactly one wheel, found {wheels}")
        wheel = wheels[0]

        commands.append(run([python, "-m", "venv", venv_dir], cwd=repo))
        clean_python = venv_dir / "bin" / "python"
        commands.append(run([
            clean_python, "-m", "pip", "install", "--no-deps", wheel,
        ], cwd=repo, timeout=300))
        smoke = f"""
from pathlib import Path
import chess_rl_native as n
assert n.native_abi_version() == '1'
assert n.chess_library_commit() == '{EXPECTED_COMMIT}'
assert n.chess_library_header_sha256() == '{EXPECTED_HEADER_SHA256}'
assert n.build_info()['cplusplus'] == 201703
assert 'MIT License' in (Path(n.__file__).resolve().parent / 'LICENSE').read_text()
assert n.perft({START_FEN!r}, 5) == 4865609
assert n.perft({KIWIPETE_FEN!r}, 5) == 193690690
cycle = ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2
position = n.Position.from_uci_history({START_FEN!r}, cycle)
assert position.is_repetition(3)
assert position.outcome(claim_draw=False) is None
assert position.outcome(claim_draw=True) == {{'winner': None, 'termination': 'threefold_repetition'}}
print('clean-wheel native foundation: PASS')
"""
        commands.append(run([clean_python, "-c", smoke], cwd=repo, timeout=300))

        return {
            "schema": "chess-rl.native-foundation-gate",
            "schema_version": 1,
            "status": "pass",
            "elapsed_s": time.perf_counter() - started,
            "wheel_name": wheel.name,
            "wheel_sha256": sha256(wheel),
            "chess_library_commit": EXPECTED_COMMIT,
            "chess_library_header_sha256": EXPECTED_HEADER_SHA256,
            "perft": {
                "start_depth_5": 4865609,
                "kiwipete_depth_5": 193690690,
            },
            "commands": commands,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo = Path(args.repo).resolve()
    output = Path(args.json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(repo, Path(sys.executable).absolute())
    except Exception as exc:
        result = {
            "schema": "chess-rl.native-foundation-gate",
            "schema_version": 1,
            "status": "fail",
            "error": str(exc),
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "elapsed_s": result["elapsed_s"],
        "wheel_sha256": result["wheel_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
