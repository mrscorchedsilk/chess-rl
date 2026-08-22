"""Machine-readable acceptance gate runner for ChessNet v2 (Phase 1).

Executes an ordered set of acceptance gates, records machine-readable
evidence into a per-run directory, and exits nonzero when any *selected
required* gate fails (or is skipped despite being required).

Guarantees
----------
* subprocess is always invoked with an argv *sequence* (never a shell
  string), ``shell=False``, an explicit ``cwd`` and ``timeout``. No shell
  pipelines, so exit codes can never be masked by ``| head`` / ``tail``.
* Every command gate writes ``<gate_id>.stdout.txt`` and
  ``<gate_id>.stderr.txt``; the run writes exactly one ``summary.json``
  plus a ``fingerprint.json`` (host / git / python / torch / GPU).
* Gates are classified ``pass`` | ``fail`` | ``skip(reason)``. Gates whose
  artifacts do not exist in this checkout (native C++, GPU runtime, resume,
  canary) are explicit ``skip`` with reason ``missing-artifact: ...`` --
  never a false pass.
* The built-in CLI registry never starts production training: known training
  entrypoints (``train.py``, ``train_server.py``, and their ``python -m`` forms)
  are rejected, and the canary gate is skip-only by construction. Programmatic
  registries are a trusted in-process testing/integration hook, not a CLI input.

Summary schema (``summary.json``)::

    {
      "schema": "chess-rl.gate-summary",
      "schema_version": 1,
      "run_id": "20260822T173000-1a2b3c4d",
      "timestamp": "2026-08-22T17:30:00.123456+00:00",
      "repo_root": "/abs/path",
      "evidence_dir": "/abs/path/evidence/gates/<run_id>",
      "python": "/abs/path/to/python",
      "fingerprint_file": "fingerprint.json",
      "gates": [
        {
          "id": "pytest",
          "description": "...",
          "status": "pass" | "fail" | "skip",
          "reason": null | str,
          "required": true | false,
          "command": ["...", "..."] | null,
          "cwd": "/abs/path" | null,
          "timeout_s": 1200.0,
          "exit_code": 0 | 1 | null,
          "timed_out": false,
          "duration_s": 1.23 | null,
          "stdout_file": "pytest.stdout.txt" | null,
          "stderr_file": "pytest.stderr.txt" | null
        }
      ],
      "summary": {
        "total": 10, "pass": 6, "fail": 0, "skip": 4,
        "required_failed": 0, "required_skipped": 0, "ok": true
      },
      "exit_code": 0
    }

Exit codes
----------
0  all selected required gates passed (optional skips are fine)
1  a selected required gate FAILED (or timed out)
2  a selected required gate was SKIPPED (e.g. a demanded artifact is absent)
3  configuration / usage error (unknown gate, forbidden entrypoint, bad cwd)

CLI::

    python scripts/gate_runner.py [--gates id1,id2] [--require id] [--skip id]
        [--repo PATH] [--python PATH] [--evidence PATH]
        [--light-chess-dir PATH] [--dashboard-dir PATH] [--list] [--version]
        [--no-probe]

``--gates`` selects a subset, ``--require`` promotes a gate to required and
``--skip`` records an explicit skip without weakening required status.
Placeholders in trusted in-process test registries: ``{python}``, ``{repo}``,
``{evidence}``, ``{light_chess}``, ``{dashboard}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

GATE_SCHEMA_VERSION = 1
SCHEMA_NAME = "chess-rl.gate-summary"

EXIT_OK = 0
EXIT_GATE_FAILURE = 1
EXIT_REQUIRED_SKIPPED = 2
EXIT_CONFIG_ERROR = 3

# Known training entrypoints that a gate must NEVER invoke.
FORBIDDEN_ENTRYPOINTS = ("train.py", "train_server.py", "train_server")

# Placeholders resolved per-run.
_PLACEHOLDERS = ("python", "repo", "evidence", "light_chess", "dashboard")

# cwd markers resolved per-run; anything else is treated as an absolute path.
_CWD_MARKERS = ("repo", "light_chess", "dashboard")

_TORCH_PROBE = (
    "import json, torch;"
    "print(json.dumps({"
    "'version': torch.__version__,"
    "'cuda_available': torch.cuda.is_available(),"
    "'device_count': torch.cuda.device_count(),"
    "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
    "}))"
)


@dataclass(frozen=True)
class GateSpec:
    """Definition of one gate."""

    id: str
    description: str = ""
    cmd: tuple = ()                     # argv templates; () for skip gates
    cwd: str = "repo"                   # marker or absolute path; ignored for skip
    timeout_s: float = 120.0
    required: bool = True
    kind: str = "command"               # "command" | "parallel_pipeline" | "skip"
    skip_reason: str = ""               # required for kind == "skip"
    check_games: int = 4                # parallel_pipeline: expected game count
    check_max_wall_seconds: float = 30.0  # parallel_pipeline: <30s assertion
    check_json: str = "{evidence}/parallel_pipeline.json"  # result artifact template
    env: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "cmd": list(self.cmd),
            "cwd": self.cwd,
            "timeout_s": self.timeout_s,
            "required": self.required,
            "kind": self.kind,
            "skip_reason": self.skip_reason,
            "check_games": self.check_games,
            "check_max_wall_seconds": self.check_max_wall_seconds,
            "check_json": self.check_json,
            "env": dict(self.env),
        }


# ---------------------------------------------------------------------------
# Default registry: the Phase-1 acceptance gates.
# ---------------------------------------------------------------------------

DEFAULT_GATES = {
    "pytest": GateSpec(
        id="pytest",
        description="Current Python pytest suite (repo root)",
        cmd=("{python}", "-m", "pytest", "-q"),
        cwd="repo",
        timeout_s=1200.0,
        required=True,
    ),
    "smoke_test": GateSpec(
        id="smoke_test",
        description="smoke_test.py encoding/model/MCTS/selfplay checks",
        cmd=("{python}", "smoke_test.py"),
        cwd="repo",
        timeout_s=900.0,
        required=True,
    ),
    "checkpoint_helpers": GateSpec(
        id="checkpoint_helpers",
        description="test_checkpoint_helpers.py checkpoint helper checks",
        cmd=("{python}", "test_checkpoint_helpers.py"),
        cwd="repo",
        timeout_s=300.0,
        required=True,
    ),
    "light_chess": GateSpec(
        id="light_chess",
        description="Light Chess node engine tests (node test.js)",
        cmd=("node", "test.js"),
        cwd="light_chess",
        timeout_s=120.0,
        required=True,
    ),
    "dashboard": GateSpec(
        id="dashboard",
        description="Dashboard static contract node test (node test_dashboard.js)",
        cmd=("node", "test_dashboard.js"),
        cwd="dashboard",
        timeout_s=120.0,
        required=True,
    ),
    "parallel_pipeline": GateSpec(
        id="parallel_pipeline",
        description="benchmarks/parallel_pipeline.py four-game result under 30s",
        cmd=("{python}", "benchmarks/parallel_pipeline.py", "--games", "4",
             "--json", "{evidence}/parallel_pipeline.json"),
        cwd="repo",
        timeout_s=600.0,
        required=True,
        kind="parallel_pipeline",
        check_games=4,
        check_max_wall_seconds=30.0,
        check_json="{evidence}/parallel_pipeline.json",
    ),
    "native_cpp": GateSpec(
        id="native_cpp",
        description="Native C++ rules/MCTS build and parity tests",
        kind="skip",
        skip_reason=(
            "missing-artifact: native C++ build/parity artifacts "
            "(native-foundation track) are not present in this checkout"
        ),
        required=False,
    ),
    "gpu_runtime": GateSpec(
        id="gpu_runtime",
        description="GPU fast-path runtime profile (CUDA events, batch fill)",
        kind="skip",
        skip_reason=(
            "missing-artifact: GPU runtime profile artifact is not present "
            "in this checkout"
        ),
        required=False,
    ),
    "resume": GateSpec(
        id="resume",
        description="Fresh-process stop/save/resume parity test",
        kind="skip",
        skip_reason=(
            "missing-artifact: schema-v3 checkpoint resume artifact is not "
            "present in this checkout"
        ),
        required=False,
    ),
    "canary": GateSpec(
        id="canary",
        description="30-minute training canary (never auto-started)",
        kind="skip",
        skip_reason=(
            "missing-artifact: 30-minute canary artifact is not present; "
            "the gate runner never starts production training"
        ),
        required=False,
    ),
}


def validate_registry(registry: dict) -> None:
    """Static validation of a gate registry. Raises ValueError on violation."""
    for gid, spec in registry.items():
        if spec.id != gid:
            raise ValueError(f"gate key {gid!r} does not match spec id {spec.id!r}")
        if spec.kind == "skip":
            if not spec.skip_reason:
                raise ValueError(f"skip gate {gid!r} requires a skip_reason")
            if spec.cmd:
                raise ValueError(f"skip gate {gid!r} must not define cmd")
            continue
        if spec.kind not in ("command", "parallel_pipeline"):
            raise ValueError(f"gate {gid!r} has unknown kind {spec.kind!r}")
        if spec.cwd not in _CWD_MARKERS and not isinstance(spec.cwd, (str, os.PathLike)):
            raise ValueError(f"gate {gid!r} has invalid cwd {spec.cwd!r}")
        for index, arg in enumerate(spec.cmd):
            if os.path.basename(arg) in FORBIDDEN_ENTRYPOINTS:
                raise ValueError(
                    f"forbidden training entrypoint {os.path.basename(arg)!r} "
                    f"in gate {gid!r}; the gate runner must never start production training"
                )
            if (
                arg == "-m"
                and index + 1 < len(spec.cmd)
                and spec.cmd[index + 1] in ("train", "train_server")
            ):
                raise ValueError(
                    f"forbidden training module {spec.cmd[index + 1]!r} "
                    f"in gate {gid!r}"
                )


def load_registry_file(path: Path) -> dict:
    """Load a registry override file: {"gates": {id: {...}}}."""
    data = json.loads(Path(path).read_text())
    raw_gates = data.get("gates", data)
    if not isinstance(raw_gates, dict):
        raise ValueError(f"registry file {path} must contain a 'gates' object")
    registry = {}
    for gid, raw in raw_gates.items():
        cmd = tuple(raw.get("cmd") or ())
        registry[gid] = GateSpec(
            id=gid,
            description=raw.get("description", ""),
            cmd=cmd,
            cwd=raw.get("cwd", "repo"),
            timeout_s=float(raw.get("timeout_s", 120.0)),
            required=bool(raw.get("required", True)),
            kind=raw.get("kind", "command"),
            skip_reason=raw.get("skip_reason", ""),
            check_games=int(raw.get("check_games", 4)),
            check_max_wall_seconds=float(raw.get("check_max_wall_seconds", 30.0)),
            check_json=raw.get("check_json", "{evidence}/parallel_pipeline.json"),
            env=dict(raw.get("env") or {}),
        )
    return registry


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class GateRunner:
    """Runs a gate registry and writes machine-readable evidence."""

    def __init__(
        self,
        registry: dict | None = None,
        repo_root: str | os.PathLike | None = None,
        python: str | None = None,
        evidence_dir: str | os.PathLike | None = None,
        light_chess_dir: str | os.PathLike | None = None,
        dashboard_dir: str | os.PathLike | None = None,
        selected: list[str] | None = None,
        require: list[str] | None = None,
        skip: list[str] | None = None,
        probe: bool = True,
    ) -> None:
        self.registry = dict(registry if registry is not None else DEFAULT_GATES)
        validate_registry(self.registry)

        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"repo root does not exist: {self.repo_root}")
        self.python = python or sys.executable
        self.evidence_dir_arg = evidence_dir
        self.light_chess_dir = Path(
            light_chess_dir or (self.repo_root.parent / "light-chess")
        )
        self.dashboard_dir = Path(
            dashboard_dir or (self.repo_root.parent / "chess-training-dashboard")
        )
        self.probe = probe

        # Resolve selection + overrides.
        if selected is None:
            self.selected_ids = list(self.registry.keys())
        else:
            unknown = [g for g in selected if g not in self.registry]
            if unknown:
                raise ValueError(f"unknown gate id(s): {', '.join(unknown)}")
            self.selected_ids = [g for g in self.registry if g in selected]
        self.require = set(require or [])
        self.skip = set(skip or [])
        unknown_require = sorted(self.require - set(self.registry))
        unknown_skip = sorted(self.skip - set(self.registry))
        if unknown_require:
            raise ValueError(f"unknown require gate id(s): {', '.join(unknown_require)}")
        if unknown_skip:
            raise ValueError(f"unknown skip gate id(s): {', '.join(unknown_skip)}")

        # Validate cwd directories for every non-skip gate that will run.
        for gid in self.selected_ids:
            spec = self.registry[gid]
            if spec.kind == "skip":
                continue
            cwd_path = self._resolve_cwd(spec)
            if not cwd_path.is_dir():
                raise ValueError(
                    f"gate {gid!r} cwd directory does not exist "
                    f"({spec.cwd!r} -> {cwd_path})"
                )

        self.summary: dict | None = None

    # -- path / argv resolution --------------------------------------------

    def _placeholders(self) -> dict:
        return {
            "python": self.python,
            "repo": str(self.repo_root),
            "evidence": str(self.evidence_dir),
            "light_chess": str(self.light_chess_dir),
            "dashboard": str(self.dashboard_dir),
        }

    def _resolve(self, template: str) -> str:
        """Brace-safe placeholder substitution (never str.format on argv)."""
        result = template
        for name in _PLACEHOLDERS:
            result = result.replace("{" + name + "}", str(self._placeholders()[name]))
        return result

    def _resolve_cwd(self, spec: GateSpec) -> Path:
        if spec.cwd == "repo":
            return self.repo_root
        if spec.cwd == "light_chess":
            return self.light_chess_dir
        if spec.cwd == "dashboard":
            return self.dashboard_dir
        return Path(spec.cwd).expanduser()

    # -- fingerprint -------------------------------------------------------

    def _probe(self, argv: list[str], cwd: Path, timeout: float = 10.0):
        """Best-effort subprocess probe; never raises."""
        try:
            return subprocess.run(
                argv,
                cwd=str(cwd),
                timeout=timeout,
                capture_output=True,
                text=True,
                shell=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

    def fingerprint(self) -> dict:
        """host / git / python / torch / GPU fingerprint (best effort)."""
        host = {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        }

        if not self.probe:
            git = {"repo_root": str(self.repo_root), "commit": None,
                   "branch": None, "dirty": None}
            py = {
                "runner_version": platform.python_version(),
                "runner_executable": sys.executable,
                "gate_python": self.python,
                "gate_python_version": None,
            }
            torch = {"available": False, "reason": "probes disabled"}
            gpu = {"available": False, "reason": "probes disabled"}
        else:
            git = {"repo_root": str(self.repo_root), "commit": None,
                   "branch": None, "dirty": None}
            p = self._probe(["git", "rev-parse", "HEAD"], self.repo_root)
            if p is not None and p.returncode == 0:
                git["commit"] = p.stdout.strip() or None
            p = self._probe(["git", "symbolic-ref", "--short", "HEAD"], self.repo_root)
            if p is not None and p.returncode == 0:
                git["branch"] = p.stdout.strip() or None
            p = self._probe(["git", "status", "--porcelain"], self.repo_root)
            if p is not None:
                git["dirty"] = bool(p.stdout.strip())

            py = {
                "runner_version": platform.python_version(),
                "runner_executable": sys.executable,
                "gate_python": self.python,
                "gate_python_version": None,
            }
            p = self._probe([self.python, "-V"], self.repo_root)
            if p is not None and p.returncode == 0:
                py["gate_python_version"] = (p.stdout or p.stderr or "").strip() or None

            torch = {"available": False}
            p = self._probe([self.python, "-c", _TORCH_PROBE], self.repo_root,
                            timeout=30.0)
            if p is not None and p.returncode == 0 and p.stdout.strip():
                try:
                    torch = json.loads(p.stdout.strip())
                    torch["available"] = True
                except json.JSONDecodeError:
                    torch = {"available": False, "error": "unparseable torch probe"}
            else:
                torch = {"available": False,
                         "error": (p.stderr.strip() if p is not None else "no torch")}

            gpu = {"available": False}
            p = self._probe(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                self.repo_root,
            )
            if p is not None and p.returncode == 0:
                gpu = {
                    "available": True,
                    "lines": [ln for ln in p.stdout.splitlines() if ln.strip()],
                }

        return {
            "host": host,
            "git": git,
            "python": py,
            "torch": torch,
            "gpu": gpu,
        }

    # -- gate execution ----------------------------------------------------

    def _run_command(self, argv: list[str], cwd: Path, spec: GateSpec):
        """Run one command gate without a shell. Returns (result_dict, ok)."""
        started = time.perf_counter()
        timed_out = False
        exit_code = None
        stdout = ""
        stderr = ""
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd),
                timeout=spec.timeout_s,
                capture_output=True,
                text=True,
                shell=False,
                env={**os.environ, **spec.env},
            )
            exit_code = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        except FileNotFoundError:
            stderr = f"command not found: {argv[0]}"
        duration_s = round(time.perf_counter() - started, 3)

        if exit_code is None and not timed_out:
            reason = f"command not found: {argv[0]}"
            ok = False
        elif timed_out:
            reason = f"timeout after {spec.timeout_s:g}s (no exit code)"
            ok = False
        elif exit_code == 0:
            reason = None
            ok = True
        else:
            reason = f"exit code {exit_code}"
            ok = False
        return {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_s": duration_s,
            "stdout": stdout,
            "stderr": stderr,
            "reason": reason,
            "ok": ok,
        }

    def _check_parallel_pipeline(self, spec: GateSpec, result: dict) -> dict:
        """Post-process a parallel_pipeline gate result (four-game <30s)."""
        if not result["ok"]:
            return result
        artifact = Path(self._resolve(spec.check_json))
        if not artifact.is_file():
            result["ok"] = False
            result["reason"] = f"result artifact missing: {artifact}"
            return result
        try:
            data = json.loads(artifact.read_text())
        except json.JSONDecodeError:
            result["ok"] = False
            result["reason"] = f"result artifact is not valid JSON: {artifact}"
            return result
        games = data.get("games")
        wall = data.get("wall_seconds")
        if type(games) is not int or games != spec.check_games:
            result["ok"] = False
            result["reason"] = (
                f"four-game result: expected integer {spec.check_games} games, "
                f"got {games!r}"
            )
            return result
        if (
            type(wall) not in (int, float)
            or isinstance(wall, bool)
            or not math.isfinite(wall)
            or wall < 0.0
            or wall >= spec.check_max_wall_seconds
        ):
            result["ok"] = False
            result["reason"] = (
                f"four-game result: wall={wall!r} s; requires a finite value "
                f"in [0, {spec.check_max_wall_seconds})"
            )
            return result
        return result

    # -- run ---------------------------------------------------------------

    def run(self) -> dict:
        """Execute selected gates, write evidence, return the summary dict."""
        run_id = (
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        )
        if self.evidence_dir_arg is not None:
            self.evidence_dir = Path(self.evidence_dir_arg).resolve()
        else:
            self.evidence_dir = self.repo_root / "evidence" / "gates" / run_id
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        fp = self.fingerprint()
        (self.evidence_dir / "fingerprint.json").write_text(
            json.dumps(fp, indent=2, sort_keys=True) + "\n"
        )

        entries = []
        for gid in self.selected_ids:
            spec = self.registry[gid]
            forced_skip = gid in self.skip
            required = spec.required or gid in self.require
            if forced_skip or spec.kind == "skip":
                reason = (
                    "explicitly-skipped"
                    if forced_skip
                    else spec.skip_reason
                )
                entries.append({
                    "id": gid,
                    "description": spec.description,
                    "status": "skip",
                    "reason": reason,
                    "required": required,
                    "command": None,
                    "cwd": None,
                    "timeout_s": spec.timeout_s,
                    "exit_code": None,
                    "timed_out": False,
                    "duration_s": None,
                    "stdout_file": None,
                    "stdout_bytes": None,
                    "stdout_sha256": None,
                    "stderr_file": None,
                    "stderr_bytes": None,
                    "stderr_sha256": None,
                })
                continue

            argv = [self._resolve(arg) for arg in spec.cmd]
            cwd = self._resolve_cwd(spec)
            if spec.kind == "parallel_pipeline":
                # A reused evidence directory must never let an old passing
                # benchmark satisfy a command that failed to write fresh data.
                Path(self._resolve(spec.check_json)).unlink(missing_ok=True)
            result = self._run_command(argv, cwd, spec)
            if spec.kind == "parallel_pipeline":
                result = self._check_parallel_pipeline(spec, result)

            out_name = f"{gid}.stdout.txt"
            err_name = f"{gid}.stderr.txt"
            out_payload = result["stdout"].encode("utf-8")
            err_payload = result["stderr"].encode("utf-8")
            (self.evidence_dir / out_name).write_bytes(out_payload)
            (self.evidence_dir / err_name).write_bytes(err_payload)

            entries.append({
                "id": gid,
                "description": spec.description,
                "status": "pass" if result["ok"] else "fail",
                "reason": result["reason"],
                "required": required,
                "command": argv,
                "cwd": str(cwd),
                "timeout_s": spec.timeout_s,
                "exit_code": result["exit_code"],
                "timed_out": result["timed_out"],
                "duration_s": result["duration_s"],
                "stdout_file": out_name,
                "stdout_bytes": len(out_payload),
                "stdout_sha256": hashlib.sha256(out_payload).hexdigest(),
                "stderr_file": err_name,
                "stderr_bytes": len(err_payload),
                "stderr_sha256": hashlib.sha256(err_payload).hexdigest(),
            })

        required_failed = sum(
            1 for e in entries if e["status"] == "fail" and e["required"]
        )
        required_skipped = sum(
            1 for e in entries if e["status"] == "skip" and e["required"]
        )
        summary_counts = {
            "total": len(entries),
            "pass": sum(1 for e in entries if e["status"] == "pass"),
            "fail": sum(1 for e in entries if e["status"] == "fail"),
            "skip": sum(1 for e in entries if e["status"] == "skip"),
            "required_failed": required_failed,
            "required_skipped": required_skipped,
            "ok": required_failed == 0 and required_skipped == 0,
        }
        if required_failed:
            exit_code = EXIT_GATE_FAILURE
        elif required_skipped:
            exit_code = EXIT_REQUIRED_SKIPPED
        else:
            exit_code = EXIT_OK

        summary = {
            "schema": SCHEMA_NAME,
            "schema_version": GATE_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(self.repo_root),
            "evidence_dir": str(self.evidence_dir),
            "python": self.python,
            "fingerprint_file": "fingerprint.json",
            "gates": entries,
            "summary": summary_counts,
            "exit_code": exit_code,
        }
        (self.evidence_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        self.summary = summary
        self._exit_code = exit_code
        return summary

    @property
    def exit_code(self) -> int:
        if self.summary is None:
            raise RuntimeError("run() has not been called")
        return self._exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gate_runner",
        description="Machine-readable acceptance gate runner for ChessNet v2.",
    )
    parser.add_argument("--gates", default=None,
                        help="comma-separated subset of gate ids to run")
    parser.add_argument("--require", action="append", default=[],
                        help="promote a gate to required (repeatable)")
    parser.add_argument("--skip", action="append", default=[],
                        help="force-skip a gate (repeatable; required skips exit 2)")
    parser.add_argument("--repo", default=None,
                        help="repo root (default: parent of this file)")
    parser.add_argument("--python", default=None,
                        help="python executable for {python} placeholders")
    parser.add_argument("--evidence", default=None,
                        help="evidence directory (default: <repo>/evidence/gates/<run_id>)")
    parser.add_argument("--light-chess-dir", default=None,
                        help="Light Chess directory for the light_chess gate")
    parser.add_argument("--dashboard-dir", default=None,
                        help="dashboard directory for the dashboard gate")
    parser.add_argument("--list", action="store_true",
                        help="list gate definitions and exit")
    parser.add_argument("--version", action="store_true",
                        help="print summary schema name/version and exit")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip external fingerprint probes (git/python/torch/nvidia-smi)")
    args = parser.parse_args(argv)
    if args.gates:
        args.gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    return args


def _print_listing(registry: dict) -> None:
    for gid, spec in registry.items():
        role = "required" if spec.required else "optional"
        print(f"{gid:<20} [{role:<8}] ({spec.kind}) {spec.description}")
        if spec.kind == "skip":
            print(f"{'':<20}   skip reason: {spec.skip_reason}")


def main(argv=None, registry: dict | None = None) -> None:
    """CLI entry point. Raises SystemExit with the run exit code."""
    args = parse_args(argv)

    if args.version:
        print(f"{SCHEMA_NAME} v{GATE_SCHEMA_VERSION}")
        raise SystemExit(EXIT_OK)
    if args.list:
        reg = registry if registry is not None else DEFAULT_GATES
        _print_listing(reg)
        raise SystemExit(EXIT_OK)

    try:
        # ``registry`` is an in-process testing/integration hook. The public
        # CLI intentionally exposes no arbitrary registry-file execution.
        reg = registry if registry is not None else DEFAULT_GATES
        runner = GateRunner(
            registry=reg,
            repo_root=args.repo,
            python=args.python,
            evidence_dir=args.evidence,
            light_chess_dir=args.light_chess_dir,
            dashboard_dir=args.dashboard_dir,
            selected=args.gates,
            require=args.require,
            skip=args.skip,
            probe=not args.no_probe,
        )
        summary = runner.run()
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_CONFIG_ERROR) from exc

    for entry in summary["gates"]:
        status = entry["status"]
        reason = f" ({entry['reason']})" if entry["reason"] else ""
        print(f"gate {entry['id']}: {status}{reason}")
    counts = summary["summary"]
    print(f"gate-runner summary: pass={counts['pass']} fail={counts['fail']} "
          f"skip={counts['skip']} required_failed={counts['required_failed']} "
          f"required_skipped={counts['required_skipped']} ok={counts['ok']}")
    print(f"evidence: {summary['evidence_dir']}")
    raise SystemExit(runner.exit_code)


if __name__ == "__main__":
    main()
