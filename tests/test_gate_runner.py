"""Unit tests for the machine-readable gate runner (scripts/gate_runner.py).

Strict TDD: these tests were written first (RED) and only exercise harmless
fake commands -- never the real heavy gates, never training. The CLI's gate
selection and registry override make this possible.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.gate_runner as gr

EXPECTED_GATES = [
    "pytest",
    "smoke_test",
    "checkpoint_helpers",
    "light_chess",
    "dashboard",
    "parallel_pipeline",
    "native_cpp",
    "gpu_runtime",
    "resume",
    "canary",
]

MISSING_ARTIFACT_GATES = ["native_cpp", "gpu_runtime", "resume", "canary"]

PASS_CODE = "import sys; sys.exit(0)"
FAIL_CODE = "import sys; sys.exit(1)"


def fake_registry(tmp_path, **overrides):
    """Build a registry of harmless fake gates rooted at tmp_path."""
    reg = {
        "fake_pass": gr.GateSpec(
            id="fake_pass",
            description="fake passing gate",
            cmd=("{python}", "-c", PASS_CODE),
            cwd="repo",
            timeout_s=30,
            required=True,
        ),
        "fake_fail": gr.GateSpec(
            id="fake_fail",
            description="fake failing gate",
            cmd=("{python}", "-c", FAIL_CODE),
            cwd="repo",
            timeout_s=30,
            required=True,
        ),
        "fake_skip": gr.GateSpec(
            id="fake_skip",
            description="fake skip gate",
            kind="skip",
            skip_reason="missing-artifact: unit-test fake",
            required=False,
        ),
    }
    reg.update(overrides)
    return reg


def make_runner(tmp_path, registry=None, **kwargs):
    registry = registry if registry is not None else fake_registry(tmp_path)
    kwargs.setdefault("repo_root", tmp_path)
    kwargs.setdefault("probe", False)
    return gr.GateRunner(registry=registry, **kwargs)


# ---------------------------------------------------------------------------
# Schema / default registry contract
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_version_constant(self):
        assert isinstance(gr.GATE_SCHEMA_VERSION, int)
        assert gr.GATE_SCHEMA_VERSION >= 1
        assert gr.SCHEMA_NAME.startswith("chess-rl.")

    def test_default_registry_has_all_expected_gates(self):
        assert list(gr.DEFAULT_GATES.keys()) == EXPECTED_GATES

    def test_parallel_pipeline_gate_config_contract(self):
        g = gr.DEFAULT_GATES["parallel_pipeline"]
        assert g.kind == "parallel_pipeline"
        assert g.check_games == 4
        assert g.check_max_wall_seconds == 30.0
        assert "--json" in g.cmd
        assert g.required is True

    def test_missing_artifact_gates_are_explicit_skip_not_pass(self):
        for gid in MISSING_ARTIFACT_GATES:
            g = gr.DEFAULT_GATES[gid]
            assert g.kind == "skip", gid
            assert g.skip_reason.startswith("missing-artifact"), gid
            assert g.required is False, gid
            assert g.cmd == (), gid  # skipped gates must never execute a command

    def test_default_registry_has_no_forbidden_training_entrypoints(self):
        gr.validate_registry(gr.DEFAULT_GATES)

    def test_required_gates_are_the_six_real_gates(self):
        required = {gid for gid, g in gr.DEFAULT_GATES.items() if g.required}
        assert required == {"pytest", "smoke_test", "checkpoint_helpers",
                            "light_chess", "dashboard", "parallel_pipeline"}


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_forbidden_training_entrypoint_rejected(self, tmp_path):
        reg = {"bad": gr.GateSpec(id="bad", cmd=("{python}", "train.py"), cwd="repo")}
        with pytest.raises(ValueError, match="train"):
            make_runner(tmp_path, registry=reg)

    def test_train_server_rejected(self, tmp_path):
        reg = {"bad": gr.GateSpec(id="bad", cmd=("{python}", "train_server.py"), cwd="repo")}
        with pytest.raises(ValueError, match="train_server"):
            make_runner(tmp_path, registry=reg)

    @pytest.mark.parametrize("module", ["train", "train_server"])
    def test_python_module_training_entrypoint_rejected(self, tmp_path, module):
        reg = {"bad": gr.GateSpec(
            id="bad", cmd=("{python}", "-m", module), cwd="repo",
        )}
        with pytest.raises(ValueError, match=module):
            make_runner(tmp_path, registry=reg)

    def test_skip_gate_must_have_reason(self, tmp_path):
        reg = {"bad": gr.GateSpec(id="bad", kind="skip", skip_reason="")}
        with pytest.raises(ValueError, match="skip_reason"):
            make_runner(tmp_path, registry=reg)

    def test_skip_gate_must_not_have_command(self, tmp_path):
        reg = {"bad": gr.GateSpec(id="bad", kind="skip", skip_reason="x",
                                  cmd=("{python}", "-c", PASS_CODE))}
        with pytest.raises(ValueError, match="cmd"):
            make_runner(tmp_path, registry=reg)

    def test_unknown_cwd_marker_rejected(self, tmp_path):
        reg = {"bad": gr.GateSpec(id="bad", cmd=("{python}", "-c", PASS_CODE),
                                  cwd="not_a_marker")}
        with pytest.raises(ValueError, match="cwd"):
            make_runner(tmp_path, registry=reg)

    def test_missing_cwd_directory_rejected(self, tmp_path):
        reg = {"bad": gr.GateSpec(id="bad", cmd=("{python}", "-c", PASS_CODE),
                                  cwd="light_chess")}
        with pytest.raises(ValueError, match="light_chess"):
            make_runner(tmp_path, registry=reg,
                        light_chess_dir=tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_has_all_sections(self, tmp_path):
        runner = make_runner(tmp_path)
        fp = runner.fingerprint()
        for key in ("host", "git", "python", "torch", "gpu"):
            assert key in fp, key

    def test_fingerprint_git_commit_when_in_repo(self):
        runner = make_runner(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        fp = runner.fingerprint()
        assert fp["git"]["repo_root"]
        assert fp["python"]["gate_python"] == sys.executable

    def test_fingerprint_json_written_to_evidence(self, tmp_path):
        runner = make_runner(tmp_path)
        runner.run()
        fp_path = tmp_path / "evidence" / "gates"
        # find the run dir (single one) and read fingerprint.json
        run_dirs = list(fp_path.iterdir())
        assert len(run_dirs) == 1
        data = json.loads((run_dirs[0] / "fingerprint.json").read_text())
        for key in ("host", "git", "python", "torch", "gpu"):
            assert key in data


# ---------------------------------------------------------------------------
# Status classification and exit codes
# ---------------------------------------------------------------------------


class TestClassification:
    def test_pass_classification(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_pass"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["status"] == "pass"
        assert entry["exit_code"] == 0
        assert entry["timed_out"] is False
        assert summary["summary"]["ok"] is True
        assert runner.exit_code == gr.EXIT_OK

    def test_fail_classification_and_nonzero_exit(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_fail"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["status"] == "fail"
        assert entry["exit_code"] == 1
        assert summary["summary"]["ok"] is False
        assert runner.exit_code == gr.EXIT_GATE_FAILURE

    def test_skip_classification_with_reason(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_skip"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["status"] == "skip"
        assert entry["reason"].startswith("missing-artifact")
        assert entry["exit_code"] is None
        assert entry["stdout_file"] is None
        assert runner.exit_code == gr.EXIT_OK  # optional skip does not fail the run

    def test_required_skip_yields_dedicated_nonzero_exit(self, tmp_path):
        reg = fake_registry(tmp_path)
        reg["fake_skip"] = gr.GateSpec(
            id="fake_skip", kind="skip", skip_reason="missing-artifact: fake",
            required=True)
        runner = make_runner(tmp_path, registry=reg, selected=["fake_skip"])
        runner.run()
        assert runner.exit_code == gr.EXIT_REQUIRED_SKIPPED
        assert runner.summary["summary"]["required_skipped"] == 1

    def test_fail_takes_precedence_over_required_skip(self, tmp_path):
        reg = fake_registry(tmp_path)
        reg["fake_skip"] = gr.GateSpec(
            id="fake_skip", kind="skip", skip_reason="missing-artifact: fake",
            required=True)
        runner = make_runner(tmp_path, registry=reg,
                            selected=["fake_fail", "fake_skip"])
        runner.run()
        assert runner.exit_code == gr.EXIT_GATE_FAILURE

    def test_timeout_classified_as_fail(self, tmp_path):
        reg = {"sleepy": gr.GateSpec(
            id="sleepy", cmd=("{python}", "-c", "import time; time.sleep(5)"),
            cwd="repo", timeout_s=0.5, required=True)}
        runner = make_runner(tmp_path, registry=reg, selected=["sleepy"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["status"] == "fail"
        assert entry["timed_out"] is True
        assert "timeout" in (entry["reason"] or "").lower()
        assert runner.exit_code == gr.EXIT_GATE_FAILURE

    def test_missing_command_classified_as_fail(self, tmp_path):
        reg = {"ghost": gr.GateSpec(
            id="ghost", cmd=("/nonexistent/bin/definitely-not-a-command",),
            cwd="repo", required=True)}
        runner = make_runner(tmp_path, registry=reg, selected=["ghost"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["status"] == "fail"
        assert "not found" in (entry["reason"] or "").lower()
        assert runner.exit_code == gr.EXIT_GATE_FAILURE


# ---------------------------------------------------------------------------
# Evidence directory, stdout/stderr files, summary.json
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_evidence_dir_created_under_repo(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_pass"])
        runner.run()
        assert (tmp_path / "evidence" / "gates").is_dir()

    def test_exactly_one_summary_json(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_pass"])
        runner.run()
        run_dir = list((tmp_path / "evidence" / "gates").iterdir())[0]
        summaries = [p for p in run_dir.iterdir() if p.name == "summary.json"]
        assert len(summaries) == 1

    def test_summary_structure(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_pass"])
        summary = runner.run()
        run_dir = list((tmp_path / "evidence" / "gates").iterdir())[0]
        on_disk = json.loads((run_dir / "summary.json").read_text())
        assert on_disk["schema"] == gr.SCHEMA_NAME
        assert on_disk["schema_version"] == gr.GATE_SCHEMA_VERSION
        assert on_disk["run_id"]
        assert on_disk["timestamp"]
        assert on_disk["evidence_dir"] == str(run_dir)
        assert on_disk["fingerprint_file"] == "fingerprint.json"
        assert on_disk["exit_code"] == gr.EXIT_OK
        assert on_disk["summary"]["total"] == 1
        assert on_disk["summary"]["pass"] == 1
        # runner.run() returns the same object that was serialized
        assert summary == on_disk

    def test_stdout_stderr_files_written(self, tmp_path):
        code = "import sys; print('OUT-LINE'); print('ERR-LINE', file=sys.stderr); sys.exit(0)"
        reg = {"talky": gr.GateSpec(
            id="talky", cmd=("{python}", "-c", code), cwd="repo", required=True)}
        runner = make_runner(tmp_path, registry=reg, selected=["talky"])
        runner.run()
        run_dir = list((tmp_path / "evidence" / "gates").iterdir())[0]
        out = (run_dir / "talky.stdout.txt").read_text()
        err = (run_dir / "talky.stderr.txt").read_text()
        assert out == "OUT-LINE\n"
        assert err == "ERR-LINE\n"

    def test_command_and_cwd_recorded_in_entry(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_pass"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["cwd"] == str(tmp_path)
        assert "python" in os.path.basename(entry["command"][0])
        assert entry["timeout_s"] == 30

    def test_gate_runs_in_configured_cwd(self, tmp_path):
        code = "import os,sys; open(sys.argv[1],'w').write(os.getcwd())"
        reg = {"pwdrec": gr.GateSpec(
            id="pwdrec", cmd=("{python}", "-c", code, "{evidence}/cwd.txt"),
            cwd="repo", required=True)}
        runner = make_runner(tmp_path, registry=reg, selected=["pwdrec"])
        runner.run()
        run_dir = list((tmp_path / "evidence" / "gates").iterdir())[0]
        assert (run_dir / "cwd.txt").read_text() == str(tmp_path)

    def test_evidence_placeholder_resolves_inside_evidence_dir(self, tmp_path):
        code = "import os,sys; open(sys.argv[1],'w').write('x')"
        reg = {"evid": gr.GateSpec(
            id="evid", cmd=("{python}", "-c", code, "{evidence}/marker.txt"),
            cwd="repo", required=True)}
        runner = make_runner(tmp_path, registry=reg, selected=["evid"])
        runner.run()
        run_dir = list((tmp_path / "evidence" / "gates").iterdir())[0]
        assert (run_dir / "marker.txt").exists()


# ---------------------------------------------------------------------------
# subprocess safety: no shell, no pipelines, explicit cwd/timeout
# ---------------------------------------------------------------------------


class TestSubprocessSafety:
    def test_all_subprocess_calls_use_list_args_and_shell_false(self, tmp_path, monkeypatch):
        calls = []
        real_run = subprocess.run

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", spy)
        runner = make_runner(tmp_path, selected=["fake_pass"])
        runner.run()
        assert calls, "expected at least one subprocess.run call"
        for args, kwargs in calls:
            assert isinstance(args[0], (list, tuple)), "argv must be a sequence"
            assert all(isinstance(a, str) for a in args[0]), "argv must be strings"
            assert kwargs.get("shell") is False, "shell=True is forbidden"
            assert kwargs.get("timeout") is not None, "timeout is required"
            assert "cwd" in kwargs, "cwd is required"

    def test_no_shell_pipeline_characters_in_argv(self, tmp_path):
        for gate in gr.DEFAULT_GATES.values():
            for arg in gate.cmd:
                assert "|" not in arg, gate.id
                assert "&&" not in arg, gate.id
                assert ";" not in arg, gate.id


# ---------------------------------------------------------------------------
# parallel_pipeline gate behavior
# ---------------------------------------------------------------------------


def _pp_spec(json_path_template, code):
    return gr.GateSpec(
        id="pp", kind="parallel_pipeline", description="fake pp",
        cmd=("{python}", "-c", code, json_path_template),
        cwd="repo", timeout_s=30, required=True,
        check_json=json_path_template,
        check_games=4, check_max_wall_seconds=30.0,
    )


class TestParallelPipelineGate:
    def test_fast_four_game_result_passes(self, tmp_path):
        code = ("import json,sys;"
                "json.dump({'games':4,'wall_seconds':1.0}, open(sys.argv[1],'w'))")
        reg = {"pp": _pp_spec("{evidence}/pp.json", code)}
        runner = make_runner(tmp_path, registry=reg, selected=["pp"])
        summary = runner.run()
        assert summary["gates"][0]["status"] == "pass", summary
        assert runner.exit_code == gr.EXIT_OK

    def test_slow_result_fails_with_30s_assertion(self, tmp_path):
        code = ("import json,sys;"
                "json.dump({'games':4,'wall_seconds':40.0}, open(sys.argv[1],'w'))")
        reg = {"pp": _pp_spec("{evidence}/pp.json", code)}
        runner = make_runner(tmp_path, registry=reg, selected=["pp"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["status"] == "fail"
        assert "30.0" in entry["reason"]
        assert runner.exit_code == gr.EXIT_GATE_FAILURE

    def test_wrong_game_count_fails(self, tmp_path):
        code = ("import json,sys;"
                "json.dump({'games':3,'wall_seconds':1.0}, open(sys.argv[1],'w'))")
        reg = {"pp": _pp_spec("{evidence}/pp.json", code)}
        runner = make_runner(tmp_path, registry=reg, selected=["pp"])
        summary = runner.run()
        assert summary["gates"][0]["status"] == "fail"

    def test_missing_result_artifact_fails(self, tmp_path):
        reg = {"pp": _pp_spec("{evidence}/pp.json", "pass")}
        runner = make_runner(tmp_path, registry=reg, selected=["pp"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["status"] == "fail"
        assert "missing" in (entry["reason"] or "").lower()

    def test_command_failure_fails_even_with_artifact(self, tmp_path):
        code = ("import json,sys;"
                "json.dump({'games':4,'wall_seconds':1.0}, open(sys.argv[1],'w'));"
                "sys.exit(1)")
        reg = {"pp": _pp_spec("{evidence}/pp.json", code)}
        runner = make_runner(tmp_path, registry=reg, selected=["pp"])
        summary = runner.run()
        assert summary["gates"][0]["status"] == "fail"

    def test_stale_result_is_deleted_before_command(self, tmp_path):
        evidence = tmp_path / "evidence-fixed"
        evidence.mkdir()
        (evidence / "pp.json").write_text('{"games":4,"wall_seconds":1.0}')
        reg = {"pp": _pp_spec("{evidence}/pp.json", "pass")}
        runner = make_runner(
            tmp_path, registry=reg, selected=["pp"], evidence_dir=evidence,
        )
        summary = runner.run()
        assert summary["gates"][0]["status"] == "fail"
        assert "missing" in summary["gates"][0]["reason"].lower()

    @pytest.mark.parametrize(
        ("games_literal", "wall_literal"),
        [
            ("4.0", "1.0"),
            ("True", "1.0"),
            ("4", "float('nan')"),
            ("4", "float('inf')"),
            ("4", "-1.0"),
            ("4", "True"),
        ],
    )
    def test_invalid_result_types_or_nonfinite_timing_fail(
        self, tmp_path, games_literal, wall_literal,
    ):
        code = (
            "import json,sys;"
            f"json.dump({{'games':{games_literal},'wall_seconds':{wall_literal}}},"
            "open(sys.argv[1],'w'))"
        )
        reg = {"pp": _pp_spec("{evidence}/pp.json", code)}
        runner = make_runner(tmp_path, registry=reg, selected=["pp"])
        summary = runner.run()
        assert summary["gates"][0]["status"] == "fail"


# ---------------------------------------------------------------------------
# Gate selection, overrides, and CLI
# ---------------------------------------------------------------------------


class TestSelectionAndCli:
    def test_selection_runs_only_selected_gates(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_pass", "fake_skip"])
        summary = runner.run()
        assert [e["id"] for e in summary["gates"]] == ["fake_pass", "fake_skip"]
        assert summary["summary"]["total"] == 2

    def test_unselected_gate_not_in_summary(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_pass"])
        summary = runner.run()
        ids = [e["id"] for e in summary["gates"]]
        assert "fake_fail" not in ids

    def test_skip_override_of_required_gate_remains_required_and_fails_closed(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_fail"], skip=["fake_fail"])
        summary = runner.run()
        entry = summary["gates"][0]
        assert entry["status"] == "skip"
        assert entry["required"] is True
        assert "explicitly-skipped" in entry["reason"]
        assert runner.exit_code == gr.EXIT_REQUIRED_SKIPPED

    def test_skip_override_of_optional_gate_can_succeed(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_skip"], skip=["fake_skip"])
        runner.run()
        assert runner.exit_code == gr.EXIT_OK

    def test_unknown_require_or_skip_gate_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown.*require"):
            make_runner(tmp_path, require=["does_not_exist"])
        with pytest.raises(ValueError, match="unknown.*skip"):
            make_runner(tmp_path, skip=["does_not_exist"])

    def test_require_override_promotes_optional_gate(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_skip"], require=["fake_skip"])
        summary = runner.run()
        assert summary["gates"][0]["required"] is True
        assert runner.exit_code == gr.EXIT_REQUIRED_SKIPPED

    def test_parse_args_gates_list(self):
        args = gr.parse_args(["--gates", "a,b,c"])
        assert args.gates == ["a", "b", "c"]

    def test_list_mode_lists_default_gates(self, capsys):
        with pytest.raises(SystemExit) as exc:
            gr.main(["--list"])
        assert exc.value.code == gr.EXIT_OK
        out = capsys.readouterr().out
        for gid in EXPECTED_GATES:
            assert gid in out

    def test_version_mode(self, capsys):
        with pytest.raises(SystemExit) as exc:
            gr.main(["--version"])
        assert exc.value.code == gr.EXIT_OK
        assert str(gr.GATE_SCHEMA_VERSION) in capsys.readouterr().out

    def test_cli_end_to_end_with_fake_registry(self, tmp_path, capsys):
        reg = fake_registry(tmp_path)
        with pytest.raises(SystemExit) as exc:
            gr.main(["--repo", str(tmp_path), "--gates", "fake_pass"],
                    registry=reg)
        assert exc.value.code == gr.EXIT_OK
        out = capsys.readouterr().out
        assert '"status": "pass"' in out or "ok" in out.lower()

    def test_cli_failure_exit_code(self, tmp_path):
        reg = fake_registry(tmp_path)
        with pytest.raises(SystemExit) as exc:
            gr.main(["--repo", str(tmp_path), "--gates", "fake_fail"],
                    registry=reg)
        assert exc.value.code == gr.EXIT_GATE_FAILURE

    def test_cli_unknown_gate_is_config_error(self, tmp_path):
        reg = fake_registry(tmp_path)
        with pytest.raises(SystemExit) as exc:
            gr.main(["--repo", str(tmp_path), "--gates", "does_not_exist"],
                    registry=reg)
        assert exc.value.code == gr.EXIT_CONFIG_ERROR

    def test_default_evidence_dir_under_repo(self, tmp_path):
        runner = make_runner(tmp_path, selected=["fake_pass"])
        runner.run()
        assert (tmp_path / "evidence" / "gates").is_dir()

    def test_cli_rejects_arbitrary_registry_file_option(self):
        with pytest.raises(SystemExit) as exc:
            gr.parse_args(["--registry-file", "/tmp/untrusted.json"])
        assert exc.value.code == 2


class TestRunId:
    def test_run_ids_are_unique_across_runs(self, tmp_path):
        r1 = make_runner(tmp_path, selected=["fake_pass"])
        r1.run()
        r2 = make_runner(tmp_path, selected=["fake_pass"])
        r2.run()
        run_dirs = sorted((tmp_path / "evidence" / "gates").iterdir())
        assert len(run_dirs) == 2
        assert run_dirs[0].name != run_dirs[1].name
