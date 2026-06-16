from __future__ import annotations

from wm_poc.local_global.runtime import REEXEC_FLAG, setup_mujoco_runtime


def _calls():
    captured = {}

    def execv(exe, argv):
        captured["exe"] = exe
        captured["argv"] = list(argv)

    return captured, execv


def test_no_reexec_when_mujoco_absent():
    # Local/CPU box: no mujoco dirs -> no re-exec, but env defaults still set.
    env = {"HOME": "/home/u"}
    captured, execv = _calls()
    triggered = setup_mujoco_runtime(
        environ=env, isdir=lambda p: False, execv=execv, argv=["x"], executable="py"
    )
    assert triggered is False
    assert "exe" not in captured  # execv never called
    assert env["MUJOCO_PY_MUJOCO_PATH"] == "/home/u/.mujoco/mujoco210"
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert env["MUJOCO_GL"] == "egl"


def test_reexec_when_mujoco_bin_missing_from_ld_path():
    env = {"HOME": "/root", "LD_LIBRARY_PATH": "/usr/lib"}
    bins = {"/root/.mujoco/mujoco210/bin"}
    captured, execv = _calls()
    triggered = setup_mujoco_runtime(
        environ=env,
        isdir=lambda p: p in bins,
        execv=execv,
        argv=["run_planning_eval.py", "--smoke"],
        executable="/usr/bin/python3",
    )
    assert triggered is True
    assert captured["exe"] == "/usr/bin/python3"
    assert captured["argv"] == ["/usr/bin/python3", "run_planning_eval.py", "--smoke"]
    # mujoco bin prepended, existing entries preserved, re-exec guard set.
    assert env["LD_LIBRARY_PATH"].startswith("/root/.mujoco/mujoco210/bin:")
    assert env["LD_LIBRARY_PATH"].endswith(":/usr/lib")
    assert env[REEXEC_FLAG] == "1"


def test_idempotent_after_reexec_flag_set():
    # The re-exec'd child must not loop even though the dir is still missing.
    env = {"HOME": "/root", "LD_LIBRARY_PATH": "/usr/lib", REEXEC_FLAG: "1"}
    captured, execv = _calls()
    triggered = setup_mujoco_runtime(
        environ=env, isdir=lambda p: True, execv=execv, argv=["x"], executable="py"
    )
    assert triggered is False
    assert "exe" not in captured


def test_no_reexec_when_already_on_ld_path():
    # mujoco bin already present and the (only existing) candidate; no nvidia
    # dirs on this box -> nothing to add -> no re-exec.
    env = {"HOME": "/root", "LD_LIBRARY_PATH": "/root/.mujoco/mujoco210/bin:/usr/lib"}
    captured, execv = _calls()
    triggered = setup_mujoco_runtime(
        environ=env,
        isdir=lambda p: p == "/root/.mujoco/mujoco210/bin",
        execv=execv,
        argv=["x"],
        executable="py",
    )
    assert triggered is False
    assert "exe" not in captured


def test_respects_explicit_mujoco_dir_override():
    env = {"HOME": "/root", "DINO_MUJOCO210_DIR": "/opt/mujoco210"}
    captured, execv = _calls()
    triggered = setup_mujoco_runtime(
        environ=env, isdir=lambda p: p == "/opt/mujoco210/bin", execv=execv, argv=["x"], executable="py"
    )
    assert triggered is True
    assert env["LD_LIBRARY_PATH"] == "/opt/mujoco210/bin"
    assert env["MUJOCO_PY_MUJOCO_PATH"] == "/opt/mujoco210"
