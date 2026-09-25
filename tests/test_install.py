"""Installer checks without changing the environment or downloading wheels."""

import subprocess
import sys
from unittest.mock import Mock

import pytest

from tools import install


@pytest.fixture
def pip_run(monkeypatch):
    run = Mock()
    monkeypatch.setattr(install.subprocess, "run", run)
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    return run


def test_legacy_installs_cuda_before_project_and_keeps_pin(pip_run, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    install.main(["--legacy-cuda", "--notebook", "--test"])
    first, second = pip_run.call_args_list
    assert first.args[0] == [
        sys.executable, "-m", "pip", "install", "torch==2.14.0+cu126",
        "--index-url", "https://download.pytorch.org/whl/cu126",
    ]
    assert second.args[0] == [
        sys.executable, "-m", "pip", "install", "-e",
        str(install.ROOT) + "[notebook,test]", "torch==2.14.0+cu126",
    ]
    assert first.kwargs == second.kwargs == {"check": True}


def test_default_leaves_torch_selection_to_pip(pip_run):
    install.main([])
    pip_run.assert_called_once_with(
        [sys.executable, "-m", "pip", "install", "-e", str(install.ROOT)],
        check=True,
    )


def test_dry_run_never_installs(pip_run, capsys):
    install.main(["--legacy-cuda", "--dry-run"])
    pip_run.assert_not_called()
    assert "torch==2.14.0+cu126" in capsys.readouterr().out


def test_failed_cuda_install_stops_before_project(pip_run):
    pip_run.side_effect = subprocess.CalledProcessError(1, "pip")
    with pytest.raises(subprocess.CalledProcessError):
        install.main(["--legacy-cuda"])
    assert pip_run.call_count == 1


def test_unsupported_platform_fails_before_install(pip_run, monkeypatch):
    monkeypatch.setattr(install.platform, "system", lambda: "Darwin")
    with pytest.raises(SystemExit) as error:
        install.main(["--legacy-cuda"])
    assert error.value.code == 2
    pip_run.assert_not_called()
