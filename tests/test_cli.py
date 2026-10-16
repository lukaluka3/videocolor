"""CLI diagnostics should catch unsupported installs before downloading models."""

import sys
from types import SimpleNamespace

import pytest

from videocolor import cli


@pytest.mark.parametrize("ffmpeg_version,torch_version,expected_message", [
    ("5.1.6", "2.11.0+cpu", "ffmpeg 6 or newer is required"),
    ("8.1", "2.5.0", "PyTorch 2.6 or newer is required"),
])
def test_doctor_reports_unsupported_runtime_versions(
    monkeypatch, capsys, ffmpeg_version, torch_version, expected_message,
):
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)

    def run(command, **kwargs):
        if "-encoders" in command:
            return SimpleNamespace(returncode=0, stdout="libx264 video\n aac audio")
        return SimpleNamespace(returncode=0, stdout=f"{command[0]} version {ffmpeg_version}\n")

    monkeypatch.setattr(cli.subprocess, "run", run)
    for name in ("numpy", "cv2", "huggingface_hub"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__=torch_version, cuda=SimpleNamespace(is_available=lambda: False),
    ))
    assert cli.doctor() == 1
    assert expected_message in capsys.readouterr().out


def test_process_error_is_printed_without_a_traceback(tmp_path, capsys):
    assert cli.main([
        "process", str(tmp_path / "missing.mp4"), "--output", str(tmp_path / "job"),
        "--duration", "nan",
    ]) == 1
    captured = capsys.readouterr()
    assert "positive finite" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "job").exists()


@pytest.mark.parametrize("strength", ["-0.1", "nan", "1.1"])
def test_invalid_color_strength_fails_before_creating_job(tmp_path, capsys, strength):
    assert cli.main([
        "process", str(tmp_path / "missing.mp4"), "--output", str(tmp_path / "job"),
        "--color-strength", strength,
    ]) == 1
    assert "Color strength" in capsys.readouterr().err
    assert not (tmp_path / "job").exists()


@pytest.mark.parametrize("option,value", [("--tint", "nan"), ("--tint", "21"), ("--warmth", "-21")])
def test_invalid_manual_balance_fails_before_creating_job(tmp_path, capsys, option, value):
    assert cli.main(["process", str(tmp_path / "missing.mp4"), "--output", str(tmp_path / "job"),
                     option, value]) == 1
    assert "between -20 and 20" in capsys.readouterr().err
    assert not (tmp_path / "job").exists()
