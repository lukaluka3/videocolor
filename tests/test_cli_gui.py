import math

import pytest

from videocolor.cli import main, parser
from videocolor.gui import parse_settings


def test_cli_failed_input_returns_failure(capsys):
    assert main(["info", "missing-file-42.mp4"]) == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_conflicting_duration_rejected():
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(["process", "input.mp4", "--output", "job", "--full-video", "--duration", "10"])
    assert exc.value.code == 2


@pytest.fixture
def gui_values(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    return dict(source=str(source), output=str(tmp_path / "job"), model="tiny", device="cpu",
                start="0", duration="1", height="240", smoothing="0.35", full=False, resume=False)


@pytest.mark.parametrize("key,value", [("start", "nan"), ("start", "-1"), ("duration", "0"),
                                     ("duration", "inf"), ("height", "1"), ("smoothing", "nan"),
                                     ("smoothing", "1.1"), ("output", ""), ("source", "missing-file-42.mp4")])
def test_gui_rejects_invalid_input_before_work(gui_values, key, value):
    gui_values[key] = value
    with pytest.raises(ValueError):
        parse_settings(gui_values)


def test_gui_full_video_and_resume(gui_values):
    gui_values.update(full=True, resume=True)
    settings = parse_settings(gui_values)
    assert settings["duration"] is None
    assert settings["resume"] is True
    assert math.isclose(settings["smoothing"], 0.35)
