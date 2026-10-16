"""Desktop settings and resume recovery without a display or network."""

import json

import pytest

from videocolor.gui import load_resume_settings, parse_settings


@pytest.fixture
def values(tmp_path):
    source = tmp_path / "original clip.mp4"
    source.write_bytes(b"fixture name; no media decoding in GUI validation")
    return {
        "source": str(source), "output": str(tmp_path / "job"), "model": "tiny",
        "device": "cpu", "start": "0", "duration": "3", "height": "240",
        "smoothing": "0.35", "full": False, "resume": False,
        "no_resize": False, "input_size": "256", "color_strength": "0.85",
    }


def test_full_video_and_preserve_size_ignore_inactive_fields(values):
    values.update(full=True, no_resize=True, duration="", height="")
    result = parse_settings(values)
    assert result["duration"] is None
    assert result["max_height"] is None
    assert result["input_size"] == 256
    assert result["color_strength"] == 0.85


@pytest.mark.parametrize("updates", [
    {"start": "nan"}, {"duration": "0"}, {"height": "1"},
    {"smoothing": "inf"}, {"model": "unknown"}, {"device": "cuda:1"},
    {"source": ""}, {"output": " "}, {"input_size": "1024"},
    {"color_strength": "nan"}, {"color_strength": "-0.1"}, {"color_strength": "1.1"},
    {"tint": "nan"}, {"tint": "21"}, {"warmth": "-21"},
])
def test_gui_rejects_invalid_settings_before_work(values, updates):
    values.update(updates)
    with pytest.raises(ValueError):
        parse_settings(values)


@pytest.mark.parametrize("balance", [{}, {"tint": -6, "warmth": 3}])
def test_resume_restores_cli_settings_without_storing_a_source_path(tmp_path, values, balance):
    job = {
        "format": "videocolor-job", "version": 1,
        "input": {"name": "original clip.mp4"},
        "config": {
            "model": "large", "device": "cpu", "input_size": 512,
            "start": 18.5, "duration": None, "max_height": None, "smoothing": 0.2,
            "color_strength": 0.7, **balance,
        },
    }
    (tmp_path / "videocolor-job.json").write_text(json.dumps(job), encoding="utf-8")
    restored, name = load_resume_settings(tmp_path)
    assert name == "original clip.mp4"
    assert "source" not in restored
    values.update(restored)
    parsed = parse_settings(values)
    assert parsed == dict(model="large", device="cpu", input_size=512, start=18.5,
                          duration=None, max_height=None, smoothing=0.2, color_strength=0.7,
                          tint=balance.get("tint", 0.0), warmth=balance.get("warmth", 0.0), resume=True)


@pytest.mark.parametrize("manifest", [
    "[]", "{broken", '{"format":"unknown"}',
    '{"format":"videocolor-job","version":1,"config":{}}',
])
def test_invalid_resume_manifest_reports_a_useful_error(tmp_path, manifest):
    (tmp_path / "videocolor-job.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load this VideoColor job"):
        load_resume_settings(tmp_path)
