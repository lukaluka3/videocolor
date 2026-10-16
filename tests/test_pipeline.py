"""Media/resume integration uses a deterministic fake, not a claim of AI quality."""

import hashlib
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from videocolor.media import MediaError, _probe, executable, inspect_video, run_ffmpeg
from videocolor.pipeline import JOB_FILE, JobError, _job_lock, process_video


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg integration tools unavailable"
)


class FakeColorizer:
    """Deterministic test image transform; this is NOT the production backend."""
    model_info = {"name": "test-fake", "device": "cpu"}

    def __init__(self):
        self.calls = 0

    def colorize(self, source):
        self.calls += 1
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        lab[:, :, 1] = 132
        lab[:, :, 2] = 145
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@pytest.fixture
def make_video(tmp_path):
    def make(name="input.mp4", audio=False, audio_duration=1.2):
        path = tmp_path / name
        args = ["-n", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=10:duration=1.2"]
        if audio:
            args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={audio_duration}"]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if audio:
            args += ["-c:a", "aac", "-metadata:s:a:0", "title=PRIVATE-AUDIO-TITLE",
                     "-metadata:s:a:0", "handler_name=PRIVATE-AUDIO-HANDLER"]
        args += ["-metadata", "comment=PRIVATE-TEST-SOURCE", str(path)]
        run_ffmpeg(args)
        return path
    return make


@pytest.mark.parametrize("audio", [False, True])
def test_preview_preserves_count_sound_and_removes_metadata(make_video, tmp_path, audio):
    source = make_video(audio=audio)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    colorizer = FakeColorizer()
    progress = []
    result = process_video(source, tmp_path / "out", duration=0.8, _colorizer=colorizer,
                           progress=lambda completed, total: progress.append((completed, total)))
    assert result.is_file()
    assert colorizer.calls == 8
    assert progress[0] == (0, 8) and progress[-1] == (8, 8)
    probe = _probe(result, count_frames=True)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert int(video["nb_read_frames"]) == 8
    assert inspect_video(result)["has_audio"] == audio
    assert "PRIVATE-TEST-SOURCE" not in str(probe)
    assert "PRIVATE-AUDIO" not in str(probe)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    job = json.loads((result.parent / JOB_FILE).read_text())
    assert job["status"] == "complete"
    assert str(source.parent) not in str(job)


def test_short_audio_does_not_truncate_video(make_video, tmp_path):
    source = make_video(audio=True, audio_duration=0.25)
    result = process_video(source, tmp_path / "out", duration=1, _colorizer=FakeColorizer())
    probe = _probe(result, count_frames=True)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert int(video["nb_read_frames"]) == 10
    assert float(audio["duration"]) >= 0.95


def _audio_samples(path):
    result = subprocess.run([
        executable("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-f", "f32le", "-ac", "1", "-ar", "48000", "pipe:1",
    ], capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def test_delayed_audio_keeps_its_initial_silence_and_trim_alignment(tmp_path):
    source = tmp_path / "late-audio.mp4"
    run_ffmpeg([
        "-n", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=10:duration=1.2",
        "-itsoffset", "0.4", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.8",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
    ])
    result = process_video(source, tmp_path / "out", duration=0.8, _colorizer=FakeColorizer())
    samples = _audio_samples(result)
    assert np.sqrt(np.mean(samples[:12000] ** 2)) < 0.001
    assert np.sqrt(np.mean(samples[24000:33600] ** 2)) > 0.02
    trimmed = process_video(source, tmp_path / "trimmed", start=0.5, duration=0.5, _colorizer=FakeColorizer())
    samples = _audio_samples(trimmed)
    assert np.sqrt(np.mean(samples[:9600] ** 2)) > 0.02


def test_interval_after_audio_ended_is_filled_with_silence(make_video, tmp_path):
    source = make_video(audio=True, audio_duration=0.25)
    result = process_video(source, tmp_path / "out", start=0.6, duration=0.5, _colorizer=FakeColorizer())
    samples = _audio_samples(result)
    assert len(samples) >= 24000
    assert np.max(abs(samples)) < 0.001


def test_late_starting_video_uses_first_video_frame_as_time_zero(tmp_path):
    source = tmp_path / "late-video.mp4"
    run_ffmpeg([
        "-n", "-itsoffset", "0.4", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=10:duration=0.8",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.2",
        "-c:v", "libx264", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
    ])
    info = inspect_video(source)
    assert info["video_start_offset"] == pytest.approx(0.4, abs=0.02)
    result = process_video(source, tmp_path / "out", duration=0.5, _colorizer=FakeColorizer())
    probe = _probe(result, count_frames=True)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert int(video["nb_read_frames"]) == 5
    assert np.sqrt(np.mean(_audio_samples(result)[:9600] ** 2)) > 0.02


@pytest.mark.parametrize("balance", [{}, {"tint": -6.0, "warmth": 3.0}])
def test_resume_preserves_completed_frames_and_releases_lock(make_video, tmp_path, balance):
    source = make_video()
    output = tmp_path / "out"

    def cancel(completed, total):
        if completed == 3:
            raise InterruptedError("Test cancellation")

    with pytest.raises(InterruptedError):
        process_video(source, output, duration=0.8, progress=cancel, _colorizer=FakeColorizer(), **balance)
    before = {p.name: p.read_bytes() for p in (output / "colored").glob("*.png")}
    assert len(before) == 3
    assert json.loads((output / JOB_FILE).read_text())["status"] == "interrupted"
    colorizer = FakeColorizer()
    result = process_video(source, output, duration=0.8, resume=True, _colorizer=colorizer, **balance)
    assert result.exists()
    assert colorizer.calls == 5
    assert all((output / "colored" / name).read_bytes() == data for name, data in before.items())
    fresh = process_video(source, tmp_path / "fresh", duration=0.8, _colorizer=FakeColorizer(), **balance)
    assert [p.read_bytes() for p in sorted((output / "colored").glob("*.png"))] == [
        p.read_bytes() for p in sorted((fresh.parent / "colored").glob("*.png"))
    ]
    no_calls = FakeColorizer()
    assert process_video(source, output, duration=0.8, resume=True, _colorizer=no_calls, **balance) == result
    assert no_calls.calls == 0


def test_unrelated_directory_and_existing_job_are_protected(make_video, tmp_path):
    source = make_video()
    unrelated = tmp_path / "personal"
    unrelated.mkdir()
    secret = unrelated / "keep.txt"
    secret.write_text("keep me")
    with pytest.raises(JobError, match="not an owned"):
        process_video(source, unrelated, _colorizer=FakeColorizer())
    assert secret.read_text() == "keep me"
    output = tmp_path / "out"
    process_video(source, output, duration=0.2, _colorizer=FakeColorizer())
    with pytest.raises(JobError, match="already contains"):
        process_video(source, output, duration=0.2, _colorizer=FakeColorizer())
    with pytest.raises(JobError, match="same input"):
        process_video(source, output, duration=0.3, resume=True, _colorizer=FakeColorizer())
    source.write_bytes(source.read_bytes() + b"changed source bytes")
    with pytest.raises(JobError, match="same input"):
        process_video(source, output, duration=0.2, resume=True, _colorizer=FakeColorizer())


def test_resume_rejects_mixing_old_and_new_color_processing(make_video, tmp_path):
    source = make_video()
    output = tmp_path / "old-job"

    def stop(done, total):
        if done == 2:
            raise InterruptedError("Pause before rendering the rest")

    with pytest.raises(InterruptedError):
        process_video(source, output, duration=0.4, _colorizer=FakeColorizer(), progress=stop)
    marker = output / JOB_FILE
    job = json.loads(marker.read_text())
    job.pop("color_pipeline_revision")  # Jobs from the first release had no revision.
    marker.write_text(json.dumps(job), encoding="utf-8")
    saved = {p.name: p.read_bytes() for p in (output / "colored").glob("*.png")}
    colorizer = FakeColorizer()
    with pytest.raises(JobError, match="different color-processing version"):
        process_video(source, output, duration=0.4, _colorizer=colorizer, resume=True)
    assert colorizer.calls == 0
    assert saved == {p.name: p.read_bytes() for p in (output / "colored").glob("*.png")}


def test_completed_older_job_can_still_be_opened_without_rerendering(make_video, tmp_path):
    source = make_video()
    output = tmp_path / "complete-old-job"
    result = process_video(source, output, duration=0.2, _colorizer=FakeColorizer())
    before = result.read_bytes()
    marker = output / JOB_FILE
    job = json.loads(marker.read_text())
    job.pop("color_pipeline_revision")
    job["config"].pop("tint")
    job["config"].pop("warmth")
    marker.write_text(json.dumps(job), encoding="utf-8")
    colorizer = FakeColorizer()
    assert process_video(source, output, duration=0.2, _colorizer=colorizer, resume=True) == result
    assert colorizer.calls == 0
    assert result.read_bytes() == before


def test_corrupt_completed_frame_is_not_silently_reused(make_video, tmp_path):
    source = make_video()
    output = tmp_path / "out"
    process_video(source, output, duration=0.2, _colorizer=FakeColorizer())
    (output / "colored" / "00000001.png").write_bytes(b"broken")
    with pytest.raises(JobError, match="Invalid frame"):
        process_video(source, output, duration=0.2, resume=True, _colorizer=FakeColorizer())


@pytest.mark.parametrize("older_renderer", [False, True])
def test_failed_encoding_can_resume_without_recoloring(make_video, tmp_path, monkeypatch, older_renderer):
    import videocolor.pipeline as pipeline
    source = make_video()
    output = tmp_path / "out"
    real_encode = pipeline.encode_video

    def failed_encode(source, frames, destination, info, **options):
        destination.write_bytes(b"partial-output")
        raise MediaError("Simulated encoder failure")

    monkeypatch.setattr(pipeline, "encode_video", failed_encode)
    with pytest.raises(MediaError, match="encoder failure"):
        process_video(source, output, duration=0.3, _colorizer=FakeColorizer())
    assert not (output / "result.mp4").exists()
    assert json.loads((output / JOB_FILE).read_text())["status"] == "failed"
    if older_renderer:
        marker = output / JOB_FILE
        job = json.loads(marker.read_text())
        job.pop("color_pipeline_revision")
        marker.write_text(json.dumps(job), encoding="utf-8")
    monkeypatch.setattr(pipeline, "encode_video", real_encode)
    colorizer = FakeColorizer()
    assert process_video(source, output, duration=0.3, resume=True, _colorizer=colorizer).exists()
    assert colorizer.calls == 0


def test_resume_does_not_mix_changed_model_revisions(make_video, tmp_path):
    source = make_video()
    output = tmp_path / "out"
    colorizer = FakeColorizer()
    colorizer.model_info = {"device": "cpu", "sha256": "original-model"}

    def cancel(completed, total):
        if completed == 1:
            raise InterruptedError("Test cancellation")

    with pytest.raises(InterruptedError):
        process_video(source, output, duration=0.3, progress=cancel, _colorizer=colorizer)
    colorizer.model_info = {"device": "cpu", "sha256": "different-model"}
    with pytest.raises(JobError, match="model revision"):
        process_video(source, output, duration=0.3, resume=True, _colorizer=colorizer)
    assert len(list((output / "colored").glob("*.png"))) == 1


def test_color_strength_is_saved_and_must_match_to_resume(make_video, tmp_path):
    source = make_video()
    output = tmp_path / "out"
    process_video(source, output, duration=0.2, color_strength=0.4, _colorizer=FakeColorizer())
    job = json.loads((output / JOB_FILE).read_text())
    assert job["config"]["color_strength"] == 0.4
    with pytest.raises(JobError, match="same input"):
        process_video(source, output, duration=0.2, color_strength=0.85, resume=True, _colorizer=FakeColorizer())
    assert process_video(source, output, duration=0.2, color_strength=0.4, resume=True, _colorizer=FakeColorizer()).is_file()


def test_lock_prevents_concurrent_jobs(tmp_path):
    with _job_lock(tmp_path):
        with pytest.raises(JobError, match="Another process"):
            with _job_lock(tmp_path):
                pass


def test_resume_missing_job_does_not_leave_an_unusable_directory(make_video, tmp_path):
    source = make_video()
    output = tmp_path / "missing-job"
    with pytest.raises(JobError, match="no saved job"):
        process_video(source, output, resume=True, _colorizer=FakeColorizer())
    assert not output.exists()


def test_low_disk_space_fails_before_frame_extraction(make_video, tmp_path, monkeypatch):
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("videocolor.pipeline.shutil.disk_usage", lambda _: usage(100, 99, 1))
    source = make_video()
    output = tmp_path / "out"
    with pytest.raises(JobError, match="Insufficient disk space"):
        process_video(source, output, duration=0.5, _colorizer=FakeColorizer())
    assert not list(output.glob("source-*"))
    assert json.loads((output / JOB_FILE).read_text())["status"] == "failed"


@pytest.mark.parametrize("options", [
    {"start": -1}, {"start": float("nan")}, {"duration": 0}, {"duration": float("inf")},
    {"smoothing": 1.1}, {"max_height": 1}, {"input_size": 33},
    {"color_strength": -0.01}, {"color_strength": 1.01}, {"color_strength": float("nan")},
])
def test_invalid_options_rejected_before_work(tmp_path, options):
    with pytest.raises(ValueError):
        process_video(tmp_path / "missing.mp4", tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()
