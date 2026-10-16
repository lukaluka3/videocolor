"""FFmpeg integration checks use generated media, never private footage."""

import shutil
from fractions import Fraction

import pytest

from videocolor.media import MediaError, extract_frames, inspect_video, output_dimensions, run_ffmpeg


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg integration tools unavailable"
)


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / "video with spaces & ქართული.mp4"
    run_ffmpeg([
        "-n", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30000/1001:duration=0.5",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-metadata", "comment=private source note", str(path),
    ])
    return path


def test_inspection_and_rational_frame_rate(movie):
    info = inspect_video(movie)
    assert (info["width"], info["height"]) == (96, 64)
    assert Fraction(info["fps"]) == Fraction(30000, 1001)
    assert info["has_audio"]
    assert info["duration"] > 0
    assert str(movie) not in str(info)
    assert "private source note" not in str(info)


def test_portrait_rotation_is_in_display_dimensions(movie, tmp_path):
    rotated = tmp_path / "rotated.mp4"
    run_ffmpeg(["-n", "-display_rotation", "90", "-i", str(movie), "-c", "copy", str(rotated)])
    info = inspect_video(rotated)
    assert (info["width"], info["height"]) == (64, 96)
    assert (info["coded_width"], info["coded_height"]) == (96, 64)
    frames = tmp_path / "frames"
    frames.mkdir()
    extract_frames(rotated, frames, info, start=0, duration=0.2, width=64, height=96)
    assert len(list(frames.glob("*.png"))) == 6


def test_dimensions_only_reduce_and_make_even():
    assert output_dimensions({"width": 1920, "height": 1080}, 720) == (1280, 720)
    assert output_dimensions({"width": 95, "height": 63}, 720) == (94, 62)
    assert output_dimensions({"width": 96, "height": 64}, None) == (96, 64)


def test_missing_or_nonvideo_source_fails(tmp_path):
    with pytest.raises(MediaError, match="does not exist"):
        inspect_video(tmp_path / "absent.mp4")
    invalid = tmp_path / "broken.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(MediaError, match="could not read"):
        inspect_video(invalid)


def test_failed_ffmpeg_is_not_reported_as_success():
    with pytest.raises(MediaError, match="FFmpeg failed"):
        run_ffmpeg(["-definitely-not-an-ffmpeg-option"])


def test_local_playlist_cannot_fetch_network_media(tmp_path):
    playlist = tmp_path / "remote.m3u8"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n"
        "#EXTINF:1,\nhttp://127.0.0.1:9/private.ts\n#EXT-X-ENDLIST\n"
    )
    with pytest.raises(MediaError, match="whitelist"):
        inspect_video(playlist)
