"""Transcode pipeline MP4 output to H.264/AVC (OpenCV typically writes MPEG-4 Part 2 ``mp4v``)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def transcode_mp4_to_h264(path: Path, *, crf: str = "23", preset: str = "medium") -> bool:
    """
    Replace ``path`` in place with an H.264 (AVC) MP4: yuv420p, ``faststart`` for streaming.

    OpenCV ``VideoWriter`` usually emits MPEG-4 Visual, which many players reject; H.264 is
    the common interchange format for MP4.

    Returns True if transcoding ran, False if ffmpeg was missing or transcoding failed.
    """
    path = Path(path).resolve()
    if not path.is_file():
        return False

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "Warning: ffmpeg not on PATH; leaving OpenCV output (often MPEG-4 Visual). "
            "Install ffmpeg to transcode to H.264."
        )
        return False

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                preset,
                "-crf",
                crf,
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        os.replace(tmp, path)
        print(f"Transcoded to H.264 (AVC): {path.name}")
        return True
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        err = (exc.stderr or exc.stdout or "").strip()
        print(f"Warning: ffmpeg transcode failed; left OpenCV output. {err[:300]}")
        return False
