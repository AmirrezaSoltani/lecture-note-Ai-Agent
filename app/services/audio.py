from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class AudioPreprocessor:
    def normalize_to_wav(self, input_path: Path, output_path: Path) -> Path:
        """
        Convert the audio into a deterministic mono 16kHz WAV.
        Uses a light normalization filter to improve transcription stability.
        Falls back to simple copy when ffmpeg is unavailable and input is already WAV.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            if input_path.suffix.lower() == ".wav":
                shutil.copy2(input_path, output_path)
                return output_path
            raise RuntimeError("ffmpeg is required for non-WAV inputs.")

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-vn",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
        return output_path
