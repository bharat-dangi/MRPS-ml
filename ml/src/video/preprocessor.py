import os
import tempfile

import ffmpeg
import noisereduce as nr
import numpy as np
import soundfile as sf


def extract_audio(video_path: str) -> str:
    """
    Extract audio from video using ffmpeg-python.
    Outputs pcm_s16le WAV at 16kHz mono — required by Whisper.
    Returns path to the temporary WAV file.
    """
    suffix = ".wav"
    fd, output_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        (
            ffmpeg
            .input(video_path)
            .output(
                output_path,
                acodec="pcm_s16le",
                ar=16000,
                ac=1,
                vn=None,  # no video stream
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr.decode()}") from e

    # Noise reduction to improve Whisper accuracy
    audio_data, sample_rate = sf.read(output_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    reduced = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.75)
    sf.write(output_path, reduced, sample_rate)

    return output_path
