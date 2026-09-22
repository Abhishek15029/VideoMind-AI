import os
import shutil

import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ============================================================
# FIND DENO
# ============================================================

def get_deno_path():
    """
    Find Deno installed on the system.
    """

    deno_path = shutil.which("deno")

    if deno_path:
        print(f"Deno found: {deno_path}")
        return deno_path

    # Common Streamlit/Linux location
    possible_paths = [
        "/usr/bin/deno",
        "/usr/local/bin/deno",
        "/home/appuser/.deno/bin/deno",
        "/home/adminuser/.deno/bin/deno",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Deno found: {path}")
            return path

    print("WARNING: Deno was not found.")
    return None


# ============================================================
# YOUTUBE DOWNLOAD
# ============================================================

def download_youtube_audio(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    deno_path = get_deno_path()

    ydl_opts = {
        "format": "bestaudio*/best",

        "outtmpl": output_path,

        "noplaylist": True,

        "quiet": True,

        # ----------------------------------------------------
        # YouTube JavaScript runtime
        # ----------------------------------------------------
        #
        # Correct yt-dlp Python API format:
        #
        # "deno": {
        #     "path": "/path/to/deno"
        # }
        #
        # ----------------------------------------------------

        "js_runtimes": {
            "deno": {
                "path": deno_path
            }
        } if deno_path else {
            "deno": {}
        },

        # ----------------------------------------------------
        # Convert downloaded audio to WAV
        # ----------------------------------------------------

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    print("Downloading YouTube audio...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        filename = ydl.prepare_filename(info)

        base, _ = os.path.splitext(filename)

        wav_path = base + ".wav"

    return wav_path


# ============================================================
# LOCAL FILE → WAV
# ============================================================

def convert_to_wav(input_path: str) -> str:

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    audio = AudioSegment.from_file(input_path)

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav"
    )

    return output_path


# ============================================================
# CHUNK AUDIO
# ============================================================

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = (
            f"{wav_path}_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(chunk_path)

    return chunks


# ============================================================
# MAIN PROCESSOR
# ============================================================

def process_input(source: str) -> list:

    if (
        source.startswith("http://")
        or source.startswith("https://")
    ):

        print(
            "Detected YouTube URL. "
            "Downloading audio..."
        )

        wav_path = download_youtube_audio(source)

    else:

        print(
            "Detected local file. "
            "Converting to WAV..."
        )

        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(
        f"Audio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks