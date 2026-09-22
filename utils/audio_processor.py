import os
import stat
import zipfile
import urllib.request
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ============================================================
# DENO SETUP
# ============================================================

def setup_deno():
    """
    Make sure Deno is available.
    Works locally and on Streamlit Cloud.
    """

    # Check if Deno is already installed
    deno_path = "deno"

    try:
        import shutil

        existing_deno = shutil.which("deno")

        if existing_deno:
            print(f"Deno found: {existing_deno}")
            return existing_deno

    except Exception:
        pass

    # Local/cache location for Streamlit Cloud
    deno_dir = Path("/tmp/deno")
    deno_exe = deno_dir / "deno"

    if deno_exe.exists():
        print(f"Deno found at: {deno_exe}")
        return str(deno_exe)

    print("Deno not found. Installing Deno...")

    deno_dir.mkdir(parents=True, exist_ok=True)

    zip_path = deno_dir / "deno.zip"

    # Linux x86_64 binary used by Streamlit Cloud
    deno_url = (
        "https://github.com/denoland/deno/releases/latest/"
        "download/deno-x86_64-unknown-linux-gnu.zip"
    )

    try:
        urllib.request.urlretrieve(deno_url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(deno_dir)

        # Make executable
        deno_exe.chmod(
            deno_exe.stat().st_mode
            | stat.S_IEXEC
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

        zip_path.unlink(missing_ok=True)

        print(f"Deno installed at: {deno_exe}")

        return str(deno_exe)

    except Exception as e:
        print(f"Failed to install Deno: {e}")
        raise


# ============================================================
# YOUTUBE DOWNLOAD
# ============================================================

def download_youtube_audio(url: str) -> str:

    deno_path = setup_deno()

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {

        # Audio
        "format": "bestaudio*/best",

        "outtmpl": output_path,

        # Use Deno for YouTube JavaScript challenges
        "js_runtimes": {
            "deno": deno_path
        },

        # Allow yt-dlp to obtain EJS if required
        "remote_components": {
            "ejs": "github"
        },

        # Convert to WAV
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": True,

        "noplaylist": True,

        # Avoid unnecessary playlist processing
        "nocheckcertificate": True,
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
# AUDIO CHUNKING
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
# MAIN INPUT PROCESSOR
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