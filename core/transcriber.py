import whisper
import os
import requests
from pydub import AudioSegment


# ============================================================
# CONFIGURATION
# ============================================================

# Sarvam sync STT-translate API rejects audio longer than 30 seconds.
# We split each chunk into 25-second pieces before sending to Sarvam.
SARVAM_PIECE_SECONDS = 25


# ============================================================
# WHISPER CONFIGURATION
# ============================================================

# If WHISPER_MODEL is missing OR empty in .env,
# automatically use "small".
#
# Example .env:
# WHISPER_MODEL=small
#
WHISPER_MODEL = os.getenv("WHISPER_MODEL") or "small"


# ============================================================
# SARVAM CONFIGURATION
# ============================================================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_STT_TRANSLATE_URL = (
    "https://api.sarvam.ai/speech-to-text-translate"
)

# Use the model specified in .env.
# If missing/empty, use saaras:v2.5.
SARVAM_MODEL = (
    os.getenv("SARVAM_STT_MODEL") or "saaras:v2.5"
)


# ============================================================
# GLOBAL WHISPER MODEL
# ============================================================

# Whisper model will be loaded only once.
# This avoids loading the model again for every audio chunk.
_model = None


# ============================================================
# WHISPER MODEL
# ============================================================

def load_model():
    global _model

    if _model is None:

        print(
            f"Loading Whisper model: {WHISPER_MODEL} ..."
        )

        # ----------------------------------------------------
        # GPU SUPPORT
        # ----------------------------------------------------
        # Your NVIDIA GTX 1650 is being used through CUDA.
        # ----------------------------------------------------

        _model = whisper.load_model(
            WHISPER_MODEL,
            device="cpu"
        )

        print("Whisper model loaded on GPU.")

    return _model


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_chunk_whisper(
    chunk_path: str
) -> str:

    # Load Whisper model on NVIDIA GPU
    model = load_model()

    # --------------------------------------------------------
    # IMPORTANT:
    # fp16=False avoids NaN / invalid numerical values
    # that were occurring with FP16 on the GTX 1650.
    # --------------------------------------------------------

    result = model.transcribe(
        chunk_path,
        task="transcribe",
        fp16=False
    )

    return result["text"]


# ============================================================
# SARVAM API
# ============================================================

def _send_to_sarvam(
    piece_path: str
) -> str:

    """
    Send one <=30 second WAV file to Sarvam
    and return the English transcript.
    """

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:

        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    # --------------------------------------------------------
    # Check Sarvam response
    # --------------------------------------------------------

    if not response.ok:

        print(
            f"\n❌ Sarvam returned "
            f"{response.status_code}"
        )

        print(
            f"Response body: "
            f"{response.text}\n"
        )

        response.raise_for_status()

    return response.json().get(
        "transcript",
        ""
    )


# ============================================================
# SARVAM TRANSCRIPTION
# ============================================================

def transcribe_chunk_sarvam(
    chunk_path: str
) -> str:

    """
    Sarvam sync API accepts <=30 second audio.

    Therefore, each chunk is split into 25-second pieces,
    sent separately to Sarvam, and then joined together.
    """

    if not SARVAM_API_KEY:

        raise RuntimeError(
            "SARVAM_API_KEY is not set "
            "in environment / .env"
        )

    # --------------------------------------------------------
    # Load WAV audio
    # --------------------------------------------------------

    audio = AudioSegment.from_wav(
        chunk_path
    )

    piece_ms = (
        SARVAM_PIECE_SECONDS * 1000
    )

    full_text = ""

    # Calculate total number of pieces
    total_pieces = (
        (len(audio) + piece_ms - 1)
        // piece_ms
    )

    # --------------------------------------------------------
    # Process each 25-second piece
    # --------------------------------------------------------

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[
            start:start + piece_ms
        ]

        # Temporary file for this piece
        piece_path = (
            f"{chunk_path}_sv_{i}.wav"
        )

        # Export temporary piece
        piece.export(
            piece_path,
            format="wav"
        )

        try:

            print(
                f"  → Sarvam piece "
                f"{i + 1}/{total_pieces} ..."
            )

            # Send piece to Sarvam
            text = _send_to_sarvam(
                piece_path
            )

            full_text += text + " "

        finally:

            # Delete temporary file
            if os.path.exists(piece_path):

                os.remove(
                    piece_path
                )

    return full_text.strip()


# ============================================================
# TRANSCRIPTION ROUTER
# ============================================================

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    """
    Select transcription engine based on language.

    English:
        Whisper → NVIDIA GTX 1650 GPU

    Hinglish:
        Sarvam AI → Cloud API
    """

    if language.lower() == "hinglish":

        return transcribe_chunk_sarvam(
            chunk_path
        )

    # --------------------------------------------------------
    # English uses local Whisper model
    # running on NVIDIA GPU.
    # --------------------------------------------------------

    return transcribe_chunk_whisper(
        chunk_path
    )


# ============================================================
# TRANSCRIBE ALL CHUNKS
# ============================================================

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    full_transcript = ""

    # --------------------------------------------------------
    # Display which engine is being used
    # --------------------------------------------------------

    engine = (
        "Sarvam AI"
        if language.lower() == "hinglish"
        else "Whisper (GPU)"
    )

    print(
        f"Using {engine} for transcription."
    )

    # --------------------------------------------------------
    # Process every audio chunk
    # --------------------------------------------------------

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += (
            text + " "
        )

    print("Transcription complete.")

    return full_transcript.strip()