from dotenv import load_dotenv
load_dotenv()

import os

from utils.audio_processor import process_input
from core.transcriber import transcribe_all


print("KEY LOADED:", bool(os.getenv("SARVAM_API_KEY")))
print("CWD:", os.getcwd())

source = "https://youtu.be/vFP1mgZ_LEY?si=6LwnE5yteem8W0hE"
language = "hinglish" #change to "hinglish" to test sarvam
# if english whisper will work on that and for hinglish sarvam will work on that

chunks = process_input(source)

transcript = transcribe_all(chunks, language=language)

print("\n=== TRANSCRIPT ===\n")
print(transcript)