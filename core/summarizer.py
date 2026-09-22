import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

# ============================================================
# MISTRAL MODEL
# ============================================================

llm = ChatMistralAI(
    model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)


# ============================================================
# SAFE MISTRAL CALL
# ============================================================

def safe_invoke(prompt: str, fallback: str) -> str:
    """
    Calls Mistral safely.

    If Mistral is rate-limited or unavailable,
    return a fallback instead of crashing the application.
    """

    try:
        response = llm.invoke(prompt)

        if hasattr(response, "content"):
            return response.content.strip()

        return str(response).strip()

    except Exception as e:

        error_text = str(e)

        if "429" in error_text or "Rate limit exceeded" in error_text:

            print("⚠️ Mistral rate limit reached.")
            print("Using fallback response.")

            return fallback

        print(f"⚠️ Mistral error: {e}")

        return fallback


# ============================================================
# TITLE
# ============================================================

def generate_title(transcript: str) -> str:

    prompt = f"""
Generate a short and professional title for this video transcript.

Transcript:
{transcript[:2000]}

Return only the title.
"""

    return safe_invoke(
        prompt,
        "Video Analysis"
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(transcript: str) -> str:

    prompt = f"""
Summarize the following video transcript.

Give a concise summary containing the most important
information and discussion points.

Transcript:
{transcript[:12000]}
"""

    return safe_invoke(
        prompt,
        "AI summary is temporarily unavailable because the Mistral API rate limit was reached."
    )