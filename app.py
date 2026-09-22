import streamlit as st
import os
import re
from pathlib import Path
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="VideoMind AI",
    page_icon="🎥",
    layout="wide"
)

DOWNLOAD_DIR = Path("downloads")


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(99,102,241,0.15),
        rgba(139,92,246,0.10)
    );
    border: 1px solid rgba(128,128,128,0.2);
    margin-bottom: 25px;
}

.hero h1 {
    margin-bottom: 5px;
}

.hero p {
    color: #888;
    font-size: 16px;
}

.stat-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.2);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "language" not in st.session_state:
    st.session_state.language = "english"

if "page" not in st.session_state:
    st.session_state.page = "New Video"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary" not in st.session_state:
    st.session_state.summary = ""


# ============================================================
# MISTRAL
# ============================================================

def get_mistral():

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        st.error("MISTRAL_API_KEY is missing in .env")
        return None

    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2,
        api_key=api_key
    )


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(transcript):

    llm = get_mistral()

    if llm is None:
        return None

    prompt = f"""
You are an AI assistant that summarizes video transcripts.

Create a clear and useful summary.

Include:

1. Short overview
2. Main points
3. Important details
4. Key takeaways

Do not invent information.

Transcript:
{transcript}
"""

    try:

        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content

    except Exception as e:

        if "429" in str(e) or "Rate limit" in str(e):

            st.error(
                "Mistral API rate limit reached. "
                "Please wait and try again."
            )

        else:

            st.error(f"Error generating summary: {e}")

        return None


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(transcript, question):

    llm = get_mistral()

    if llm is None:
        return None

    prompt = f"""
Answer the user's question using ONLY the transcript.

If the answer is not present in the transcript, say:

"I could not find this information in the transcript."

Do not invent information.

Transcript:
{transcript}

Question:
{question}
"""

    try:

        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content

    except Exception as e:

        if "429" in str(e) or "Rate limit" in str(e):

            st.error(
                "Mistral API rate limit reached. "
                "Please wait and try again."
            )

        else:

            st.error(f"Error answering question: {e}")

        return None


# ============================================================
# SEARCH TRANSCRIPT
# ============================================================

def search_transcript(transcript, query):

    if not query.strip():
        return []

    query_words = query.lower().split()

    paragraphs = re.split(
        r"\n\s*\n",
        transcript
    )

    results = []

    for paragraph in paragraphs:

        paragraph_lower = paragraph.lower()

        score = sum(
            1
            for word in query_words
            if word in paragraph_lower
        )

        if score > 0:
            results.append(
                (score, paragraph.strip())
            )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        text
        for score, text in results[:10]
    ]


# ============================================================
# FIND AUDIO FILES
# ============================================================

def get_audio_files():

    if not DOWNLOAD_DIR.exists():
        return []

    audio_extensions = [
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg"
    ]

    files = []

    for file in DOWNLOAD_DIR.iterdir():

        if file.is_file() and file.suffix.lower() in audio_extensions:
            files.append(file)

    files.sort(
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    return files


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎥 VideoMind AI")

    st.caption(
        "Video → Transcript → AI Tools"
    )

    st.divider()

    if st.button(
        "🎬 New Video",
        use_container_width=True
    ):
        st.session_state.page = "New Video"

    if st.button(
        "📄 Transcript",
        use_container_width=True,
        disabled=not bool(st.session_state.transcript)
    ):
        st.session_state.page = "Transcript"

    if st.button(
        "🛠️ Transcript Tools",
        use_container_width=True,
        disabled=not bool(st.session_state.transcript)
    ):
        st.session_state.page = "Tools"

    st.divider()

    if st.session_state.transcript:

        st.success("Transcript ready")

        word_count = len(
            st.session_state.transcript.split()
        )

        st.metric(
            "Words",
            word_count
        )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🎥 VideoMind AI</h1>

<p>
Turn YouTube videos or audio files into transcripts
and use AI tools to understand them.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# NEW VIDEO
# ============================================================

if st.session_state.page == "New Video":

    st.subheader("🎬 Add a Video")

    st.write(
        "Upload a video/audio file or provide a YouTube URL."
    )

    input_method = st.radio(
        "Input method",
        [
            "YouTube URL",
            "Upload File"
        ],
        horizontal=True
    )

    source = None

    # --------------------------------------------------------
    # YOUTUBE
    # --------------------------------------------------------

    if input_method == "YouTube URL":

        source = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=..."
        )

    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload video/audio",
            type=[
                "mp4",
                "mp3",
                "wav",
                "m4a",
                "webm",
                "mov",
                "avi"
            ]
        )

        if uploaded_file:

            os.makedirs(
                "uploads",
                exist_ok=True
            )

            file_path = os.path.join(
                "uploads",
                uploaded_file.name
            )

            with open(
                file_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            source = file_path

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    language = st.selectbox(
        "Transcription Language",
        [
            "english",
            "hinglish"
        ]
    )

    st.session_state.language = language

    st.divider()

    # --------------------------------------------------------
    # TRANSCRIBE
    # --------------------------------------------------------

    if st.button(
        "🎙️ Generate Transcript",
        type="primary",
        use_container_width=True
    ):

        if not source:

            st.warning(
                "Please provide a YouTube URL or upload a file."
            )

        else:

            try:

                progress = st.progress(0)

                status = st.empty()

                status.info(
                    "Preparing audio..."
                )

                progress.progress(20)

                chunks = process_input(
                    source
                )

                status.info(
                    "Audio prepared. Starting transcription..."
                )

                progress.progress(40)

                transcript = transcribe_all(
                    chunks,
                    language=language
                )

                progress.progress(100)

                status.success(
                    "Transcript generated successfully!"
                )

                st.session_state.transcript = transcript

                st.session_state.source_name = source

                st.session_state.chat_history = []

                st.session_state.summary = ""

                st.session_state.page = "Transcript"

                st.rerun()

            except Exception as e:

                st.error(
                    f"Processing failed: {e}"
                )


# ============================================================
# TRANSCRIPT READY
# ============================================================

elif st.session_state.page == "Transcript":

    transcript = st.session_state.transcript

    st.subheader("📄 Transcript Ready")

    st.write(
        "Your video has been transcribed. "
        "What would you like to do?"
    )

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    words = len(transcript.split())

    characters = len(transcript)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="stat-card">
            <h3>{words:,}</h3>
            <p>Words</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="stat-card">
            <h3>{characters:,}</h3>
            <p>Characters</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="stat-card">
            <h3>{st.session_state.language.title()}</h3>
            <p>Language</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    st.subheader("What would you like to do?")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📝 Summary",
            use_container_width=True
        ):

            st.session_state.page = "Summary"

            st.rerun()

        if st.button(
            "🔎 Search Transcript",
            use_container_width=True
        ):

            st.session_state.page = "Search"

            st.rerun()

        if st.button(
            "📄 Read Full Transcript",
            use_container_width=True
        ):

            st.session_state.page = "Full Transcript"

            st.rerun()

    with col2:

        if st.button(
            "💬 Ask Questions",
            use_container_width=True
        ):

            st.session_state.page = "Q&A"

            st.rerun()

        if st.button(
            "🎧 Audio",
            use_container_width=True
        ):

            st.session_state.page = "Audio"

            st.rerun()

    st.divider()

    st.subheader("Transcript Preview")

    st.text_area(
        "Preview",
        transcript[:3000],
        height=300,
        disabled=True
    )


# ============================================================
# SUMMARY
# ============================================================

elif st.session_state.page == "Summary":

    st.subheader("📝 AI Summary")

    st.caption(
        "Summary is generated only when you request it."
    )

    if not st.session_state.summary:

        if st.button(
            "✨ Generate Summary",
            type="primary"
        ):

            with st.spinner(
                "Generating summary..."
            ):

                summary = generate_summary(
                    st.session_state.transcript
                )

            if summary:

                st.session_state.summary = summary

                st.rerun()

    else:

        st.markdown(
            st.session_state.summary
        )

        st.download_button(
            "⬇️ Download Summary",
            st.session_state.summary,
            file_name="summary.txt",
            mime="text/plain"
        )

    st.divider()

    if st.button("← Back to Transcript"):

        st.session_state.page = "Transcript"

        st.rerun()


# ============================================================
# Q&A
# ============================================================

elif st.session_state.page == "Q&A":

    st.subheader("💬 Ask Questions")

    st.caption(
        "Ask questions about the generated transcript."
    )

    for item in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])

    question = st.chat_input(
        "Ask something about the video..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                answer = ask_question(
                    st.session_state.transcript,
                    question
                )

            if answer:

                st.write(answer)

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

    st.divider()

    if st.button("← Back to Transcript"):

        st.session_state.page = "Transcript"

        st.rerun()


# ============================================================
# SEARCH
# ============================================================

elif st.session_state.page == "Search":

    st.subheader("🔎 Search Transcript")

    st.caption(
        "Search locally inside the transcript. "
        "No Mistral API call is required."
    )

    query = st.text_input(
        "Search",
        placeholder="e.g. machine learning, Python, RAG..."
    )

    if query:

        results = search_transcript(
            st.session_state.transcript,
            query
        )

        if results:

            st.success(
                f"Found {len(results)} relevant section(s)."
            )

            for i, result in enumerate(
                results,
                start=1
            ):

                with st.expander(
                    f"Result {i}"
                ):

                    st.write(result)

        else:

            st.info(
                "No matching text found."
            )

    st.divider()

    if st.button("← Back to Transcript"):

        st.session_state.page = "Transcript"

        st.rerun()


# ============================================================
# FULL TRANSCRIPT
# ============================================================

elif st.session_state.page == "Full Transcript":

    st.subheader("📄 Full Transcript")

    st.text_area(
        "Transcript",
        st.session_state.transcript,
        height=600
    )

    st.download_button(
        "⬇️ Download Transcript",
        st.session_state.transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )

    st.divider()

    if st.button("← Back to Transcript"):

        st.session_state.page = "Transcript"

        st.rerun()


# ============================================================
# AUDIO
# ============================================================

elif st.session_state.page == "Audio":

    st.subheader("🎧 Audio")

    st.write(
        "Your generated audio files are available here."
    )

    audio_files = get_audio_files()

    if not audio_files:

        st.warning(
            "No audio files were found in the downloads folder."
        )

        st.info(
            "Generate the audio using your existing audio-generation code, "
            "then return here and refresh the page."
        )

    else:

        st.success(
            f"{len(audio_files)} audio file(s) found."
        )

        # ----------------------------------------------------
        # SELECT AUDIO
        # ----------------------------------------------------

        selected_audio = st.selectbox(
            "Select audio",
            audio_files,
            format_func=lambda x: x.name
        )

        if selected_audio:

            st.markdown(
                f"### 🎵 {selected_audio.name}"
            )

            # Audio player
            with open(
                selected_audio,
                "rb"
            ) as audio_file:

                audio_bytes = audio_file.read()

            st.audio(
                audio_bytes,
                format=f"audio/{selected_audio.suffix[1:]}"
            )

            st.download_button(
                "⬇️ Download Audio",
                data=audio_bytes,
                file_name=selected_audio.name,
                mime=f"audio/{selected_audio.suffix[1:]}"
            )

    st.divider()

    if st.button("← Back to Transcript"):

        st.session_state.page = "Transcript"

        st.rerun()


# ============================================================
# TOOLS
# ============================================================

elif st.session_state.page == "Tools":

    st.subheader("🛠️ Transcript Tools")

    st.write(
        "Choose an action to work with your transcript."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📝 Summary",
            use_container_width=True
        ):

            st.session_state.page = "Summary"

            st.rerun()

        if st.button(
            "🔎 Search",
            use_container_width=True
        ):

            st.session_state.page = "Search"

            st.rerun()

    with col2:

        if st.button(
            "💬 Ask Questions",
            use_container_width=True
        ):

            st.session_state.page = "Q&A"

            st.rerun()

        if st.button(
            "🎧 Audio",
            use_container_width=True
        ):

            st.session_state.page = "Audio"

            st.rerun()