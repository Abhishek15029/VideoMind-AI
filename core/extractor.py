import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# ============================================================
# LLM
# ============================================================

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2
    )


# ============================================================
# SAFE CHAIN CALL
# ============================================================

def run_chain(chain, transcript, fallback):

    try:
        return chain.invoke(transcript)

    except Exception as e:

        error = str(e)

        if "429" in error or "Rate limit exceeded" in error:

            print("⚠️ Mistral rate limit reached.")
            print("Using fallback result.")

            return fallback

        raise


# ============================================================
# BUILD CHAIN
# ============================================================

def build_chain(system_prompt: str):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )


# ============================================================
# ACTION ITEMS
# ============================================================

def extract_action_items(transcript: str) -> str:

    chain = build_chain(
        """
        You are an expert meeting analyst.

        From the meeting transcript, extract all action items.

        For each action item provide:
        - Task description
        - Owner
        - Deadline

        If the owner or deadline is not mentioned,
        write "Not specified".

        Format as a numbered list.

        If no action items are found, write:
        "No action items found."
        """
    )

    return run_chain(
        chain,
        transcript,
        "AI action-item extraction is temporarily unavailable because the Mistral API rate limit was reached."
    )


# ============================================================
# KEY DECISIONS
# ============================================================

def extract_key_decisions(transcript: str) -> str:

    chain = build_chain(
        """
        You are an expert meeting analyst.

        From the meeting transcript, extract all key decisions made.

        Format the result as a numbered list.

        If no decisions are found, write:
        "No key decisions found."
        """
    )

    return run_chain(
        chain,
        transcript,
        "AI decision extraction is temporarily unavailable because the Mistral API rate limit was reached."
    )


# ============================================================
# OPEN QUESTIONS
# ============================================================

def extract_questions(transcript: str) -> str:

    chain = build_chain(
        """
        You are an expert meeting analyst.

        From the meeting transcript, extract all unresolved questions
        or topics that need follow-up.

        Format the result as a numbered list.

        If no open questions are found, write:
        "No open questions found."
        """
    )

    return run_chain(
        chain,
        transcript,
        "AI question extraction is temporarily unavailable because the Mistral API rate limit was reached."
    )