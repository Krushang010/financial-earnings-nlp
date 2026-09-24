import os
import re
from functools import lru_cache

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer


MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

MAX_CHUNK_TOKENS = 700


@lru_cache(maxsize=1)
def load_tokenizer():
    """
    Load only the tokenizer locally.

    The DistilBART model itself is NOT loaded into Streamlit RAM.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@lru_cache(maxsize=1)
def get_inference_client():
    """
    Create Hugging Face Inference client.

    HF_TOKEN is stored in Streamlit Secrets.
    Root-level Streamlit secrets are exposed as environment variables.
    """

    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it to Streamlit Secrets."
        )

    return InferenceClient(
        provider="hf-inference",
        api_key=token,
    )


def token_count(text: str) -> int:
    tokenizer = load_tokenizer()

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitting for summarization chunking.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    return re.split(
        r"(?<=[.!?])\s+",
        text,
    )


def create_chunks(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:

    tokenizer = load_tokenizer()

    sentences = split_sentences(text)

    chunks = []

    current_sentences = []
    current_tokens = 0

    for sentence in sentences:

        sentence_tokens = len(
            tokenizer.encode(
                sentence,
                add_special_tokens=False,
            )
        )

        # Handle unusually long individual sentences
        if sentence_tokens > max_tokens:

            if current_sentences:
                chunks.append(
                    " ".join(current_sentences)
                )

                current_sentences = []
                current_tokens = 0

            encoded = tokenizer.encode(
                sentence,
                add_special_tokens=False,
            )

            for start in range(
                0,
                len(encoded),
                max_tokens,
            ):

                part = encoded[
                    start:start + max_tokens
                ]

                chunks.append(
                    tokenizer.decode(
                        part,
                        skip_special_tokens=True,
                    )
                )

            continue

        if (
            current_tokens + sentence_tokens
            > max_tokens
        ):

            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = [sentence]
            current_tokens = sentence_tokens

        else:

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


def summarize_chunk(text: str) -> str:
    """
    Summarize one chunk remotely using Hugging Face Inference.
    """

    client = get_inference_client()

    result = client.summarization(
        text,
        model=MODEL_NAME,
    )

    # huggingface_hub may return an object
    if hasattr(result, "summary_text"):
        return result.summary_text.strip()

    return str(result).strip()


def generate_executive_summary(
    transcript: str,
) -> str:
    """
    Hierarchical summarization.

    Transcript
        ↓
    chunks
        ↓
    remote DistilBART summaries
        ↓
    recursive reduction
        ↓
    final executive summary
    """

    if not transcript.strip():
        return ""

    chunks = create_chunks(transcript)

    if not chunks:
        return ""

    summaries = []

    for chunk in chunks:
        summary = summarize_chunk(chunk)

        if summary:
            summaries.append(summary)

    if not summaries:
        return ""

    combined = " ".join(summaries)

    # Recursive reduction if summaries are still too long
    max_rounds = 4

    rounds = 0

    while (
        token_count(combined) > MAX_CHUNK_TOKENS
        and rounds < max_rounds
    ):

        chunks = create_chunks(combined)

        reduced = []

        for chunk in chunks:
            summary = summarize_chunk(chunk)

            if summary:
                reduced.append(summary)

        if not reduced:
            break

        new_combined = " ".join(reduced)

        # Safety: avoid infinite recursion
        if len(new_combined) >= len(combined):
            break

        combined = new_combined

        rounds += 1

    # One final compression if needed
    if token_count(combined) <= MAX_CHUNK_TOKENS:

        try:
            final_summary = summarize_chunk(combined)

            if final_summary:
                return final_summary

        except Exception:
            pass

    return combined