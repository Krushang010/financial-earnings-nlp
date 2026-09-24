from functools import lru_cache

import spacy
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# MODEL LOADERS
# =========================================================

@lru_cache(maxsize=1)
def load_spacy():
    """
    Lightweight spaCy pipeline used only for
    sentence segmentation.
    """

    nlp = spacy.blank("en")

    nlp.add_pipe(
        "sentencizer"
    )

    nlp.max_length = 3_000_000

    return nlp


@lru_cache(maxsize=1)
def load_summarizer():
    """
    Load DistilBART tokenizer and summarization model once.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARY_MODEL
    )

    # IMPORTANT:
    # Generation settings belong in generation_config,
    # NOT model.config.
    model.generation_config.forced_bos_token_id = 0

    model.eval()

    return tokenizer, model


# =========================================================
# SENTENCE SEGMENTATION
# =========================================================

def extract_sentences(text: str) -> list[str]:

    nlp = load_spacy()

    doc = nlp(text)

    return [
        sentence.text.strip()
        for sentence in doc.sents
        if sentence.text.strip()
    ]


# =========================================================
# TOKEN COUNTING
# =========================================================

def count_tokens(
    text: str,
    tokenizer
) -> int:
    """
    Safe token counting without tokenizer.encode()
    maximum-length warnings.
    """

    return len(
        tokenizer.tokenize(text)
    )


# =========================================================
# TOKEN-AWARE CHUNKING
# =========================================================

def chunk_text_by_tokens(
    texts: list[str],
    tokenizer,
    max_tokens: int = 700
) -> list[str]:

    chunks = []

    current_chunk = []
    current_tokens = 0

    for text in texts:

        token_count = count_tokens(
            text,
            tokenizer
        )

        if (
            current_chunk
            and
            current_tokens + token_count > max_tokens
        ):

            chunks.append(
                " ".join(current_chunk)
            )

            current_chunk = []
            current_tokens = 0

        current_chunk.append(text)
        current_tokens += token_count

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


# =========================================================
# SINGLE CHUNK SUMMARIZATION
# =========================================================

def summarize_chunk(
    text: str,
    tokenizer,
    model,
    max_input_tokens: int = 800,
    max_summary_tokens: int = 150,
    min_summary_tokens: int = 40
) -> str:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        summary_ids = model.generate(
            **inputs,
            max_length=max_summary_tokens,
            min_length=min_summary_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary.strip()


# =========================================================
# HIERARCHICAL SUMMARIZATION
# =========================================================

def generate_executive_summary(
    text: str
) -> str:

    tokenizer, model = load_summarizer()

    model = model.to(DEVICE)

    try:

        sentences = extract_sentences(
            text
        )

        if not sentences:
            return ""


        # -------------------------------------------------
        # LEVEL 1
        # -------------------------------------------------

        level1_chunks = chunk_text_by_tokens(
            sentences,
            tokenizer,
            max_tokens=700
        )

        summaries = []

        for chunk in level1_chunks:

            summary = summarize_chunk(
                text=chunk,
                tokenizer=tokenizer,
                model=model,
                max_input_tokens=800,
                max_summary_tokens=130,
                min_summary_tokens=40
            )

            summaries.append(summary)


        # -------------------------------------------------
        # RECURSIVE REDUCTION
        # -------------------------------------------------

        while True:

            combined = " ".join(
                summaries
            )

            combined_token_count = (
                count_tokens(
                    combined,
                    tokenizer
                )
            )

            if combined_token_count <= 700:
                break

            grouped_summaries = (
                chunk_text_by_tokens(
                    summaries,
                    tokenizer,
                    max_tokens=700
                )
            )

            next_level_summaries = []

            for group in grouped_summaries:

                reduced_summary = (
                    summarize_chunk(
                        text=group,
                        tokenizer=tokenizer,
                        model=model,
                        max_input_tokens=800,
                        max_summary_tokens=180,
                        min_summary_tokens=60
                    )
                )

                next_level_summaries.append(
                    reduced_summary
                )

            summaries = (
                next_level_summaries
            )


        # -------------------------------------------------
        # FINAL SUMMARY
        # -------------------------------------------------

        final_input = " ".join(
            summaries
        )

        final_summary = summarize_chunk(
            text=final_input,
            tokenizer=tokenizer,
            model=model,
            max_input_tokens=800,
            max_summary_tokens=220,
            min_summary_tokens=90
        )

        return final_summary


    finally:

        model.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()