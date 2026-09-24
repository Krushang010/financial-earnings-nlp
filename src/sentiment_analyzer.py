from functools import lru_cache

import pandas as pd
import spacy
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


FINBERT_MODEL = "ProsusAI/finbert"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
def load_finbert():
    """
    Load FinBERT tokenizer and model once.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        FINBERT_MODEL
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL
    )

    model.eval()

    return tokenizer, model


def extract_sentences(
    text: str,
    min_words: int = 8
) -> list[str]:
    """
    Split transcript into reasonably substantive sentences.

    Very short statements are excluded because earnings-call
    transcripts contain many fragments such as:
    'Thank you.'
    'Yes, sir.'
    'Good morning.'
    """

    nlp = load_spacy()

    doc = nlp(text)

    sentences = [
        sentence.text.strip()
        for sentence in doc.sents
        if len(sentence.text.split()) >= min_words
    ]

    return sentences


def predict_sentiment_batch(
    sentences: list[str],
    batch_size: int = 16,
    max_length: int = 256
) -> pd.DataFrame:
    """
    Run sentence-level FinBERT inference.
    """

    tokenizer, model = load_finbert()

    model = model.to(DEVICE)

    rows = []

    for start in range(
        0,
        len(sentences),
        batch_size
    ):

        batch = sentences[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

        probabilities = (
            probabilities
            .detach()
            .cpu()
        )

        for sentence, probs in zip(
            batch,
            probabilities
        ):

            label_scores = {}

            for index, probability in enumerate(probs):

                label = (
                    model.config.id2label[index]
                    .lower()
                )

                label_scores[label] = (
                    probability.item()
                )

            predicted_label = max(
                label_scores,
                key=label_scores.get
            )

            rows.append(
                {
                    "sentence": sentence,

                    "sentiment":
                        predicted_label,

                    "positive_probability":
                        label_scores.get(
                            "positive",
                            0.0
                        ),

                    "negative_probability":
                        label_scores.get(
                            "negative",
                            0.0
                        ),

                    "neutral_probability":
                        label_scores.get(
                            "neutral",
                            0.0
                        )
                }
            )

    # Important for your 4 GB RTX 3050:
    # return FinBERT to CPU before the summarizer
    # eventually uses the GPU.

    model.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def get_overall_sentiment(
    sentiment_df: pd.DataFrame
) -> dict:
    """
    Aggregate sentence-level FinBERT probabilities into
    one model-derived transcript sentiment.
    """

    avg_positive = sentiment_df[
        "positive_probability"
    ].mean()

    avg_negative = sentiment_df[
        "negative_probability"
    ].mean()

    avg_neutral = sentiment_df[
        "neutral_probability"
    ].mean()

    scores = {
        "positive": avg_positive,
        "negative": avg_negative,
        "neutral": avg_neutral
    }

    overall_label = max(
        scores,
        key=scores.get
    )

    return {
        "label": overall_label,

        "positive": round(
            float(avg_positive) * 100,
            2
        ),

        "neutral": round(
            float(avg_neutral) * 100,
            2
        ),

        "negative": round(
            float(avg_negative) * 100,
            2
        )
    }


def get_sentiment_distribution(
    sentiment_df: pd.DataFrame
) -> dict:
    """
    Percentage of analyzed sentences assigned to each
    FinBERT class.
    """

    distribution = (
        sentiment_df["sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
    )

    return {
        "positive": float(
            distribution.get("positive", 0)
        ),

        "neutral": float(
            distribution.get("neutral", 0)
        ),

        "negative": float(
            distribution.get("negative", 0)
        )
    }


def analyze_sentiment(
    text: str
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Complete sentiment pipeline.

    Returns:
        sentence-level dataframe
        overall transcript sentiment
        hard-label sentence distribution
    """

    sentences = extract_sentences(text)

    sentiment_df = predict_sentiment_batch(
        sentences
    )

    overall_sentiment = (
        get_overall_sentiment(
            sentiment_df
        )
    )

    distribution = (
        get_sentiment_distribution(
            sentiment_df
        )
    )

    return (
        sentiment_df,
        overall_sentiment,
        distribution
    )