from pathlib import Path

from .pdf_processor import (
    extract_transcript_from_pdf
)

from .financial_extractor import (
    extract_financial_metrics
)

from .sentiment_analyzer import (
    analyze_sentiment
)

from .summarizer import (
    generate_executive_summary
)


def analyze_earnings_call(
    pdf_path
) -> tuple[dict, object]:
    """
    Run the complete Financial Earnings Call NLP pipeline.

    Steps:
        1. Extract and clean PDF transcript
        2. Extract explicit financial metrics
        3. Run FinBERT sentiment analysis
        4. Aggregate transcript-level sentiment
        5. Generate hierarchical DistilBART summary

    Returns:
        report:
            structured results for the frontend

        sentiment_df:
            sentence-level FinBERT predictions
    """

    pdf_path = Path(pdf_path)

    # --------------------------------------------------
    # 1. TRANSCRIPT
    # --------------------------------------------------

    transcript = extract_transcript_from_pdf(
        pdf_path
    )

    if not transcript.strip():
        raise ValueError(
            "No transcript text could be extracted from the PDF."
        )


    # --------------------------------------------------
    # 2. FINANCIAL METRICS
    # --------------------------------------------------

    financial_metrics = (
        extract_financial_metrics(
            transcript
        )
    )


    # --------------------------------------------------
    # 3. SENTIMENT
    # --------------------------------------------------

    (
        sentiment_df,
        overall_sentiment,
        sentiment_distribution
    ) = analyze_sentiment(
        transcript
    )


    # --------------------------------------------------
    # 4. EXECUTIVE SUMMARY
    # --------------------------------------------------

    executive_summary = (
        generate_executive_summary(
            transcript
        )
    )


    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    report = {

        "file_name":
            pdf_path.name,

        "financial_metrics":
            financial_metrics,

        "overall_sentiment":
            overall_sentiment,

        "sentiment_distribution":
            sentiment_distribution,

        "executive_summary":
            executive_summary,

        "transcript":
            transcript,

        "analyzed_sentences":
            len(sentiment_df)
    }

    return report, sentiment_df