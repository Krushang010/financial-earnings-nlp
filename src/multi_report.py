from datetime import datetime
import re


# =========================================================
# DATE / ORDER HELPERS
# =========================================================

MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}


def extract_date_from_filename(
    file_name: str
) -> datetime:
    """
    Convert filenames such as:

        july_25.pdf
        nov_25.pdf
        feb_26.pdf
        may_26.pdf
        august_26.pdf

    into sortable dates.
    """

    name = (
        file_name
        .lower()
        .replace(".pdf", "")
        .strip()
    )

    parts = name.split("_")

    if len(parts) < 2:
        return datetime.max

    month_text = parts[0][:3]

    month = MONTH_MAP.get(
        month_text
    )

    if month is None:
        return datetime.max

    try:
        year = int(
            parts[-1]
        )

        if year < 100:
            year += 2000

        return datetime(
            year,
            month,
            1
        )

    except ValueError:
        return datetime.max


def sort_reports_chronologically(
    reports: list[dict]
) -> list[dict]:
    """
    Sort reports chronologically using filenames.

    Upload order therefore does not affect trend analysis.
    """

    return sorted(
        reports,
        key=lambda report:
            extract_date_from_filename(
                report.get(
                    "file_name",
                    ""
                )
            )
    )


# =========================================================
# FINANCIAL NUMBER PARSING
# =========================================================

def parse_financial_number(
    value
):
    """
    Convert values such as:

        Rs. 960-odd crores
        Rs. 1,604 crores
        INR159 crores
        Rs. 67.1 crores

    into numeric values for comparison.

    Returns None when the value cannot be parsed.
    """

    if value is None:
        return None

    cleaned = (
        str(value)
        .replace(",", "")
        .replace("-odd", "")
        .replace(" odd", "")
        .strip()
    )

    match = re.search(
        r"(\d+(?:\.\d+)?)",
        cleaned
    )

    if not match:
        return None

    return float(
        match.group(1)
    )


# =========================================================
# COMBINED SENTIMENT
# =========================================================

def calculate_combined_sentiment(
    reports: list[dict]
) -> dict:
    """
    Calculate sentence-weighted sentiment across
    all analyzed transcripts.
    """

    total_sentences = sum(
        report.get(
            "analyzed_sentences",
            0
        )
        for report in reports
    )

    if total_sentences == 0:

        return {
            "label": "unknown",
            "positive": 0.0,
            "neutral": 0.0,
            "negative": 0.0
        }


    positive = 0.0
    neutral = 0.0
    negative = 0.0


    for report in reports:

        sentence_count = report.get(
            "analyzed_sentences",
            0
        )

        if sentence_count == 0:
            continue

        weight = (
            sentence_count
            / total_sentences
        )


        sentiment = report.get(
            "overall_sentiment",
            {}
        )


        positive += (
            sentiment.get(
                "positive",
                0
            )
            * weight
        )

        neutral += (
            sentiment.get(
                "neutral",
                0
            )
            * weight
        )

        negative += (
            sentiment.get(
                "negative",
                0
            )
            * weight
        )


    scores = {
        "positive": positive,
        "neutral": neutral,
        "negative": negative
    }


    overall_label = max(
        scores,
        key=scores.get
    )


    return {
        "label":
            overall_label,

        "positive":
            round(
                positive,
                2
            ),

        "neutral":
            round(
                neutral,
                2
            ),

        "negative":
            round(
                negative,
                2
            )
    }


# =========================================================
# METRIC COMPARISON TABLE
# =========================================================

def build_metric_comparison(
    reports: list[dict]
) -> list[dict]:
    """
    Build one row per transcript for the Streamlit
    cross-quarter comparison table.
    """

    rows = []

    for report in reports:

        metrics = report.get(
            "financial_metrics",
            {}
        )

        row = {
            "Transcript":
                report.get(
                    "file_name",
                    "Unknown"
                )
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return rows


# =========================================================
# INDIVIDUAL METRIC TREND
# =========================================================

def describe_metric_trend(
    reports: list[dict],
    metric: str
):
    """
    Compare the earliest and latest AVAILABLE values
    for a metric.

    Example:
        Revenue:
        feb_26 -> may_26 -> august_26

    Missing observations are ignored.
    """

    observations = []

    for report in reports:

        raw_value = (
            report
            .get(
                "financial_metrics",
                {}
            )
            .get(metric)
        )

        numeric_value = (
            parse_financial_number(
                raw_value
            )
        )

        if numeric_value is None:
            continue

        observations.append(
            {
                "file":
                    report.get(
                        "file_name",
                        "Unknown"
                    ),

                "value":
                    numeric_value,

                "raw":
                    raw_value
            }
        )


    if len(observations) < 2:
        return None


    first = observations[0]
    last = observations[-1]


    difference = (
        last["value"]
        - first["value"]
    )


    if difference > 0:

        direction = "increased"

    elif difference < 0:

        direction = "decreased"

    else:

        direction = (
            "remained unchanged"
        )


    change_pct = None

    if first["value"] != 0:

        change_pct = (
            difference
            / first["value"]
            * 100
        )


    return {
        "metric":
            metric,

        "first_file":
            first["file"],

        "first_value":
            first["raw"],

        "last_file":
            last["file"],

        "last_value":
            last["raw"],

        "direction":
            direction,

        "change_pct":
            (
                round(
                    change_pct,
                    1
                )
                if change_pct is not None
                else None
            )
    }


# =========================================================
# COMPARATIVE SUMMARY
# =========================================================

def build_comparative_summary(
    reports: list[dict]
) -> str:
    """
    Build a deterministic multi-quarter summary.

    Financial numbers come only from already-extracted
    structured metrics.

    Individual DistilBART summaries are appended separately
    rather than being re-summarized again.
    """

    summary_parts = []


    summary_parts.append(
        f"{len(reports)} earnings-call transcripts "
        f"were analyzed chronologically."
    )


    # -----------------------------------------------------
    # FINANCIAL TREND SUMMARY
    # -----------------------------------------------------

    financial_metrics = [
        "Revenue",
        "EBITDA",
        "PBT",
        "PAT"
    ]


    for metric in financial_metrics:

        trend = describe_metric_trend(
            reports,
            metric
        )

        if trend is None:
            continue


        sentence = (
            f"{metric} {trend['direction']} "
            f"from {trend['first_value']} "
            f"in {trend['first_file']} "
            f"to {trend['last_value']} "
            f"in {trend['last_file']}"
        )


        if (
            trend["change_pct"] is not None
            and
            trend["direction"]
            != "remained unchanged"
        ):

            sentence += (
                f", an approximate "
                f"{abs(trend['change_pct']):.1f}% "
                f"change across the available observations"
            )


        sentence += "."

        summary_parts.append(
            sentence
        )


    # -----------------------------------------------------
    # SENTIMENT BY TRANSCRIPT
    # -----------------------------------------------------

    sentiment_sequence = []

    for report in reports:

        sentiment = report.get(
            "overall_sentiment",
            {}
        )

        sentiment_sequence.append(
            (
                report.get(
                    "file_name",
                    "Unknown"
                ),

                sentiment.get(
                    "label",
                    "unknown"
                )
            )
        )


    sentiment_text = ", ".join(
        f"{file_name}: {label.upper()}"
        for file_name, label
        in sentiment_sequence
    )


    summary_parts.append(
        "Model-derived transcript sentiment across "
        f"the analyzed calls was: {sentiment_text}."
    )


    # -----------------------------------------------------
    # INDIVIDUAL CALL HIGHLIGHTS
    # -----------------------------------------------------

    summary_parts.append(
        "\nIndividual call highlights:"
    )


    for report in reports:

        file_name = report.get(
            "file_name",
            "Unknown"
        )

        executive_summary = (
            report.get(
                "executive_summary",
                ""
            )
        )

        if not executive_summary:
            continue

        summary_parts.append(
            f"\n{file_name}: "
            f"{executive_summary}"
        )


    return "\n".join(
        summary_parts
    )


# =========================================================
# MASTER MULTI-REPORT FUNCTION
# =========================================================

def generate_multi_report_summary(
    reports: list[dict]
) -> dict:
    """
    Master function used by Streamlit for multi-PDF analysis.
    """

    if len(reports) < 2:

        raise ValueError(
            "At least two earnings-call reports "
            "are required for multi-document analysis."
        )


    # CRITICAL:
    # Never use upload order for financial trend analysis.
    reports = (
        sort_reports_chronologically(
            reports
        )
    )


    combined_sentiment = (
        calculate_combined_sentiment(
            reports
        )
    )


    combined_summary = (
        build_comparative_summary(
            reports
        )
    )


    comparison_rows = (
        build_metric_comparison(
            reports
        )
    )


    return {
        "summary":
            combined_summary,

        "overall_sentiment":
            combined_sentiment,

        "comparison":
            comparison_rows,

        "documents_analyzed":
            len(reports),

        "ordered_files": [
            report.get(
                "file_name",
                "Unknown"
            )
            for report in reports
        ]
    }