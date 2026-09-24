import os
import tempfile

import pandas as pd
import streamlit as st

from src.pipeline import analyze_earnings_call

from src.multi_report import (
    generate_multi_report_summary
)
# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Financial Earnings Call Intelligence",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("📊 Financial Earnings Call Intelligence")

st.write(
    """
    Upload one or more earnings-call transcript PDFs to generate
    executive summaries, analyze financial sentiment using FinBERT,
    and extract key reported financial metrics.
    """
)

st.caption(
    "Financial NLP • FinBERT • DistilBART • PyMuPDF"
)

st.divider()


# =========================================================
# MULTIPLE FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Upload Earnings Call PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF(s) uploaded successfully."
    )

    for uploaded_file in uploaded_files:
        st.write(f"• {uploaded_file.name}")


    analyze_button = st.button(
        "Analyze Earnings Calls",
        type="primary",
        use_container_width=True
    )


    if analyze_button:

        results = []

        progress_bar = st.progress(0)

        status_text = st.empty()

        total_files = len(uploaded_files)


        # =================================================
        # ANALYZE EACH PDF
        # =================================================

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1
        ):

            temp_path = None

            try:

                status_text.write(
                    f"Analyzing {index}/{total_files}: "
                    f"{uploaded_file.name}"
                )


                # -----------------------------------------
                # TEMPORARY PDF
                # -----------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name


                # -----------------------------------------
                # RUN COMPLETE BACKEND PIPELINE
                # -----------------------------------------

                report, sentiment_df = (
                    analyze_earnings_call(
                        temp_path
                    )
                )


                # Replace temporary filename with
                # original uploaded filename
                report["file_name"] = (
                    uploaded_file.name
                )


                results.append(
                    {
                        "report": report,
                        "sentiment_df": sentiment_df
                    }
                )


            except Exception as error:

                st.error(
                    f"Analysis failed for "
                    f"{uploaded_file.name}: {error}"
                )


            finally:

                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):
                    os.remove(temp_path)


            progress_bar.progress(
                index / total_files
            )


        status_text.empty()
        progress_bar.empty()


        # =================================================
        # RESULTS
        # =================================================

        if results:

            st.success(
                f"Analysis completed for "
                f"{len(results)} PDF(s)."
            )

            st.divider()

            # =================================================
            # MULTI-DOCUMENT ANALYSIS
            # =================================================

            if len(results) >= 2:

                st.header(
                    "📈 Multi-Quarter Earnings Summary"
                )

                st.caption(
                    f"Combined analysis across "
                    f"{len(results)} earnings-call transcripts."
                )

                reports_for_comparison = [
                    result["report"]
                    for result in results
                ]

                with st.spinner(
                        "Generating cross-quarter summary..."
                ):

                    multi_report = (
                        generate_multi_report_summary(
                            reports_for_comparison
                        )
                    )

                # ---------------------------------------------
                # COMBINED SENTIMENT
                # ---------------------------------------------

                combined_sentiment = (
                    multi_report[
                        "overall_sentiment"
                    ]
                )

                st.subheader(
                    "Combined Transcript Sentiment"
                )

                st.subheader(
                    combined_sentiment[
                        "label"
                    ].upper()
                )

                multi_col1, multi_col2, multi_col3 = (
                    st.columns(3)
                )

                multi_col1.metric(
                    "Positive",
                    f"{combined_sentiment['positive']:.2f}%"
                )

                multi_col2.metric(
                    "Neutral",
                    f"{combined_sentiment['neutral']:.2f}%"
                )

                multi_col3.metric(
                    "Negative",
                    f"{combined_sentiment['negative']:.2f}%"
                )

                # ---------------------------------------------
                # MULTI-QUARTER SUMMARY
                # ---------------------------------------------

                st.subheader(
                    "Combined Executive Summary"
                )

                st.write(
                    multi_report["summary"]
                )

                # ---------------------------------------------
                # COMPARATIVE METRICS
                # ---------------------------------------------

                st.subheader(
                    "Financial Metrics Across Calls"
                )

                comparison_rows = []

                for result in results:
                    report = result[
                        "report"
                    ]

                    row = {
                        "Transcript":
                            report[
                                "file_name"
                            ]
                    }

                    row.update(
                        report[
                            "financial_metrics"
                        ]
                    )

                    comparison_rows.append(
                        row
                    )

                comparison_df = pd.DataFrame(
                    comparison_rows
                )

                comparison_df = (
                    comparison_df.fillna(
                        "Not reported"
                    )
                )

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    """
                    Combined summaries are generated from each
                    transcript's extracted metrics, sentiment,
                    and individual executive summary rather
                    than from concatenated raw transcripts.
                    """
                )

                st.divider()


            # =================================================
            # CREATE ONE TAB PER TRANSCRIPT
            # =================================================

            tab_names = [
                result["report"]["file_name"]
                for result in results
            ]

            tabs = st.tabs(tab_names)


            for tab, result in zip(
                tabs,
                results
            ):

                report = result["report"]
                sentiment_df = result[
                    "sentiment_df"
                ]


                with tab:

                    st.subheader(
                        report["file_name"]
                    )


                    # =====================================
                    # OVERALL SENTIMENT
                    # =====================================

                    st.header(
                        "Overall Transcript Sentiment"
                    )

                    sentiment = report[
                        "overall_sentiment"
                    ]

                    st.subheader(
                        sentiment[
                            "label"
                        ].upper()
                    )


                    col1, col2, col3 = (
                        st.columns(3)
                    )


                    col1.metric(
                        "Positive",
                        f"{sentiment['positive']:.2f}%"
                    )

                    col2.metric(
                        "Neutral",
                        f"{sentiment['neutral']:.2f}%"
                    )

                    col3.metric(
                        "Negative",
                        f"{sentiment['negative']:.2f}%"
                    )


                    # =====================================
                    # EXECUTIVE SUMMARY
                    # =====================================

                    st.divider()

                    st.header(
                        "Executive Summary"
                    )

                    st.write(
                        report[
                            "executive_summary"
                        ]
                    )

                    st.caption(
                        "Generated using hierarchical "
                        "DistilBART summarization."
                    )


                    # =====================================
                    # FINANCIAL METRICS
                    # =====================================

                    st.divider()

                    st.header(
                        "Key Financial Metrics"
                    )


                    metrics = report[
                        "financial_metrics"
                    ]


                    metric_df = pd.DataFrame(
                        [
                            {
                                "Metric": metric,

                                "Value": (
                                    value
                                    if value is not None
                                    else
                                    "Not explicitly reported"
                                )
                            }

                            for metric, value
                            in metrics.items()
                        ]
                    )


                    st.dataframe(
                        metric_df,
                        use_container_width=True,
                        hide_index=True
                    )


                    st.caption(
                        "Metrics are extracted using "
                        "deterministic rules. Missing "
                        "values are not inferred."
                    )


                    # =====================================
                    # SENTENCE DISTRIBUTION
                    # =====================================

                    st.divider()

                    st.header(
                        "Sentence-Level "
                        "Sentiment Distribution"
                    )


                    distribution = report[
                        "sentiment_distribution"
                    ]


                    chart_df = pd.DataFrame(
                        {
                            "Sentiment": [
                                "Positive",
                                "Neutral",
                                "Negative"
                            ],

                            "Percentage": [
                                distribution[
                                    "positive"
                                ],

                                distribution[
                                    "neutral"
                                ],

                                distribution[
                                    "negative"
                                ]
                            ]
                        }
                    )


                    st.bar_chart(
                        chart_df,
                        x="Sentiment",
                        y="Percentage"
                    )


                    st.caption(
                        f"{report['analyzed_sentences']} "
                        "substantive transcript "
                        "sentences analyzed."
                    )


                    # =====================================
                    # SENTENCE-LEVEL DETAILS
                    # =====================================

                    with st.expander(
                        "Explore Sentence-Level "
                        "FinBERT Results"
                    ):

                        display_df = (
                            sentiment_df.copy()
                        )


                        for column in [
                            "positive_probability",
                            "neutral_probability",
                            "negative_probability"
                        ]:

                            display_df[column] = (
                                display_df[column]
                                * 100
                            ).round(2)


                        display_df = (
                            display_df.rename(
                                columns={
                                    "sentence":
                                        "Sentence",

                                    "sentiment":
                                        "Sentiment",

                                    "positive_probability":
                                        "Positive %",

                                    "neutral_probability":
                                        "Neutral %",

                                    "negative_probability":
                                        "Negative %"
                                }
                            )
                        )


                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )


                    # =====================================
                    # TRANSCRIPT
                    # =====================================

                    with st.expander(
                        "View Extracted Transcript"
                    ):

                        st.text_area(
                            "Transcript",
                            report[
                                "transcript"
                            ],
                            height=500,
                            key=(
                                "transcript_"
                                + report[
                                    "file_name"
                                ]
                            )
                        )


            st.divider()

            st.caption(
                """
                Sentiment and executive summary outputs are
                model-generated NLP results and should not
                be interpreted as investment recommendations.
                Financial metrics are extracted only when
                explicitly identified in the transcript.
                """
            )