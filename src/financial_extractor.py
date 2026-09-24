import re


# =========================================================
# HELPERS
# =========================================================

def find_first_match(
    text: str,
    patterns: list[str]
):
    """
    Try patterns in priority order.

    Returns the first reliable match.
    """

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

    return None


# =========================================================
# FINANCIAL EXTRACTION
# =========================================================

def extract_financial_metrics(
    text: str
) -> dict:
    """
    Extract explicitly reported CURRENT-QUARTER financial
    metrics from an earnings-call transcript.

    Priority is given to phrases such as:

        operating EBITDA for the quarter...
        revenue for the quarter...
        PAT for the quarter...

    rather than annual totals, historical values,
    or management guidance.
    """

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    # -----------------------------------------------------
    # MONEY
    # -----------------------------------------------------

    MONEY = (
        r"(?:Rs\.?|INR|₹)\s*"
        r"\d[\d,]*(?:\.\d+)?"
        r"(?:-odd|\s+odd|-plus|\s+plus|\+)?\s*"
        r"(?:crores?|crore|million|billion)"
    )


    PERCENT = (
        r"\d+(?:\.\d+)?\s*%"
    )


    # =====================================================
    # REVENUE
    # =====================================================

    revenue_patterns = [

        # Example:
        # consolidated total revenue for the quarter
        # stood at Rs. 1,604 crores

        rf"(?:consolidated\s+)?"
        rf"(?:total\s+)?revenue"
        rf"[^.!?]{{0,50}}?"
        rf"for\s+the\s+quarter"
        rf"[^.!?]{{0,50}}?"
        rf"(?:stood\s+at|came\s+in|was|is|of)?"
        rf"[^.!?]{{0,15}}?"
        rf"({MONEY})",


        # Example:
        # consolidated revenue ... Rs. 960-odd crores

        rf"consolidated\s+(?:total\s+)?revenue"
        rf"[^.!?]{{0,100}}?"
        rf"({MONEY})",


        rf"total\s+revenue"
        rf"[^.!?]{{0,100}}?"
        rf"({MONEY})"
    ]


    # =====================================================
    # EXPORT REVENUE
    # =====================================================

    export_revenue_patterns = [

        rf"export\s+revenue"
        rf"[^.!?]{{0,100}}?"
        rf"({MONEY})"
    ]


    # =====================================================
    # GROSS MARGIN
    # =====================================================

    gross_margin_patterns = [

        rf"gross\s+margin"
        rf"[^.!?]{{0,60}}?"
        rf"({PERCENT})"
    ]


    # =====================================================
    # OPERATING EBITDA
    # =====================================================

    ebitda_patterns = [

        # Highest priority:
        #
        # "operating EBITDA for the quarter came in
        # INR174 crores"
        #
        # "operating EBITDA for the quarter stood at
        # healthy Rs. 96 odd crores"

        rf"\boperating\s+EBITDA\b"
        rf"(?!\s+margin)"
        rf"[^.!?]{{0,60}}?"
        rf"\bfor\s+the\s+quarter\b"
        rf"[^.!?]{{0,40}}?"
        rf"(?:came\s+in|stood\s+at|was|is)?"
        rf"[^.!?]{{0,15}}?"
        rf"({MONEY})",


        # Example:
        #
        # "operating EBITDA increased by 69%
        # year-on-year to Rs. 162 crores"

        rf"\boperating\s+EBITDA\b"
        rf"(?!\s+margin)"
        rf"[^.!?]{{0,100}}?"
        rf"\bto\s+"
        rf"({MONEY})",


        # Conservative fallback:
        # still requires OPERATING EBITDA

        rf"\boperating\s+EBITDA\b"
        rf"(?!\s+margin)"
        rf"[^.!?]{{0,80}}?"
        rf"({MONEY})"
    ]


    # =====================================================
    # EBITDA MARGIN
    # =====================================================

    ebitda_margin_patterns = [

        # "Operating EBITDA margin was 11.9%"

        rf"\boperating\s+EBITDA\s+margin\b"
        rf"[^.!?]{{0,30}}?"
        rf"(?:was|is|stood\s+at|of|at)?"
        rf"[^.!?]{{0,10}}?"
        rf"({PERCENT})",


        # "11.9% EBITDA margin"

        rf"({PERCENT})"
        rf"[^.!?]{{0,15}}?"
        rf"(?:operating\s+)?EBITDA\s+margin"
    ]


    # =====================================================
    # PBT
    # =====================================================

    pbt_patterns = [

        # Explicit quarter phrasing

        rf"\bPBT\b"
        rf"(?!\s+margin)"
        rf"[^.!?]{{0,70}}?"
        rf"(?:for\s+the\s+quarter)"
        rf"[^.!?]{{0,40}}?"
        rf"(?:stood\s+at|came\s+in|was|is|of)?"
        rf"[^.!?]{{0,15}}?"
        rf"({MONEY})",


        # "PBT increased ... reaching to Rs. 141 crores"

        rf"\bPBT\b"
        rf"(?!\s+margin)"
        rf"[^.!?]{{0,100}}?"
        rf"(?:reaching\s+to|reached|to)"
        rf"\s*"
        rf"({MONEY})",


        # Local fallback

        rf"\bPBT\b"
        rf"(?!\s+margin)"
        rf"[^.!?]{{0,80}}?"
        rf"({MONEY})"
    ]


    # =====================================================
    # PBT MARGIN
    # =====================================================

    pbt_margin_patterns = [

        rf"\bPBT\s+margin\b"
        rf"[^.!?]{{0,40}}?"
        rf"({PERCENT})"
    ]


    # =====================================================
    # PAT
    # =====================================================

    pat_patterns = [

        # Example:
        # PAT for the quarter is Rs. 106 crores

        rf"\bPAT\b"
        rf"(?!\s+margin)"
        rf"[^.!?]{{0,40}}?"
        rf"\bfor\s+the\s+quarter\b"
        rf"[^.!?]{{0,30}}?"
        rf"(?:stood\s+at|came\s+in|was|is|of)?"
        rf"[^.!?]{{0,10}}?"
        rf"({MONEY})",


        # Local fallback

        rf"\bPAT\b"
        rf"(?!\s+margin)"
        rf"[^.!?]{{0,80}}?"
        rf"({MONEY})"
    ]


    # =====================================================
    # PAT MARGIN
    # =====================================================

    pat_margin_patterns = [

        rf"\bPAT\s+margin\b"
        rf"[^.!?]{{0,40}}?"
        rf"({PERCENT})",


        rf"({PERCENT})"
        rf"[^.!?]{{0,15}}?"
        rf"PAT\s+margin"
    ]


    # =====================================================
    # RESULTS
    # =====================================================

    return {

        "Revenue":
            find_first_match(
                text,
                revenue_patterns
            ),

        "Export Revenue":
            find_first_match(
                text,
                export_revenue_patterns
            ),

        "Gross Margin":
            find_first_match(
                text,
                gross_margin_patterns
            ),

        "EBITDA":
            find_first_match(
                text,
                ebitda_patterns
            ),

        "EBITDA Margin":
            find_first_match(
                text,
                ebitda_margin_patterns
            ),

        "PBT":
            find_first_match(
                text,
                pbt_patterns
            ),

        "PBT Margin":
            find_first_match(
                text,
                pbt_margin_patterns
            ),

        "PAT":
            find_first_match(
                text,
                pat_patterns
            ),

        "PAT Margin":
            find_first_match(
                text,
                pat_margin_patterns
            )
    }