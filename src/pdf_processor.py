import re
import pymupdf

PAGE_PATTERN = re.compile(
    r"^Page\s+\d+\s+of\s+\d+$",
    flags=re.IGNORECASE
)

DATE_PATTERN = re.compile(
    r"^(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}$",
    flags=re.IGNORECASE
)


def clean_transcript_page(text: str) -> str:
    """
    Remove common PDF transcript artifacts while preserving
    the actual financial/conversational content.
    """

    # Fix encoded newline artifacts sometimes present in PDFs
    text = text.replace("&#xA;", "\n")

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Remove "Page X of Y"
        if PAGE_PATTERN.fullmatch(line):
            continue

        # Remove standalone transcript date headers
        if DATE_PATTERN.fullmatch(line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_spaces(text: str) -> str:
    """
    Normalize unnecessary spaces while keeping paragraph/
    line structure intact.
    """

    cleaned_lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_transcript_from_pdf(
    pdf_path,
    skip_first_page: bool = True
) -> str:
    """
    Extract and clean an earnings-call transcript from a PDF.

    For the current Syrma dataset, physical page 1 is a
    stock-exchange cover letter, so skip_first_page=True.
    """

    doc = pymupdf.open(pdf_path)

    pages = []

    try:
        for page_number, page in enumerate(doc, start=1):

            if skip_first_page and page_number == 1:
                continue

            text = page.get_text("text")

            text = clean_transcript_page(text)
            text = normalize_spaces(text)

            if text:
                pages.append(text)

    finally:
        doc.close()

    return "\n".join(pages)