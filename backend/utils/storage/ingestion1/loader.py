"""
Load full-text research papers from the organized PDF corpus into normalized
in-memory records.

The metadata file is the source of truth for paper titles, categories, dates,
URLs, filenames, and local PDF paths. Text is extracted from each PDF using
pdfplumber and saved to data/processed/papers.jsonl.

Expected project layout:

    codebase/
    ├── data/
    │   ├── raw/
    │   │   ├── paper_metadata.jsonl
    │   │   └── papers/
    │   │       └── <category>/<arxiv_id>v<version>.pdf
    │   └── processed/
    └── src/
        └── ingestion/
            └── loader.py

To add more papers later:
1. Add the PDF to the correct category folder.
2. Append its metadata record to paper_metadata.jsonl.
3. Run this loader again.
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber


logging.getLogger("pdfminer").setLevel(logging.ERROR)


# loader.py location:
# <project_root>/src/ingestion/loader.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = Path(r"E:\4.Project 1\codebase\backend\data\raw\paper_metadata.jsonl")
PROCESSED_DIR = Path(r"E:\4.Project 1\codebase\backend\data\processed")
OUTPUT_PATH = PROCESSED_DIR / "papers.jsonl"

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")


@dataclass
class PaperRecord:
    arxiv_id: str
    version: Optional[int]
    filename: str
    title: str
    category: str
    arxiv_primary_category: Optional[str]
    published_month: Optional[str]
    published_date_exact: Optional[str]
    num_pages: Optional[int]
    arxiv_url: str
    pdf_url: str
    local_path: str
    full_text: str = field(default="", repr=False)


def load_metadata(path: Path = METADATA_PATH) -> List[dict]:
    """Load and validate metadata records from the JSONL file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}\n"
            "Expected location: data/raw/paper_metadata.jsonl"
        )

    records: List[dict] = []

    required_fields = {
        "arxiv_id",
        "filename",
        "title",
        "category",
        "arxiv_url",
        "pdf_url",
        "local_path",
    }

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error

            missing_fields = required_fields.difference(record)

            if missing_fields:
                missing = ", ".join(sorted(missing_fields))

                raise ValueError(
                    f"Missing required fields in {path} at line "
                    f"{line_number}: {missing}"
                )

            records.append(record)

    return records


def _resolve_pdf_path(local_path: str) -> Path:
    """
    Resolve a PDF path from metadata relative to the project root.

    Example:
        data/raw/papers/tool-use/2205.00445v1.pdf
    """
    pdf_path = Path(local_path)

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    return pdf_path.resolve()


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from each PDF page while showing page progress."""
    pages_text: List[str] = []

    print("  [1/2] Opening PDF and extracting text...", flush=True)

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):
            print(
                f"        Extracting page {page_number}/{total_pages}",
                end="\r",
                flush=True,
            )

            try:
                page_text = page.extract_text(layout=True) or ""

            except Exception as error:
                print()

                print(
                    f"        WARNING: Could not extract page "
                    f"{page_number}: {error}"
                )

                page_text = ""

            pages_text.append(page_text)

        print()

    print(
        f"  [1/2] Text extraction completed: "
        f"{len(pages_text)} pages processed",
        flush=True,
    )

    return "\n\n".join(pages_text)


def _clean_text(raw_text: str) -> str:
    """Remove standalone page numbers and trailing spaces."""
    cleaned_lines: List[str] = []

    for line in raw_text.splitlines():
        if PAGE_NUMBER_RE.match(line):
            continue

        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines).strip()


def load_papers(
    metadata: Optional[List[dict]] = None,
) -> List[PaperRecord]:
    """Load all papers and return normalized PaperRecord objects."""
    print("=" * 70)
    print("RESEARCH PAPER LOADER STARTED")
    print("=" * 70)

    overall_start_time = time.perf_counter()

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Metadata file: {METADATA_PATH}")
    print("\nLoading metadata...", flush=True)

    if metadata is None:
        metadata = load_metadata()

    total_papers = len(metadata)

    print("Metadata loaded successfully.")
    print(f"Total metadata records: {total_papers}")

    papers: List[PaperRecord] = []
    skipped_count = 0
    failed_count = 0

    for index, meta in enumerate(metadata, start=1):
        paper_start_time = time.perf_counter()

        title = meta["title"]
        arxiv_id = meta["arxiv_id"]
        pdf_path = _resolve_pdf_path(meta["local_path"])

        print("\n" + "-" * 70)
        print(f"PAPER {index}/{total_papers}")
        print(f"Title    : {title}")
        print(f"arXiv ID : {arxiv_id}")
        print(f"Category : {meta['category']}")
        print(f"PDF path : {pdf_path}")
        print("-" * 70)

        if not pdf_path.exists():
            skipped_count += 1

            print("STATUS: SKIPPED")
            print(f"Reason: PDF not found at {pdf_path}")

            continue

        try:
            raw_text = _extract_pdf_text(pdf_path)

            print("  [2/2] Cleaning extracted text...", flush=True)

            clean_text = _clean_text(raw_text)

            print(
                f"  [2/2] Text cleaning completed: "
                f"{len(clean_text):,} characters",
                flush=True,
            )

            if not clean_text:
                failed_count += 1

                print("STATUS: FAILED")
                print("Reason: No extractable text was found in this PDF.")

                continue

            paper_record = PaperRecord(
                arxiv_id=meta["arxiv_id"],
                version=meta.get("version"),
                filename=meta["filename"],
                title=meta["title"],
                category=meta["category"],
                arxiv_primary_category=meta.get(
                    "arxiv_primary_category"
                ),
                published_month=meta.get("published_month"),
                published_date_exact=meta.get(
                    "published_date_exact"
                ),
                num_pages=meta.get("num_pages"),
                arxiv_url=meta["arxiv_url"],
                pdf_url=meta["pdf_url"],
                local_path=str(pdf_path),
                full_text=clean_text,
            )

            papers.append(paper_record)

            elapsed = time.perf_counter() - paper_start_time

            print("STATUS: COMPLETED")
            print(f"Processing time: {elapsed:.2f} seconds")

        except Exception as error:
            failed_count += 1

            print("STATUS: FAILED")
            print(f"Error: {error}")
            print("Continuing with the next paper...")

    total_elapsed = time.perf_counter() - overall_start_time

    print("\n" + "=" * 70)
    print("RESEARCH PAPER LOADER FINISHED")
    print("=" * 70)
    print(f"Metadata records   : {total_papers}")
    print(f"Successfully loaded: {len(papers)}")
    print(f"Skipped PDFs       : {skipped_count}")
    print(f"Failed PDFs        : {failed_count}")
    print(f"Total time         : {total_elapsed:.2f} seconds")
    print("=" * 70)

    return papers


def save_processed_papers(
    papers: List[PaperRecord],
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save all extracted paper records to one JSONL file."""
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nSaving processed papers...")
    print(f"Output file: {output_path}")

    with output_path.open("w", encoding="utf-8") as file:
        for paper in papers:
            record = asdict(paper)

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Saved {len(papers)} paper records successfully.")


def main() -> None:
    """Run the complete extraction and saving pipeline."""
    try:
        papers = load_papers()

        if not papers:
            print("\nNo papers were successfully loaded.")
            return

        save_processed_papers(papers)

        sample = papers[0]

        print("\n" + "=" * 70)
        print("SAMPLE LOADED PAPER")
        print("=" * 70)
        print(f"Title          : {sample.title}")
        print(f"Category       : {sample.category}")
        print(f"arXiv ID       : {sample.arxiv_id}")
        print(f"Text characters: {len(sample.full_text):,}")
        print(f"Saved output   : {OUTPUT_PATH}")

        print("\nFirst 500 extracted characters:")
        print("-" * 70)
        print(sample.full_text[:500])
        print("-" * 70)

    except Exception as error:
        print("\nLOADER STOPPED DUE TO A FATAL ERROR")
        print(f"Error: {error}")

        raise


if __name__ == "__main__":
    main()