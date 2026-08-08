"""
Loads full-text documents from a metadata-driven PDF corpus into normalized
in-memory records, and extracts each PDF's text with pdfplumber.

The metadata file (paper_metadata.jsonl) is the source of truth for titles,
categories, dates, URLs, filenames, and local PDF paths. Text is extracted
from each PDF and saved to data/processed/papers.jsonl.

Expected folder layout (paths are configurable via RAW_DATA_DIR /
PROCESSED_DATA_DIR in .env — see storage/config.py):

    backend/
    ├── data/
    │   ├── raw/
    │   │   ├── paper_metadata.jsonl
    │   │   └── papers/
    │   │       └── <category>/<id>v<version>.pdf
    │   └── processed/
    └── utils/
        └── storage/
            └── loader.py

To add more documents later:
1. Add the PDF under data/raw/papers/<category>/.
2. Append its metadata record to paper_metadata.jsonl.
3. Re-run the pipeline (pipeline.py) or just this loader.
"""
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber

from .config import BACKEND_ROOT, METADATA_PATH, PAPERS_PATH

logging.getLogger("pdfminer").setLevel(logging.ERROR)

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
            "Expected location: data/raw/paper_metadata.jsonl (override with "
            "RAW_DATA_DIR in .env)."
        )

    records: List[dict] = []

    required_fields = {
        "arxiv_id", "filename", "title", "category",
        "arxiv_url", "pdf_url", "local_path",
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
                    f"Missing required fields in {path} at line {line_number}: {missing}"
                )

            records.append(record)

    return records


def _resolve_pdf_path(local_path: str) -> Path:
    """Resolve a PDF path from metadata relative to the backend root.

    Example: data/raw/papers/tool-use/2205.00445v1.pdf
    """
    pdf_path = Path(local_path)
    if not pdf_path.is_absolute():
        pdf_path = BACKEND_ROOT / pdf_path
    return pdf_path.resolve()


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from each PDF page while showing page progress."""
    pages_text: List[str] = []
    print("  [1/2] Opening PDF and extracting text...", flush=True)

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, start=1):
            print(f"        Extracting page {page_number}/{total_pages}", end="\r", flush=True)
            try:
                page_text = page.extract_text(layout=True) or ""
            except Exception as error:
                print()
                print(f"        WARNING: Could not extract page {page_number}: {error}")
                page_text = ""
            pages_text.append(page_text)
        print()

    print(f"  [1/2] Text extraction completed: {len(pages_text)} pages processed", flush=True)
    return "\n\n".join(pages_text)


def _clean_text(raw_text: str) -> str:
    """Remove standalone page numbers and trailing spaces."""
    cleaned_lines: List[str] = []
    for line in raw_text.splitlines():
        if PAGE_NUMBER_RE.match(line):
            continue
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


def load_papers(metadata: Optional[List[dict]] = None) -> List[PaperRecord]:
    """Load all papers and return normalized PaperRecord objects."""
    print("=" * 70)
    print("LOADER STARTED")
    print("=" * 70)

    overall_start_time = time.perf_counter()
    print(f"Backend root : {BACKEND_ROOT}")
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
        print(f"DOCUMENT {index}/{total_papers}")
        print(f"Title    : {title}")
        print(f"ID       : {arxiv_id}")
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
            print(f"  [2/2] Text cleaning completed: {len(clean_text):,} characters", flush=True)

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
                arxiv_primary_category=meta.get("arxiv_primary_category"),
                published_month=meta.get("published_month"),
                published_date_exact=meta.get("published_date_exact"),
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
            print("Continuing with the next document...")

    total_elapsed = time.perf_counter() - overall_start_time

    print("\n" + "=" * 70)
    print("LOADER FINISHED")
    print("=" * 70)
    print(f"Metadata records   : {total_papers}")
    print(f"Successfully loaded: {len(papers)}")
    print(f"Skipped PDFs       : {skipped_count}")
    print(f"Failed PDFs        : {failed_count}")
    print(f"Total time         : {total_elapsed:.2f} seconds")
    print("=" * 70)

    return papers


def save_processed_papers(papers: List[PaperRecord], output_path: Path = PAPERS_PATH) -> None:
    """Save all extracted paper records to one JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nSaving processed papers...")
    print(f"Output file: {output_path}")

    with output_path.open("w", encoding="utf-8") as file:
        for paper in papers:
            file.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")

    print(f"Saved {len(papers)} paper records successfully.")


def run_loader() -> List[PaperRecord]:
    """Runs the complete extraction and saving step. Used by pipeline.py."""
    papers = load_papers()
    if not papers:
        print("\nNo papers were successfully loaded.")
        return []
    save_processed_papers(papers)
    return papers


if __name__ == "__main__":
    run_loader()
