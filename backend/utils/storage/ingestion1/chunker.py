"""
Chunks the already-extracted papers from data/processed/papers.jsonl (produced by
loader.py) — does NOT touch the PDFs or re-run extraction, so this can be run
independently and repeatedly while you tune chunk size/overlap, without waiting
~2 minutes to re-extract 30 PDFs every time.

LangChain's RecursiveCharacterTextSplitter does the actual text splitting. Custom
heading detection is used only to (a) tag each chunk with its section name and
(b) exclude the References section before any text reaches the splitter.

No OCR is used anywhere in this pipeline — these are arXiv-generated PDFs with a
real embedded text layer, extracted via pdfplumber in loader.py.

Expected input:  data/processed/papers.jsonl   (from loader.py)
Output:          data/processed/chunks.jsonl
"""
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from codebase.backend.utils.storage.ingestion1.loader import PaperRecord, PROCESSED_DIR   # ✅ relative import, matches your other files
    
INPUT_PATH = PROCESSED_DIR / "papers.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "chunks.jsonl"

CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 40
CHARS_PER_WORD = 6  # rough average for English technical text, incl. spaces

SECTION_HEADING_RE = re.compile(
    r'^\s*(?P<num>\d+(\.\d+)*)\s+(?P<title>[A-Za-z][^\n]{2,90})\s*$'
)
SPECIAL_HEADINGS = {"abstract", "references", "acknowledgments", "acknowledgements"}
REFERENCES_HEADINGS = {"references"}
COLUMN_SPLIT_RE = re.compile(r' {3,}')
VERB_SIGNALS = {
    "were", "was", "is", "are", "have", "has", "had", "will", "can", "could",
    "would", "should", "drawn", "used", "based", "shows", "show", "using",
    "randomly", "trained", "achieves", "achieved", "obtained",
}


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    metadata: dict


def load_processed_papers(path: Path = INPUT_PATH) -> List[PaperRecord]:
    """Reads the cached, already-extracted papers from loader.py's output."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No processed papers found at {path}. Run loader.py first "
            f"(python -m src.ingestion.loader)."
        )

    papers = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                papers.append(PaperRecord(**record))
    return papers


def _detect_heading(raw_line: str) -> Optional[str]:
    stripped = raw_line.strip("\f").strip()
    if not stripped:
        return None

    low = stripped.lower()
    if low in SPECIAL_HEADINGS:
        return "Abstract" if low == "abstract" else stripped.title()

    m = SECTION_HEADING_RE.match(stripped)
    if not m:
        return None

    columns = [c for c in COLUMN_SPLIT_RE.split(stripped) if c]
    if len(columns) > 2:
        return None  # table row, not a heading

    num = m.group("num")
    title = m.group("title").strip()

    if "." not in num and int(num) > 20:
        return None  # looks like a year / count, not a section number
    if title.endswith((".", ",", ";")):
        return None
    if len(title.split()) > 12:
        return None

    words = {w.strip(":,-").lower() for w in title.split()}
    if words & VERB_SIGNALS:
        return None

    return f"{num} {title}"


def split_into_sections(full_text: str) -> List[tuple]:
    """Returns a list of (section_name, section_text) tuples, in document order."""
    lines = full_text.split("\n")
    sections: List[tuple] = []
    current_name = "Unstructured"
    current_lines: List[str] = []

    for line in lines:
        heading = _detect_heading(line)
        if heading:
            if current_lines:
                sections.append((current_name, "\n".join(current_lines).strip()))
            current_name = heading
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))

    return [(name, text) for name, text in sections if text.strip()]


def _normalize_whitespace(text: str) -> str:
    """Collapses layout padding and rejoins hyphen-wrapped words, e.g. 'gate-\\nway'."""
    text = re.sub(r'(\w)-\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_WORDS * CHARS_PER_WORD,
        chunk_overlap=CHUNK_OVERLAP_WORDS * CHARS_PER_WORD,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_paper(paper: PaperRecord, splitter: RecursiveCharacterTextSplitter = None) -> List[Chunk]:
    splitter = splitter or get_splitter()
    sections = split_into_sections(paper.full_text)
    chunks: List[Chunk] = []
    chunk_idx = 0

    for section_name, section_text in sections:
        if section_name.strip().lower() in REFERENCES_HEADINGS:
            continue  # skip citation lists — not useful retrieval content

        cleaned_section = _normalize_whitespace(section_text)
        pieces = splitter.split_text(cleaned_section)

        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.arxiv_id}::{chunk_idx}",
                    paper_id=paper.arxiv_id,
                    section=section_name,
                    text=piece,
                    metadata={
                        "title": paper.title,
                        "category": paper.category,
                        "section": section_name,
                        "arxiv_id": paper.arxiv_id,
                        "arxiv_url": paper.arxiv_url,
                        "published_month": paper.published_month,
                    },
                )
            )
            chunk_idx += 1

    return chunks


def chunk_all(papers: List[PaperRecord]) -> List[Chunk]:
    """Chunks every paper, printing per-paper progress (mirrors loader.py's style)."""
    print("=" * 70)
    print("CHUNKER STARTED")
    print("=" * 70)

    overall_start = time.perf_counter()
    splitter = get_splitter()
    total_papers = len(papers)
    all_chunks: List[Chunk] = []

    for index, paper in enumerate(papers, start=1):
        start = time.perf_counter()
        paper_chunks = chunk_paper(paper, splitter)
        elapsed = time.perf_counter() - start

        print(
            f"[{index}/{total_papers}] {paper.arxiv_id} "
            f"({paper.category}) -> {len(paper_chunks)} chunks "
            f"({elapsed:.2f}s)"
        )
        all_chunks.extend(paper_chunks)

    total_elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 70)
    print("CHUNKER FINISHED")
    print("=" * 70)
    print(f"Papers processed : {total_papers}")
    print(f"Total chunks     : {len(all_chunks)}")
    if all_chunks:
        word_counts = [len(c.text.split()) for c in all_chunks]
        print(f"Avg words/chunk  : {sum(word_counts) / len(word_counts):.1f}")
        print(f"Max words/chunk  : {max(word_counts)}")
    print(f"Total time       : {total_elapsed:.2f} seconds")
    print("=" * 70)

    return all_chunks


def save_chunks(chunks: List[Chunk], output_path: Path = OUTPUT_PATH) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to: {output_path}")


def main() -> None:
    try:
        papers = load_processed_papers()
        print(f"Loaded {len(papers)} processed papers from {INPUT_PATH}\n")

        chunks = chunk_all(papers)

        if chunks:
            save_chunks(chunks)

            sample = chunks[0]
            print("\n" + "=" * 70)
            print("SAMPLE CHUNK")
            print("=" * 70)
            print(f"Paper   : {sample.metadata['title']}")
            print(f"Section : {sample.section}")
            print(f"Chunk ID: {sample.chunk_id}")
            print("-" * 70)
            print(sample.text[:400])
            print("-" * 70)

    except Exception as error:
        print("\nCHUNKER STOPPED DUE TO A FATAL ERROR")
        print(f"Error: {error}")
        raise


if __name__ == "__main__":
    main()