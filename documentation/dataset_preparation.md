# Dataset Preparation — AI Agent Capability Papers

## 1. Source

Paper list curated from [`masamasa59/ai-agent-papers`](https://github.com/masamasa59/ai-agent-papers),
a GitHub repository that organizes AI agent research by capability. Two categories were
selected for this project:

| Bibliography file | Capability | Papers |
|---|---|---|
| `capability-papers/tool-use.md` | Whether/how agents use external tools | 19 |
| `capability-papers/knowledge.md` | How agents access/use external or internal knowledge | 11 |

Full PDFs (23–60 pages each) were downloaded from arXiv via the links in these files —
not just abstracts, so the corpus supports genuine section-level retrieval and multi-hop
reasoning across papers, rather than shallow one-paragraph lookups.

## 2. Why arXiv-ID matching instead of fuzzy title matching

PDF filenames follow arXiv's own ID convention (e.g. `2205.00445v1.pdf`). Rather than
guessing which paper is which from noisy extracted text, the bibliography markdown files
were fetched directly from GitHub and parsed into an **arXiv-ID → {title, month, category}**
lookup table. Every downloaded PDF's filename was matched against this table by ID.

Result: **30 / 30 PDFs matched exactly**, with zero ambiguity — no fuzzy string matching,
no manual guesswork.

## 3. Metadata enrichment from the PDFs themselves

Titles and rough publish months came from the bibliography. Two additional fields were
extracted directly from each PDF's first page, where arXiv stamps a footer like:

```
arXiv:2205.00445v1 [cs.CL] 1 May 2022
```

- `arxiv_primary_category` — arXiv's own subject tag (e.g. `cs.CL`, `cs.AI`)
- `published_date_exact` — full submission date, not just month

This succeeded for **29 of 30 papers**; one PDF doesn't carry the footer stamp on page 1,
so it falls back to the bibliography's month-level date only. This was verified by
manually inspecting the extracted text for that paper rather than assuming a parsing bug.

## 4. Validation step

Before trusting the automated pipeline across all 30 files, one paper
(`2205.00445v1.pdf`, "MRKL Systems") was checked by hand: extracted PDF text was compared
against the bibliography entry and confirmed to match exactly (title, and the arXiv
footer's category/date). Only after this spot-check did the full batch run.

## 5. Output structure

```
data/raw/
├── paper_metadata.jsonl              # one record per paper, combined
└── papers/
    ├── tool-use/                     # 19 papers
    │   ├── 2205.00445v1.pdf
    │   ├── 2205.00445v1.json         # per-paper metadata sidecar
    │   └── ...
    └── knowledge/                    # 11 papers
        └── ...
```

Each metadata record contains:

| Field | Example |
|---|---|
| `arxiv_id` | `2205.00445` |
| `version` | `1` |
| `title` | `"MRKL Systems: A modular, neuro-symbolic architecture..."` |
| `category` | `tool-use` |
| `arxiv_primary_category` | `cs.CL` |
| `published_month` | `May 2022` |
| `published_date_exact` | `2022-05-01` |
| `num_pages` | `19` |
| `arxiv_url` / `pdf_url` | arXiv abstract/PDF links |
| `source_repo` / `source_bib_file` | provenance back to the GitHub bibliography |
| `local_path` | relative path to the PDF in this repo |

## 6. What this replaces in the original scaffold

The initial project scaffold assumed a single flat `data/raw/arxiv_abstracts.jsonl` file
of short abstracts. That assumption no longer holds:

- **Corpus is now full-text PDFs**, not abstract strings → `loader.py` needs a PDF text
  extraction step, not a JSONL reader.
- **Chunking can no longer be "one chunk per paper"** → papers are 23–60 pages, so
  `chunker.py` needs section-aware or token-windowed splitting.
- **Category is now a first-class field** (`tool-use` vs `knowledge`) → useful both as
  vector-store metadata (filterable retrieval) and as a `Paper.category` property in the
  Neo4j knowledge graph.

## 7. Known limitations

- Category labels (`tool-use` / `knowledge`) reflect how the source repo classified each
  paper, not an independent judgment — a paper could arguably span both.
- One paper is missing `arxiv_primary_category` / `published_date_exact` (see §3).
- Author names were not extracted at this stage (left for the knowledge-graph extraction
  step in Week 1, Day 5–6, where entity/relation extraction runs against full text).
