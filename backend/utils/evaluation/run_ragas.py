"""
RAGAS-based quality scoring — a numeric alternative to "looks right to me".

Scores two metrics that do NOT require a hand-written reference/ground-truth
answer (only question + answer + retrieved contexts are needed):

    faithfulness      — is the answer actually supported by the retrieved context,
                         or did the LLM state things the context doesn't back up?
    answer_relevancy  — does the answer actually address the question asked?

context_precision / context_recall are deliberately NOT included here — those need
a hand-written reference answer per question to judge retrieval quality against,
which golden_set.json doesn't currently provide. Add a "reference_answer" field per
item and wire those metrics in later if you want retrieval-quality scoring
specifically, as opposed to answer-quality scoring.

RAGAS defaults to OpenAI for its judge LLM and embeddings. This project only has a
GROQ_API_KEY configured, so both are swapped: Groq (via langchain-groq) as the judge,
and the same BGE embedding model dense_retriever.py already uses (via
langchain-huggingface) — no new API key needed.

Evaluated PER SUB-QUESTION, not per top-level question: each sub-answer is judged
against the specific context chunks retrieved and used to generate IT, which is what
faithfulness is actually meant to check. Single-hop questions produce one row;
multi-hop questions produce one row per hop, so you can see if one hop is dragging
the average down instead of only getting one blended score for the whole thing.

Always bypasses the cache (calls run_pipeline() directly, not ask()).

Requires (not part of the core pipeline's dependencies):
    pip install ragas datasets langchain-groq langchain-huggingface

NOTE: RAGAS's exact API has shifted across versions (this targets a "ragas>=0.2"
style API — Dataset-based evaluate(), LangchainLLMWrapper/LangchainEmbeddingsWrapper
for custom providers). If your installed version differs, check the ragas.metrics /
ragas.llms / ragas.embeddings import paths still match — same kind of version caveat
as tracing.py has for Langfuse v4.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from codebase.backend.utils.agent.graph_definition import run_pipeline
from codebase.backend.utils.evaluation.run_golden_set import load_golden_set, REPORTS_DIR

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # same model dense_retriever.py embeds chunks with


def _build_ragas_llm_and_embeddings():
    """Wires RAGAS's judge LLM + embeddings to Groq/BGE instead of the OpenAI
    default, since that's the only API key this project has configured."""
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — RAGAS needs a judge LLM to score faithfulness/relevancy.")

    chat = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def collect_ragas_rows() -> list:
    """
    Runs every golden-set question through the full pipeline and flattens the
    result into one row per SUB-QUESTION: {golden_id, question, answer, contexts}.
    """
    golden_set = load_golden_set()
    rows = []

    for item in golden_set:
        print(f"[RAGAS] Running: {item['id']} — {item['question']}")
        result_state = run_pipeline(item["question"])

        for sa in result_state["sub_answers"]:
            rows.append({
                "golden_id": item["id"],
                "question": sa["sub_question"],
                "answer": sa["answer"],
                "contexts": sa.get("context_texts") or ["(no context retrieved)"],
            })

    return rows


def run_ragas_eval() -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    rows = collect_ragas_rows()
    if not rows:
        raise RuntimeError("No rows collected — check golden_set.json isn't empty.")

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
    })

    llm, embeddings = _build_ragas_llm_and_embeddings()

    print(f"\n[RAGAS] Scoring {len(rows)} sub-question row(s) with faithfulness + answer_relevancy...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )
    scored_df = result.to_pandas()

    per_row = []
    for i in range(len(scored_df)):
        row = scored_df.iloc[i]
        faith = row["faithfulness"]
        rel = row["answer_relevancy"]
        per_row.append({
            "golden_id": rows[i]["golden_id"],
            "question": row["question"],
            "faithfulness": None if faith != faith else float(faith),      # NaN-safe (RAGAS can return NaN if judging failed)
            "answer_relevancy": None if rel != rel else float(rel),
        })

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "mean_faithfulness": float(scored_df["faithfulness"].mean()),
        "mean_answer_relevancy": float(scored_df["answer_relevancy"].mean()),
        "per_row": per_row,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"ragas_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"RAGAS COMPLETE — mean faithfulness={summary['mean_faithfulness']:.3f}, "
          f"mean answer_relevancy={summary['mean_answer_relevancy']:.3f}")
    print(f"Report written to: {report_path}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_ragas_eval()