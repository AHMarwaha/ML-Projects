"""Evaluation harness for the ScholarRAG pipeline.

Most RAG demos stop at "it produced an answer". This module treats the
pipeline as a system under test and measures three things against a labelled
QA dataset (data/qa_dataset.json):

Retrieval quality
    - hit@k: fraction of questions whose gold source document appears in the
      top-k retrieved chunks.
    - MRR (mean reciprocal rank): position-sensitive version of the same idea.

Answer quality
    - keyword accuracy: fraction of expected answer keywords present in the
      generated answer (a cheap, transparent proxy for correctness).
    - groundedness: mean cosine similarity between the answer embedding and
      the retrieved-context embedding. Low groundedness flags answers the
      model produced from its own parametric memory rather than the context,
      the signature of hallucination in RAG systems.

Ablations
    - `--ablate` re-runs the full evaluation over a grid of chunk sizes and
      top-k values, writing a results table so the retrieval configuration is
      chosen from evidence, not defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from index import DEFAULT_EMBED_MODEL, build_store
from pipeline import RAGPipeline


# --------------------------------------------------------------------- metrics
def hit_at_k(retrieved_docs: list[str], gold_doc: str) -> float:
    """1.0 if the gold document is among the retrieved chunk sources."""
    return 1.0 if gold_doc in retrieved_docs else 0.0


def reciprocal_rank(retrieved_docs: list[str], gold_doc: str) -> float:
    """1/rank of the first chunk from the gold document, else 0."""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == gold_doc:
            return 1.0 / rank
    return 0.0


def keyword_accuracy(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0


def groundedness(answer: str, context: str, embedder: SentenceTransformer) -> float:
    """Cosine similarity between answer and retrieved context embeddings."""
    a, c = embedder.encode([answer, context])
    denom = np.linalg.norm(a) * np.linalg.norm(c)
    return float(a @ c / max(denom, 1e-12))


# ------------------------------------------------------------------ evaluation
def evaluate(store_dir: Path, qa_path: Path, top_k: int) -> dict:
    """Run the full pipeline over the QA dataset and aggregate metrics."""
    qa_items = json.loads(qa_path.read_text(encoding="utf-8"))
    rag = RAGPipeline(store_dir)
    embedder = SentenceTransformer(DEFAULT_EMBED_MODEL)

    per_metric: dict[str, list[float]] = {
        "hit_at_k": [], "mrr": [], "keyword_accuracy": [], "groundedness": []
    }

    for item in qa_items:
        result = rag.answer(item["question"], top_k=top_k)
        retrieved_docs = [chunk.doc_name for chunk, _ in result.retrieved]
        context = " ".join(chunk.text for chunk, _ in result.retrieved)

        per_metric["hit_at_k"].append(hit_at_k(retrieved_docs, item["source_doc"]))
        per_metric["mrr"].append(reciprocal_rank(retrieved_docs, item["source_doc"]))
        per_metric["keyword_accuracy"].append(
            keyword_accuracy(result.answer, item["answer_keywords"])
        )
        per_metric["groundedness"].append(groundedness(result.answer, context, embedder))

    return {name: float(np.mean(vals)) for name, vals in per_metric.items()}


def run_ablation(docs_dir: Path, qa_path: Path, out_path: Path) -> None:
    """Grid-search chunk size and top-k, evaluating each configuration."""
    chunk_sizes = [40, 80, 160]
    top_ks = [1, 3, 5]
    rows = []

    for chunk_size in chunk_sizes:
        store_dir = Path(f"store_cs{chunk_size}")
        build_store(docs_dir, store_dir, chunk_size=chunk_size, overlap=chunk_size // 4)
        for top_k in top_ks:
            metrics = evaluate(store_dir, qa_path, top_k=top_k)
            rows.append({"chunk_size": chunk_size, "top_k": top_k, **metrics})
            print(f"chunk_size={chunk_size} top_k={top_k} -> {metrics}")

    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nAblation results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ScholarRAG")
    parser.add_argument("--store", type=Path, default=Path("store"))
    parser.add_argument("--docs", type=Path, default=Path("data/docs"))
    parser.add_argument("--qa", type=Path, default=Path("data/qa_dataset.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ablate", action="store_true", help="Run the full ablation grid")
    args = parser.parse_args()

    if args.ablate:
        run_ablation(args.docs, args.qa, Path("ablation_results.json"))
    else:
        metrics = evaluate(args.store, args.qa, top_k=args.top_k)
        print(json.dumps(metrics, indent=2))
