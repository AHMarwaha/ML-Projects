"""Retrieval-augmented generation pipeline.

Given a natural-language question, this module:

1. Embeds the question with the same sentence-transformer used at indexing
   time (embedding-space mismatch is a classic RAG bug, so the model name is
   stored alongside usage in the README).
2. Retrieves the top-k most relevant chunks from the vector store.
3. Builds a grounded prompt and generates an answer with a local
   seq2seq LLM (default: google/flan-t5-base, ~250M parameters, runs on CPU).

Everything runs locally with no API keys, which keeps the project fully
reproducible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

from index import DEFAULT_EMBED_MODEL, VectorStore, Chunk

DEFAULT_GEN_MODEL = "google/flan-t5-base"

PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


@dataclass
class RAGResult:
    """Bundle of everything the pipeline produced for one question."""

    question: str
    answer: str
    retrieved: list[tuple[Chunk, float]]


class RAGPipeline:
    """End-to-end retrieve-then-generate pipeline."""

    def __init__(
        self,
        store_dir: Path,
        embed_model: str = DEFAULT_EMBED_MODEL,
        gen_model: str = DEFAULT_GEN_MODEL,
    ):
        self.store = VectorStore.load(store_dir)
        self.embedder = SentenceTransformer(embed_model)
        # max_new_tokens bounds generation cost; deterministic decoding keeps
        # evaluation runs reproducible.
        self.generator = hf_pipeline(
            "text2text-generation", model=gen_model, max_new_tokens=128, do_sample=False
        )

    def retrieve(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the top_k (chunk, cosine score) pairs for a question."""
        q_emb = self.embedder.encode(question)
        return self.store.search(q_emb, top_k=top_k)

    def answer(self, question: str, top_k: int = 3) -> RAGResult:
        """Retrieve context and generate a grounded answer."""
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n---\n".join(chunk.text for chunk, _ in retrieved)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.generator(prompt)[0]["generated_text"].strip()
        return RAGResult(question=question, answer=answer, retrieved=retrieved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask ScholarRAG a question")
    parser.add_argument("question", type=str)
    parser.add_argument("--store", type=Path, default=Path("store"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    rag = RAGPipeline(args.store)
    result = rag.answer(args.question, top_k=args.top_k)

    print(f"\nQ: {result.question}\nA: {result.answer}\n")
    print("Sources:")
    for chunk, score in result.retrieved:
        print(f"  [{score:.3f}] {chunk.doc_name} (chunk {chunk.chunk_id})")
