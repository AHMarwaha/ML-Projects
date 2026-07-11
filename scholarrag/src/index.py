"""Document chunking and embedding-based vector store.

This module handles the ingestion side of the RAG pipeline:

1. Load raw markdown/text documents from a directory.
2. Split each document into overlapping chunks of a configurable size.
3. Embed every chunk with a sentence-transformer model.
4. Persist the embeddings and chunk metadata to disk as a lightweight
   NumPy-based vector store.

The store is deliberately simple (exact cosine search over a matrix). For
corpora up to a few hundred thousand chunks this is fast enough and removes
an external dependency. Swapping in FAISS or a hosted vector database is a
one-function change (see `VectorStore.search`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    """A single retrievable unit of text."""

    chunk_id: int
    doc_name: str
    text: str


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    """Split text into overlapping word-window chunks.

    Args:
        text: Raw document text.
        chunk_size: Window length in words.
        overlap: Number of words shared between consecutive chunks. Overlap
            prevents an answer from being cut in half at a chunk boundary.

    Returns:
        List of chunk strings in document order.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_and_chunk(docs_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """Load every .md/.txt file in a directory and chunk it."""
    chunks: list[Chunk] = []
    chunk_id = 0
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for piece in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(Chunk(chunk_id=chunk_id, doc_name=path.name, text=piece))
            chunk_id += 1
    return chunks


class VectorStore:
    """Exact cosine-similarity search over an embedding matrix."""

    def __init__(self, embeddings: np.ndarray, chunks: list[Chunk]):
        # Normalise once so cosine similarity reduces to a dot product.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.clip(norms, 1e-12, None)
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the top_k most similar chunks with their scores."""
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    # ---------------------------------------------------------------- persistence
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.embeddings)
        meta = [asdict(c) for c in self.chunks]
        (out_dir / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, store_dir: Path) -> "VectorStore":
        embeddings = np.load(store_dir / "embeddings.npy")
        meta = json.loads((store_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**m) for m in meta]
        return cls(embeddings, chunks)


def build_store(
    docs_dir: Path,
    out_dir: Path,
    chunk_size: int = 80,
    overlap: int = 20,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> VectorStore:
    """Full ingestion: load, chunk, embed, persist. Returns the built store."""
    chunks = load_and_chunk(docs_dir, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise FileNotFoundError(f"No .md or .txt documents found in {docs_dir}")

    print(f"Embedding {len(chunks)} chunks with {embed_model} ...")
    model = SentenceTransformer(embed_model)
    embeddings = model.encode([c.text for c in chunks], show_progress_bar=True)

    store = VectorStore(np.asarray(embeddings), chunks)
    store.save(out_dir)
    print(f"Vector store written to {out_dir}")
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the ScholarRAG vector store")
    parser.add_argument("--docs", type=Path, default=Path("data/docs"))
    parser.add_argument("--out", type=Path, default=Path("store"))
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()

    build_store(args.docs, args.out, args.chunk_size, args.overlap, args.embed_model)
