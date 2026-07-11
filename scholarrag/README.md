# ScholarRAG: Retrieval-Augmented Generation with Systematic Evaluation

A retrieval-augmented generation (RAG) pipeline for question answering over
technical documents, built with a focus on the part most RAG demos skip:
**measuring whether the system actually works**.

Everything runs locally on CPU with open models. No API keys required.

## Why this project

RAG systems fail quietly. Retrieval can miss the relevant passage, and the
LLM will still produce a fluent answer from its own parametric memory, which
is exactly how hallucinations enter production systems. This project treats
the pipeline as a system under test:

- **Retrieval metrics**: hit@k and mean reciprocal rank (MRR) against a
  labelled QA dataset, so retrieval failures are visible before they become
  generation failures.
- **Groundedness scoring**: embedding similarity between the generated answer
  and the retrieved context, flagging answers that ignored the evidence.
- **Ablation study**: chunk size and top-k are evaluated over a grid rather
  than assumed, because retrieval configuration is an empirical question.

## Architecture

```
data/docs/*.md
     |
     v
[index.py]  chunk (overlapping windows) -> embed (MiniLM) -> vector store
     |
     v
[pipeline.py]  question -> embed -> top-k cosine retrieval -> grounded prompt
     |                                                            |
     v                                                            v
[evaluate.py]  hit@k, MRR, keyword accuracy, groundedness   flan-t5-base answer
```

Components:

| Stage | Implementation |
|---|---|
| Chunking | Overlapping word windows (configurable size/overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Exact cosine search over a NumPy matrix (FAISS-swappable) |
| Generator | `google/flan-t5-base` via Hugging Face Transformers |
| Evaluation | Labelled QA set, retrieval + answer + groundedness metrics |

## Quickstart

```bash
pip install -r requirements.txt
cd src

# 1. Build the vector store from the sample corpus
python index.py --docs ../data/docs --out ../store

# 2. Ask a question
python pipeline.py "Why can random CV splits not be used for time series?" --store ../store

# 3. Evaluate the full pipeline against the labelled QA set
python evaluate.py --store ../store --qa ../data/qa_dataset.json

# 4. Run the chunk-size / top-k ablation grid
python evaluate.py --ablate --docs ../data/docs --qa ../data/qa_dataset.json
```

First run downloads the two models (~600 MB total).

## Evaluation methodology

The QA dataset (`data/qa_dataset.json`) labels each question with its gold
source document and expected answer keywords. Metrics reported:

- **hit@k** — did retrieval surface the right document at all?
- **MRR** — how high did it rank?
- **keyword accuracy** — transparent proxy for answer correctness.
- **groundedness** — cosine similarity between answer and retrieved context;
  low values indicate the model answered from memory rather than evidence.

Separating retrieval metrics from answer metrics matters: a wrong answer with
perfect retrieval is a generation problem, a wrong answer with failed
retrieval is an indexing problem, and the fix is different in each case.

## Using your own corpus

Drop `.md` or `.txt` files into a directory and point `index.py --docs` at
it. To evaluate on your own domain, extend `qa_dataset.json` with questions,
gold source documents, and expected keywords.

## Design notes

- The vector store is exact cosine search over a normalised embedding matrix.
  For corpora beyond ~100k chunks, replace `VectorStore.search` with FAISS;
  the interface is one function.
- Generation is deterministic (`do_sample=False`) so evaluation runs are
  reproducible.
- The generator model is deliberately small (flan-t5-base, ~250M parameters)
  to keep the project runnable on a laptop; the evaluation harness is
  model-agnostic, so swapping in a larger instruction-tuned model requires
  changing one constant.

## Requirements

See `requirements.txt`. Python 3.10+.
