# Transformer Architecture

The transformer processes sequences using self-attention rather than recurrence.
Each token attends to every other token through query, key, and value
projections, letting the model capture long-range dependencies in parallel.
Multi-head attention runs several attention operations with different learned
projections. Positional encodings inject order information, since attention
itself is permutation-invariant. Transformers scale well with data and compute,
which is why they underpin modern large language models.
