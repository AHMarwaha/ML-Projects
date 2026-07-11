# Gradient Boosting

Gradient boosting builds an ensemble of shallow decision trees sequentially,
where each tree fits the residual errors of the ensemble so far. The learning
rate scales each tree's contribution and trades off against the number of trees.
Modern implementations use histogram-based split finding for speed and support
early stopping on a validation set. Gradient-boosted trees remain the strongest
baseline for tabular data, often outperforming neural networks on structured
problems with limited samples.
