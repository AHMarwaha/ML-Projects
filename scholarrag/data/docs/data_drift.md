# Data Drift and Model Monitoring

A deployed model degrades when the data it receives departs from the data it
was trained on. Covariate drift is a change in the input distribution;
concept drift is a change in the relationship between inputs and target.
Drift is detected by comparing recent feature distributions against a training
reference, using the population stability index or the Kolmogorov-Smirnov test.
When drift is confirmed, the standard response is retraining on recent data,
gated by an evaluation step before the new model is promoted.
