# Anomaly Detection

Anomaly detection identifies observations that deviate from expected behaviour.
Statistical approaches flag points outside control limits, as in statistical
process control. Isolation Forest isolates anomalies with random splits, since
outliers are separated in fewer splits than normal points. Reconstruction-based
methods train an autoencoder on normal data and flag inputs with high
reconstruction error. In streaming industrial settings, detectors must balance
sensitivity against false-alarm rate and adapt to gradual drift.
