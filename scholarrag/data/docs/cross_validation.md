# Cross-Validation

Cross-validation estimates out-of-sample error by repeatedly splitting data into
training and validation folds. In k-fold cross-validation the data is divided
into k parts; each part serves once as the validation set. Leave-one-out
cross-validation is the extreme case with k equal to the sample size. For time
series, random splits leak future information into the past, so walk-forward
validation is used instead: the model trains on a rolling historical window and
is evaluated on the period immediately after it.
