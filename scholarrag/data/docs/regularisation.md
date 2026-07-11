# Regularisation in Linear Models

Regularisation adds a penalty on coefficient size to the least-squares objective.
Ridge regression uses an L2 penalty, shrinking coefficients smoothly toward zero,
which stabilises estimates when predictors are highly correlated. Lasso uses an
L1 penalty, driving some coefficients exactly to zero and performing variable
selection. The regularisation strength is a hyperparameter, typically chosen by
cross-validation. Ridge is preferred when many predictors carry signal; lasso is
preferred when the true model is sparse. Elastic net combines both penalties.
