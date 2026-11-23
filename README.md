# Pulmonary-Fibrosis-Disease-Progression
Forecasting the decline in lung function (FVC, measured by spirometry) of patients with pulmonary fibrosis from sparse, irregular follow-up visits, and quantifying the uncertainty of each forecast.

# Problem

For each patient, given a baseline CT scan, clinical metadata (age, sex, smoking status) and a baseline FVC, predict FVC at future weeks together with a confidence value (sigma). Submissions are scored with a modified Laplace log-likelihood (higher is better; sigma is clipped at 70 and errors at 1000):

score = -sqrt(2) * delta / sigma_clipped - ln(sqrt(2) * sigma_clipped)

# Data

Downloaded from Kaggle

# What is in the notebook

notebooks/osic-pulmonary-fibrosis-progression.ipynb has three parts:

1. Exploratory analysis: FVC distribution, per-patient trajectories, feature interactions by sex and smoking status, correlations. The only strong correlation is between FVC and Percent (Percent is derived from FVC).
2.Two-stage ElasticNet: pairs every visit with every other visit of the same patient (baseline visit + weeks passed), predicts FVC with an ElasticNet (alpha=0.3, l1_ratio=0.8), then predicts the per-row confidence with a second ElasticNet. Cross-validation is a 7-fold GroupKFold by patient.
3.Quantile neural network: a small dense network (two hidden layers of 100 units) on tabular features that outputs the 0.2, 0.5 and 0.8 quantiles, trained with a blend of pinball loss and the competition metric, 7-fold CV. The code also defines a DICOM loading function, but it is not called, so this model uses tabular features only despite the "CNN" label in the notebook.
