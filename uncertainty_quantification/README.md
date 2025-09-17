# Uncertainty Quantification (UQ) for Stress Classification

This module extends the stress classification pipeline by incorporating  UQ, which enables the system to measure how confident it is about its predictions.

Two different UQ approaches are implemented depending on the model:

1. CNN with Monte Carlo (MC) Dropout

Implemented in `cnn_UQ.py`.

Uses MC Dropout at inference:

 - Dropout is kept active during prediction.
  
 - Multiple stochastic forward passes (e.g., 50) are made for each input.
  
 - Class probability distributions are averaged.
  
 - Entropy is computed to quantify prediction uncertainty.

2. Random Forest with Predictive Entropy

Implemented in `random_forest_UQ.py`.

- Uses class probability distributions from the ensemble of trees.

- Entropy of predicted probabilities serves as an uncertainty score.

## Evaluation Metrics

Both scripts provide:

- Accuracy and F1 scores

Results of:

 - Correct & Certain predictions
  
 - Correct & Uncertain predictions
  
 - Incorrect & Certain predictions
  
 - Incorrect & Uncertain predictions

Next Step: [Dialogue Manager]()
