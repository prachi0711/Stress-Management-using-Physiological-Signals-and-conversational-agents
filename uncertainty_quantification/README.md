# Uncertainty Quantification (UQ) for Stress Classification

This module extends the stress classification pipeline by incorporating  UQ, which enables the system to measure how confident it is about its predictions.

Two different UQ approaches are implemented depending on the model:

1. **CNN with Monte Carlo (MC) Dropout**

Implemented in `cnn_UQ.py`.

Uses MC Dropout at inference:

 - Dropout is kept active during prediction.
  
 - Multiple stochastic forward passes (e.g., 50) are made for each input.
  
 - Class probability distributions are averaged.
  
 - Entropy is computed to quantify prediction uncertainty.

2. **Random Forest with Predictive Entropy**

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

## Usage

#### **1. CNN with UQ**

```bash
cd uncertainty_quantification
python cnn_UQ.py
```

* Performs Leave-One-Subject-Out (LOSO) cross-validation.
* Trains a CNN with early stopping.
* Uses MC Dropout for uncertainty estimation.
* Saves:

  * Trained CNN weights → `cnn_uq_model_60.pt`
  * Scaler → `cnn_scaler.save`
---

#### **2. Random Forest with UQ**

```bash
cd uncertainty_quantification
python random_forest_UQ.py
```

* Performs LOSO cross-validation.
* Trains a Random Forest on top-selected features.
* Uses predictive entropy for UQ.
* Saves:

  * Final RF model → `rf_uq_model_60.pkl`
  * Scaler → `rf_scaler.save`


Next Step: [Dialogue Manager](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/dialogue_manager/README.md)


