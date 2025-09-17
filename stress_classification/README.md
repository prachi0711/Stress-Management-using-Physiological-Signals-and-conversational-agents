# Stress Classification Model

This document contains information regarding machine learning models for classifying psychological stress using preprocessed Electrodermal Activity (EDA) and Inter-Beat Interval (IBI) data from the WESAD dataset into stress and non-stress level. The project follows a complete pipeline from data preprocessing to model evaluation using a Leave-One-Subject-Out (LOSO) cross-validation approach.

## Models Implemented

Three distinct classifiers were compared to establish a strong baseline and explore more complex approaches:

1.  **Simple Decision Tree**: A baseline model with hyperparameters (`max_depth=5`, `min_samples_leaf=5`) to determine the inherent complexity of the task.
2.  **Random Forest**: An ensemble method with Mutual Information-based feature selection. It tests different numbers of top features (`k = [5, 10, 15, 16]`) to improve upon the baseline with better generalization and insight into feature importance.
3.  **1D Convolutional Neural Network**: A deep learning approach using an `Enhanced1DCNN` architecture with batch normalization and dropout. It employs early stopping on a validation set and retrains on the full training set for the optimal number of epochs to prevent overfitting.

## Training Methodology

*   **Validation Strategy**: Leave-One-Subject-Out (LOSO) Cross-Validation. All data from one subject is held out as the test set, and models are trained on the remaining subjects. This ensures the model generalizes to unseen individuals, which is crucial for physiological data.
*   **Data Scaling**: Features are standardized using `StandardScaler` (zero mean, unit variance) based on the training set for each LOSO fold.
*   **Class Handling**: Labels are encoded as `{'baseline': 0, 'stress': 1}`.
*   **Class Imbalance**: Addressed in the CNN via class-weighted loss, where weights are inversely proportional to class frequencies in the training set.

## Results

The models were evaluated using a Leave-One-Subject-Out (LOSO) cross-validation approach. Metrics reported are accuracy and F1-score for both the stress and baseline classes.

### 1. Baseline: Simple Decision Tree

| Window Size (s) | Accuracy | Stress F1 | Baseline F1 |
| :-------------: | :------: | :-------: | :---------: |
|       30        |  67.88%  |  66.82%   |   64.47%    |
|       45        |  68.10%  |  59.66%   |   70.13%    |
|       60        |  70.07%  |  64.74%   |   70.37%    |
|       75        |  67.90%  |  59.30%   |   70.25%    |
|       90        |  62.59%  |  54.90%   |   63.38%    |
|       120       |  69.62%  |  61.40%   |   68.95%    |

### 2. Random Forest Classifier (with Feature Selection)

The Random Forest model showed a significant improvement over the baseline. The best `k_features` for each window size was selected based on the highest accuracy.

| Window Size (s) | k_features | Accuracy | Stress F1 | Baseline F1 |
| :-------------: | :--------: | :------: | :-------: | :---------: |
|       30        |     15     |  73.85%  |  72.54%   |   75.05%    |
|       45        |     15     |  74.02%  |  69.54%   |   77.35%    |
|       60        |     16     |  76.50%  |  72.48%   |   79.50%    |
|       75        |     15     |  72.41%  |  64.29%   |   77.52%    |
|       90        |     15     |  71.42%  |  63.36%   |   76.58%    |
|       120       |     10     |  73.33%  |  64.86%   |   78.51%    |

### 3. 1D Convolutional Neural Network (1D CNN)

The CNN model performed competitively, demonstrating the ability of neural networks to learn patterns from the feature space. Training was optimized using early stopping.

| Window Size (s) | Accuracy | Stress F1 | Baseline F1 |
| :-------------: | :------: | :-------: | :---------: |
|       30        |  70.86%  |  69.58%   |   72.03%    |
|       45        |  75.25%  |  73.07%   |   77.10%    |
|       60        |  75.36%  |  70.95%   |   78.61%    |
|       75        |  70.34%  |  63.87%   |   74.85%    |
|       90        |  69.88%  |  63.21%   |   74.51%    |
|       120       |  61.03%  |  52.50%   |   66.96%    |

## Conclusion

*   **Best Performing Window Size**: A 60-second non-overlapping window consistently gave the highest performance across all models, indicating it captures the optimal temporal dynamics for stress discrimination in this dataset.
*   **Feature Selection**: For some larger window sizes (90s, 120s), using fewer features (`k=10`) gave better results, potentially reducing overfitting.
*   **Class Imbalance**: The consistently higher F1-score for the `baseline` class across all models suggests a slight class imbalance or that the `stress` condition is inherently more difficult to characterize.

## Usage

1.  Place the WESAD dataset in the `Dataset/WESAD/` directory.

2.  **Preprocess Data and Extract Features**:
    ```bash
    cd preprocessing
    python eda_features.py
    python ibi_features.py
    ```
    This will generate the necessary feature CSV files in the `preprocessing/data/` folder.

3.  **Train and Evaluate Models**:
    ```bash
    cd stress_classification
    
    # Baseline Model:
    python simple_decision_tree.py
    
    # Random Forest with Feature Selection:
    python random_forest.py
    
    # 1D CNN:
    python cnn.py
    ```

Each script will run LOSO CV for all window sizes and print the results.
    
Next: [Uncertainty Quantification](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/uncertainty_quantification/README.md)

