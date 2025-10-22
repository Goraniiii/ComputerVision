# CIFAR-10 KNN Classification

## Project Overview
This project implements and evaluates a K-Nearest Neighbor (KNN) classifier on the CIFAR-10 dataset.  
The goal is to explore different data splitting strategies and evaluate the performance of the custom KNN classifier with various hyperparameters.

---

## Experiments

1. **Train/Test Split**  
   - Simple train/test split to evaluate baseline KNN performance.  
   - Accuracy, Precision, Recall, and F1-score computed on the test set.

2. **Train/Validation/Test Split**  
   - Train set used to train the KNN classifier.  
   - Validation set used to select optimal hyperparameters (`k` and distance metric).  
   - Test set used for final evaluation after hyperparameter selection.

3. **5-Fold Cross-Validation**  
   - 5-fold CV performed on the training set to evaluate the stability of hyperparameters.  
   - Fold-wise Accuracy, Precision, Recall, and F1-score calculated.  
   - Results visualized with error bars to show variability across folds.

---

## Code Organization

- **practice_knn.ipynb**  
  - Main notebook for experiments and visualization.  

- **kNearestNeighbor.py**  
  - Custom KNN classifier implementation using sklearn.  

- **util.py**  
  - Utility functions for data loading, preprocessing, and other helper functions.

---

## Notes
- All experiments use the full CIFAR-10 dataset (50,000 training samples).  
- Distance metrics tested: Euclidean and Manhattan.  
- Hyperparameter `k` explored in a range.
