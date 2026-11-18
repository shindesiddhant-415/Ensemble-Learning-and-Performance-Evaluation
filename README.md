# Ensemble Learning and Performance Evaluation

**Practical No.:** 5  
**Practical Title:** Ensemble Learning and Performance Evaluation

## Aim
Implement and compare three ensemble learning methods—hard voting, stacking, and AdaBoost—with individual base classifiers.

## Dataset
This practical uses the Digits dataset provided by `scikit-learn`. It contains 1,797 samples, 64 numeric pixel features per sample, and 10 digit classes (0–9). Each sample represents an 8×8 handwritten digit image.

## Models
### Base classifiers
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

### Ensemble classifiers
- Hard Voting
- Stacking with Logistic Regression as the meta-classifier
- AdaBoost with depth-3 decision trees as weak learners

A depth-3 decision tree is also evaluated separately to compare the weak learner with AdaBoost.

## Evaluation
The dataset is divided into training and testing sets using an 80:20 stratified split (`random_state=42`). Models are compared using accuracy, macro precision, macro recall, macro F1-score, and 5-fold stratified cross-validation accuracy (mean and standard deviation).

## Setup and execution
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the practical:

   ```bash
   python main.py
   ```

## Generated outputs
After a successful run, the `outputs/` folder contains:
- `model_comparison.csv` — model-wise evaluation metrics.
- `ensemble_comparison.png` — accuracy and F1-score comparison chart.

Commit these generated output files to GitHub after running the program. Use your actual results in the assignment report; do not copy sample results from another student's PDF.

## Repository structure

```text
ensemble-learning-assignment/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
    ├── model_comparison.csv
    └── ensemble_comparison.png
```

## Academic integrity
Run the code yourself and use the results produced by your own execution. Add your own screenshots from the terminal or IDE if your faculty requires execution evidence.
