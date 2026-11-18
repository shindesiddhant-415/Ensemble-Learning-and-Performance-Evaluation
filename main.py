"""
Practical 5: Ensemble Learning and Performance Evaluation
Dataset: Scikit-learn Digits dataset
Run: python main.py
Outputs are saved in the outputs/ folder.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_base_models():
    """Return fresh instances of the five base classifiers."""
    return [
        (
            "Logistic Regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ),
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        (
            "KNN",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        ),
        ("Naive Bayes", GaussianNB()),
        (
            "SVM",
            make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            ),
        ),
    ]


def create_models():
    """Create base and ensemble models."""
    base_estimators = create_base_models()

    # scikit-learn uses 'estimator' in newer versions and 'base_estimator'
    # in older versions. This keeps the practical compatible with both.
    weak_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    try:
        adaboost = AdaBoostClassifier(
            estimator=weak_tree,
            n_estimators=100,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        )
    except TypeError:
        adaboost = AdaBoostClassifier(
            base_estimator=weak_tree,
            n_estimators=100,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        )

    models = {
        "Logistic Regression": dict(create_base_models())["Logistic Regression"],
        "Decision Tree": dict(create_base_models())["Decision Tree"],
        "Decision Tree (depth=3)": DecisionTreeClassifier(
            max_depth=3, random_state=RANDOM_STATE
        ),
        "KNN": dict(create_base_models())["KNN"],
        "Naive Bayes": dict(create_base_models())["Naive Bayes"],
        "SVM": dict(create_base_models())["SVM"],
        "Voting (Hard)": VotingClassifier(
            estimators=base_estimators, voting="hard"
        ),
        "Stacking": StackingClassifier(
            estimators=create_base_models(),
            final_estimator=LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE
            ),
            cv=5,
        ),
        "AdaBoost": adaboost,
    }
    return models


def main():
    print("PRACTICAL 5: ENSEMBLE LEARNING AND PERFORMANCE EVALUATION")
    print(f"scikit-learn version: {sklearn.__version__}")

    # Load the built-in handwritten digits dataset.
    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"\nDataset samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = create_models()
    cv_strategy = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    results = []
    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        cv_scores = cross_val_score(
            model, X, y, cv=cv_strategy, scoring="accuracy"
        )

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "F1-Score": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "CV Acc (mean)": cv_scores.mean(),
                "CV Acc (std)": cv_scores.std(),
            }
        )

    results_df = pd.DataFrame(results)
    numeric_columns = results_df.select_dtypes(include="number").columns
    results_df[numeric_columns] = results_df[numeric_columns].round(4)

    print("\nBASE VS ENSEMBLE MODEL COMPARISON")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_accuracy = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
    best_f1 = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
    best_cv = results_df.loc[results_df["CV Acc (mean)"].idxmax(), "Model"]

    print("\nPERFORMANCE SUMMARY")
    print(f"Highest Accuracy: {best_accuracy}")
    print(f"Highest F1-Score: {best_f1}")
    print(f"Highest CV Accuracy: {best_cv}")
    print("\nProgram executed successfully.")

    # Save a comparison chart.
    ax = results_df.set_index("Model")[["Accuracy", "F1-Score"]].plot(
        kind="bar", figsize=(12, 5)
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Base Models vs Ensemble Models - Digits Dataset")
    ax.set_xlabel("Machine Learning Model")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ensemble_comparison.png", dpi=200)
    plt.close()

    print(f"\nFiles saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
