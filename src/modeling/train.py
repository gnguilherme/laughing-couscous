"""Train a DecisionTreeClassifier model."""

from numpy import ndarray
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


def train(X_train: ndarray, X_test: ndarray, y_train: ndarray, y_test: ndarray) -> None:
    """Train a DecisionTreeClassifier model.

    Args:
        X_train (ndarray): Training features.
        X_test (ndarray): Testing features.
        y_train (ndarray): Training target.
        y_test (ndarray): Testing target

    """

    model = DecisionTreeClassifier(random_state=42)
    param_distributions = {
        "max_depth": [3, 5, 7, 9, 11],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "splitter": ["best", "random"],
    }
    search = RandomizedSearchCV(model, param_distributions, n_iter=10)
    search.fit(X_train, y_train)

    score = search.score(X_test, y_test)
    print(f"Test score: {score}")

    # this model should be save with mlflow
