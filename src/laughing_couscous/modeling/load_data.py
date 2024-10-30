"""Data module."""

from typing import List

from numpy import ndarray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data() -> List[ndarray]:
    """Load iris dataset.

    This function loads the iris dataset and splits it into training and testing sets.

    Returns:
        List[ndarray]: A list containing the following arrays:
            - X_train: Training features.
            - X_test: Testing features.
            - y_train: Training target.
            - y_test: Testing target.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    return train_test_split(X, y, test_size=0.2, random_state=42)
