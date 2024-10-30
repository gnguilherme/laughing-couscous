"""Inference module for the model."""

from numpy import ndarray
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier


def predict(X: DataFrame) -> ndarray:
    """Make a prediction with the model.

    Args:
        X (DataFrame): The input data.

    Returns:
        ndarray: The model predictions.
    """
    model = DecisionTreeClassifier()  # this should be loaded from mlflow
    return model.predict(X)
