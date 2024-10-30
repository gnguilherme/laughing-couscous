"""Inference module for the model."""

from numpy import ndarray
from pandas import DataFrame


def predict(X: DataFrame) -> ndarray:
    """Make a prediction with the model.

    Args:
        X (DataFrame): The input data.

    Returns:
        ndarray: The model predictions.
    """

    return model.predict(X)  # should be replaced with the actual model
