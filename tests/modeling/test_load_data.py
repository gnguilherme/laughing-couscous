"""Unit tests for the load_data module."""

from unittest.mock import patch

from src.modeling.load_data import load_data


@patch("src.modeling.load_data.load_iris")
@patch("src.modeling.load_data.train_test_split")
def test_load_data(mock_train_test_split, mock_load_iris):
    """Test load_data function."""
    # Given
    mock_load_iris.return_value.data = "data"
    mock_load_iris.return_value.target = "target"
    mock_train_test_split.return_value = ["X_train", "X_test", "y_train", "y_test"]

    # When
    result = load_data()

    # Then
    assert result == ["X_train", "X_test", "y_train", "y_test"]
    mock_load_iris.assert_called_once()
    mock_train_test_split.assert_called_once()
