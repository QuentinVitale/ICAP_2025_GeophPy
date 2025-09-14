import numpy as np
import pytest

# --- Tests for ArithmeticPointsMixin ---

def test_add_to_points(survey_from_file):
    """
    Tests the add_to_points method.
    """
    original_values = survey_from_file.points.values.copy()
    
    # Test with inplace=False (default behavior)
    new_survey = survey_from_file.add_to_points(10)
    assert new_survey is not survey_from_file, "Should return a new object"
    assert np.array_equal(survey_from_file.points.values, original_values), "Original should be unchanged"
    assert np.allclose(new_survey.points.values, original_values + 10), "New object should be modified"

    # Test with inplace=True
    result = survey_from_file.add_to_points(10, inplace=True)
    assert result is None, "Inplace operation should return None"
    assert np.allclose(survey_from_file.points.values, original_values + 10), "Original should be modified"

def test_subtract_from_points(survey_from_file):
    """Tests the subtract_from_points method."""
    original_values = survey_from_file.points.values.copy()
    new_survey = survey_from_file.subtract_from_points(10)
    assert np.allclose(new_survey.points.values, original_values - 10)

def test_multiply_points_by(survey_from_file):
    """Tests the multiply_points_by method."""
    original_values = survey_from_file.points.values.copy()
    new_survey = survey_from_file.multiply_points_by(2)
    assert np.allclose(new_survey.points.values, original_values * 2)

def test_divide_points_by(survey_from_file):
    """Tests the divide_points_by method."""
    original_values = survey_from_file.points.values.copy()
    new_survey = survey_from_file.divide_points_by(2)
    assert np.allclose(new_survey.points.values, original_values / 2)

def test_set_points_mean(survey_from_file):
    """Tests the set_points_mean method."""
    new_survey = survey_from_file.set_points_mean(0)
    assert np.isclose(np.nanmean(new_survey.points.values), 0.0)

def test_set_points_median(survey_from_file):
    """Tests the set_points_median method."""
    new_survey = survey_from_file.set_points_median(150)
    assert np.isclose(np.nanmedian(new_survey.points.values), 150.0)


# --- Tests for ArithmeticGridMixin ---

def test_add_to_grid(gridded_survey_from_file):
    """Tests the add_to_grid method."""
    original_mean = np.nanmean(gridded_survey_from_file.grid.z_image)
    
    # Test with inplace=False
    new_survey = gridded_survey_from_file.add_to_grid(10)
    assert new_survey is not gridded_survey_from_file, "Should return a new object"
    assert np.isclose(np.nanmean(gridded_survey_from_file.grid.z_image), original_mean), "Original should be unchanged"
    assert np.isclose(np.nanmean(new_survey.grid.z_image), original_mean + 10), "New object should be modified"

    # Test with inplace=True
    result = gridded_survey_from_file.add_to_grid(10, inplace=True)
    assert result is None, "Inplace operation should return None"
    assert np.isclose(np.nanmean(gridded_survey_from_file.grid.z_image), original_mean + 10), "Original should be modified"

def test_subtract_from_grid(gridded_survey_from_file):
    """Tests the subtract_from_grid method."""
    original_mean = np.nanmean(gridded_survey_from_file.grid.z_image)
    new_survey = gridded_survey_from_file.subtract_from_grid(10)
    assert np.isclose(np.nanmean(new_survey.grid.z_image), original_mean - 10)

def test_multiply_grid_by(gridded_survey_from_file):
    """Tests the multiply_grid_by method."""
    original_mean = np.nanmean(gridded_survey_from_file.grid.z_image)
    new_survey = gridded_survey_from_file.multiply_grid_by(2)
    assert np.isclose(np.nanmean(new_survey.grid.z_image), original_mean * 2)

def test_divide_grid_by(gridded_survey_from_file):
    """Tests the divide_grid_by method."""
    original_mean = np.nanmean(gridded_survey_from_file.grid.z_image)
    new_survey = gridded_survey_from_file.divide_grid_by(2)
    assert np.isclose(np.nanmean(new_survey.grid.z_image), original_mean / 2)

def test_set_grid_mean(gridded_survey_from_file):
    """Tests the set_grid_mean method."""
    new_survey = gridded_survey_from_file.set_grid_mean(50)
    assert np.isclose(np.nanmean(new_survey.grid.z_image), 50.0)

def test_set_grid_median(gridded_survey_from_file):
    """Tests the set_grid_median method."""
    new_survey = gridded_survey_from_file.set_grid_median(250)
    assert np.isclose(np.nanmedian(new_survey.grid.z_image), 250.0)