import numpy as np
import pytest

def test_rotate_grid(gridded_survey_from_file):
    """
    Tests the rotate_grid method.
    """
    original_shape = gridded_survey_from_file.grid.z_image.shape
    
    # Test inplace=False (default)
    rotated_survey = gridded_survey_from_file.rotate_grid(90)
    assert rotated_survey is not gridded_survey_from_file
    assert rotated_survey.grid.z_image.shape == (original_shape[1], original_shape[0])

    # Test inplace=True
    gridded_survey_from_file.rotate_grid(90, inplace=True)
    assert gridded_survey_from_file.grid.z_image.shape == (original_shape[1], original_shape[0])

def test_crop_points(survey_from_file):
    """
    Tests the crop_points method.
    """
    original_size = survey_from_file.points.x.size
    xmin, xmax = np.percentile(survey_from_file.points.x, [10, 90])
    
    cropped_survey = survey_from_file.crop_points(xmin=xmin, xmax=xmax)
    
    assert cropped_survey.points.x.size < original_size
    assert cropped_survey.points.x.min() >= xmin
    assert cropped_survey.points.x.max() <= xmax

def test_translate_grid(gridded_survey_from_file):
    """
    Tests the translate_grid method.
    """
    original_xmin = gridded_survey_from_file.info.x_min
    
    translated_survey = gridded_survey_from_file.translate_grid(x_shift=100)
    
    assert np.isclose(translated_survey.info.x_min, original_xmin + 100)