import numpy as np
import pytest

def test_get_grid_extent(gridded_survey_from_file):
    """
    Tests that get_grid_extent returns a valid tuple of coordinates.
    """
    extent = gridded_survey_from_file.get_grid_extent()
    assert isinstance(extent, tuple)
    assert len(extent) == 4
    xmin, xmax, ymin, ymax = extent
    assert xmax > xmin
    assert ymax > ymin

def test_get_points_bounding_box(survey_from_file):
    """
    Tests that get_points_bounding_box returns a valid numpy array.
    """
    bbox = survey_from_file.get_points_bounding_box()
    assert isinstance(bbox, np.ndarray)
    assert bbox.shape == (4, 2)

def test_get_xyvect(gridded_survey_from_file):
    """
    Tests that get_xyvect returns two 1D numpy arrays.
    """
    x_vect, y_vect = gridded_survey_from_file.get_xyvect()
    assert isinstance(x_vect, np.ndarray)
    assert isinstance(y_vect, np.ndarray)
    assert x_vect.ndim == 1
    assert y_vect.ndim == 1
    assert x_vect.size == gridded_survey_from_file.grid.z_image.shape[1]
    assert y_vect.size == gridded_survey_from_file.grid.z_image.shape[0]