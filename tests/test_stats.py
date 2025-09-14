import numpy as np
import pytest

def test_get_point_stats(survey_from_file):
    """
    Tests that get_point_stats returns a valid dictionary of statistics.
    """
    stats = survey_from_file.get_point_stats()
    assert isinstance(stats, dict)
    assert 'mean' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'stdev' in stats
    assert stats['max'] >= stats['min']

def test_get_grid_stats(gridded_survey_from_file):
    """
    Tests that get_grid_stats returns valid statistics for the grid.
    """
    stats = gridded_survey_from_file.get_grid_stats()
    assert isinstance(stats, dict)
    assert 'mean' in stats
    assert np.isfinite(stats['mean'])

def test_get_median_xstep(survey_from_file):
    """
    Tests that get_median_xstep returns a positive float.
    """
    x_step = survey_from_file.get_median_xstep()
    assert isinstance(x_step, float)
    assert x_step > 0

def test_get_grid_centroid(gridded_survey_from_file):
    """
    Tests that get_grid_centroid returns a valid (x, y) tuple.
    """
    centroid = gridded_survey_from_file.get_grid_centroid()
    assert isinstance(centroid, tuple)
    assert len(centroid) == 2
    assert np.isfinite(centroid[0]) and np.isfinite(centroid[1])