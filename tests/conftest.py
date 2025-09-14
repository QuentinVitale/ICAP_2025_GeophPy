import pytest
import numpy as np
import os
from geophpy import Survey

@pytest.fixture(scope="session")
def data_path():
    """A pytest fixture to provide the absolute path to the test data directory."""
    
    return os.path.join(os.path.dirname(__file__), '..', 'data')

@pytest.fixture
def survey_from_file(data_path):
    """
    A fixture that loads a Survey object from the real example ASCII file.
    """
    filepath = os.path.join(data_path, 'Mag_ex1.dat')
    if not os.path.exists(filepath):
        pytest.fail(f"Test data file not found: {filepath}")

    survey = Survey.from_file(
        filepath,
        delimiter=',',
        x_colnum=0,
        y_colnum=1,
        z_colnum=4
    )
    # This fixture also needs track data for some tests.
    # We can add a placeholder or a call to an estimation method here.
    # For now, let's assume it needs to be estimated.
    # survey.estimate_tracks_from_coords() # (Once this method is implemented)
    return survey

@pytest.fixture
def gridded_survey_from_file(survey_from_file):
    """
    A fixture that creates a gridded Survey object from the real example data.
    """
    # Use the survey loaded from the file and interpolate it
    survey_from_file.interpolate(x_step=0.15, y_step=0.15)
    return survey_from_file