# -*- coding: utf-8 -*-
"""
   geophpy.core.survey
   -------------------

   Defines the main Survey object for handling geophysical data.

   :copyright: Copyright 2014-2025 L. Darras, Q. Vitale and contributors, see AUTHORS.
   :license: GNU GPL v3.

"""

from __future__ import absolute_import
#from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from .datastructures import PointData, GridData, Info, GeoRefSystem
from . import spatial

# Import necessary dependencies from other modules and libraries
import os
import numpy as np
import json

# Import the Mixins that provide the functionality
from .io import IOMixin
from .info import InfoMixin
from .stats import StatsPointsMixin, StatsGridMixin
from .spatial import SpatialPointsMixin, SpatialGridMixin
from .arithmetic import ArithmeticPointsMixin, ArithmeticGridMixin
from ..visualization.plot import PlottingMixin
from .processing import ProcessingPointsMixin, ProcessingGridMixin
from ..mag.processing import MagneticGridMixin
from ..mag.plot import MagneticPlottingMixin

import geophpy.core.processing as proc
import geophpy.visualization.plot as plot


# --- Main Survey Class ---
class Survey(
    IOMixin,
    InfoMixin,
    StatsPointsMixin,
    StatsGridMixin,
    SpatialPointsMixin,
    SpatialGridMixin,
    ArithmeticPointsMixin,
    ArithmeticGridMixin,
    PlottingMixin,
    ProcessingPointsMixin,
    ProcessingGridMixin,
    MagneticGridMixin,
    MagneticPlottingMixin
    ):
    """
    Represents a complete geophysical survey.

    This class is the main entry point for a user. It acts as a container for
    all data (gridded and ungridded), metadata, and georeferencing information.
    It also exposes a rich set of methods for I/O, processing, and operations,
    which are provided by inheriting from various Mixin classes.

    Attributes
    ----------
    name : str
        The name of the survey, e.g., 'Mag_ex1'.
    info : Info
        A dataclass containing metadata and grid information.
    data : Data
        A dataclass containing the actual data arrays (NumPy arrays).
    georef : GeoRefSystem
        A dataclass for storing all georeferencing information.
    history : list of dict
        A log of all processing steps applied to the survey instance.

    """
    _instance_counter = 0

    def __init__(self, name: str = 'New Survey'):
        """
        Initializes the Survey object.

        Parameters
        ----------
        name : str, optional
            The name for the survey. If the default name 'New Survey' is
            used, a unique counter will be appended to it (e.g.,
            'New Survey 1', 'New Survey 2').

        """
        # --- Handle automatic naming for default instances ---
        if name == 'New Survey':
            Survey._instance_counter += 1
            self.name = f"{name} {Survey._instance_counter}"
        else:
            self.name = name
            
        self.info = Info()
        self.points = PointData()
        self.grid = None
        self.georef = GeoRefSystem()
        self.history = []
        self.log_message(f"Survey object '{self.name}' created")

    def interpolate(self, method='linear', **kwargs):
        """
        Interpolates point data to create and store grid data.
        """
        if self.points is None or self.points.values.size == 0:
            raise ValueError("No point data to interpolate.")

        # Call the specialized function from the spatial module
        gridded_data_array, updated_info_object = spatial.create_grid_from_points(
            self.points,
            self.info,
            method=method,
            **kwargs
        )
        
        # Check if the interpolation was successful before assigning
        if gridded_data_array and updated_info_object:
            self.grid = gridded_data_array
            self.info = updated_info_object
            self.log_step('interpolate', {'method': method, **kwargs})
        
        else:
            print("Warning: Interpolation failed. The survey's grid has not been updated.")

        
    def log_message(self, message: str):
        """
        Adds a simple text message to the processing history.

        Parameters
        ----------
        message : str
            The informational message to be logged.

        """
        self.history.append({'step': 'message', 'parameters': {'text': message}})

    def log_step(self, step_name: str, parameters: Dict[str, Any]):
        """
        Adds a structured processing step to the history log.

        This method is called by processing functions to record the operation
        and its parameters, creating a reproducible "recipe".

        Parameters
        ----------
        step_name : str
            The name of the method being applied (e.g., 'peakfilt').
        parameters : dict
            A dictionary of the parameters used in the method call.

        """
        self.history.append({'step': step_name, 'parameters': parameters})

    def export_history(self, filename: str) -> bool:
        """
        Exports the processing history to a JSON file.

        This creates a reusable "recipe" that can be applied to other
        datasets for batch processing.

        Parameters
        ----------
        filename : str
            The path to the output JSON file.

        Returns
        -------
        bool
            ``True`` if the file was saved successfully, ``False`` otherwise.

        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.history, f, indent=4)
            print(f"Processing history successfully saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving history to {filename}: {e}")
            return False



# Batch processing Fun to be tested first
def apply_processing_sequence_to_file(data_filepath, sequence_filepath):
    """
    Loads a data file, applies a processing sequence, and saves the result.

    This function automates a processing workflow by loading a sequence of
    operations from a JSON "recipe" file and applying them sequentially to a
    raw data file. It is designed for batch processing multiple datasets with
    the same parameters.

    Parameters
    ----------
    data_filepath : str
        The full path to the raw data file that needs to be processed.
    sequence_filepath : str
        The full path to the JSON file containing the processing sequence.
        This file should be an array of objects, where each object has a
        "step" (the method name) and "parameters" (a dictionary of arguments).

    Returns
    -------
    bool
        ``True`` if the processing was completed and the file was saved
        successfully, ``False`` otherwise.

    Raises
    ------
    FileNotFoundError
        If either `data_filepath` or `sequence_filepath` does not exist.
    AttributeError
        If a method specified in the sequence file does not exist in the
        Survey class.

    See Also
    --------
    Survey.export_history : The method used to generate the sequence file.

    Examples
    --------
    >>> # Assume 'my_recipe.json' contains a processing sequence
    >>> # and 'raw_data.dat' is the file to process.
    >>> apply_processing_sequence_to_file('data/raw_data.dat', 'recipes/my_recipe.json')
    Applying 5 steps to 'raw_data.dat'...
      -> Applying step: peakfilt
      -> Applying step: festoonfilt
      -> Applying step: destripecon
    Saving processed data to: data/raw_data_processed.tiff
    --- Processing complete for this file. ---
    True

    """
    # Load the processing sequence
    print(f"Loading sequence from: {sequence_filepath}")
    with open(sequence_filepath, 'r') as f:
        sequence = json.load(f)

    # Load the raw data file
    print(f"Loading data from: {data_filepath}")
    success, survey = Survey.from_file(
        [data_filepath],
        fileformat='ascii',
        delimiter=',',
        x_colnum=1,
        y_colnum=2,
        z_colnum=5
    )
    if not success:
        print(f"Could not load {data_filepath}. Aborting.")
        return False

    # Apply each step from the sequence
    print(f"Applying {len(sequence)} steps to '{os.path.basename(data_filepath)}'...")
    for step in sequence:
        if step.get('step') != 'message':
            method_name = step['step']
            parameters = step['parameters']
            
            print(f"  -> Applying step: {method_name}")
            method_to_call = getattr(survey, method_name)
            method_to_call(**parameters)

    # Save the result
    base, ext = os.path.splitext(data_filepath)
    output_filename = f"{base}_processed.tiff"
    print(f"Saving processed data to: {output_filename}")
    survey.to_raster('2D-SURFACE', 'gray', output_filename)

    print("--- Processing complete for this file. ---\n")
    return True