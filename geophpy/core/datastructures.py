# -*- coding: utf-8 -*-
"""
   geophpy.core.datastructures
   ---------------------------

   Defines the core data structures for the Survey object using dataclasses.

   :copyright: Copyright 2014-2025 L. Darras, P. Marty, Q. Vitale and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np

# --- Data Structure for Points ---

@dataclass
class PointData:
    """
    The base dataclass for all ungridded (scatter) survey data.

    This class holds the universal attributes required by all geophysical
    methods. Specialized methods should create subclasses that inherit from
    this to add method-specific fields (e.g., elevation, timestamp).

    Attributes
    ----------
    x : np.ndarray
        Array of local x-coordinates for each data point.
    y : np.ndarray
        Array of local y-coordinates for each data point.
    values : np.ndarray
        Array of the primary measurement values for each data point.

    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    east: np.ndarray = field(default_factory=lambda: np.array([]))
    north: np.ndarray = field(default_factory=lambda: np.array([]))
    long: np.ndarray = field(default_factory=lambda: np.array([]))
    lat: np.ndarray = field(default_factory=lambda: np.array([]))
    track: np.ndarray = field(default_factory=lambda: np.array([]))


# --- Data Structure for Grids ---

@dataclass
class GridData:
    """
    A dataclass to store gridded (image-like) data arrays.

    Attributes
    ----------
    z_image : np.ndarray, optional
        2D array of the gridded data values.
    easting_image : np.ndarray, optional
        2D array of the gridded easting coordinates for each cell.
    northing_image : np.ndarray, optional
        2D array of the gridded northing coordinates for each cell.

    """
    z_image: Optional[np.ndarray] = None
    easting_image: Optional[np.ndarray] = None
    northing_image: Optional[np.ndarray] = None


# --- Other Core Dataclasses ---

@dataclass
class Info:
    """
    A dataclass to store grid metadata and plotting settings.

    Attributes
    ----------
    x_min : float, optional
        Minimum x-value of the grid.
    x_max : float, optional
        Maximum x-value of the grid.
    y_min : float, optional
        Minimum y-value of the grid.
    y_max : float, optional
        Maximum y-value of the grid.
    x_gridding_delta : float, optional
        Grid step size in the x-direction.
    y_gridding_delta : float, optional
        Grid step size in the y-direction.
    gridding_interpolation : str, optional
        Interpolation method used for gridding.
    plottype : str
        Default plot type for the grid.
    cmapname : str
        Default colormap name for the grid plot.

    """
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    x_gridding_delta: Optional[float] = None
    y_gridding_delta: Optional[float] = None
    gridding_interpolation: Optional[str] = None
    plottype: str = '2D-SURFACE'
    cmapname: str = 'Greys'


@dataclass
class GeoRefSystem:
    """
    A dataclass for storing all georeferencing system information.

    Attributes
    ----------
    active : bool
        Status of whether the dataset is georeferenced.
    refsystem : str, optional
        The name of the reference system (e.g., 'UTM', 'WGS84').
    utm_zoneletter : str, optional
        The UTM zone letter.
    utm_zonenumber : int, optional
        The UTM zone number.
    points_list : list, optional
        A list of ground control points (GCPs).

    """
    active: bool = False
    refsystem: Optional[str] = None
    utm_zoneletter: Optional[str] = None
    utm_zonenumber: Optional[int] = None
    points_list: List = field(default_factory=list)
