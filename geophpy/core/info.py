# -*- coding: utf-8 -*-
"""
   geophpy.core.info
   -----------------
   Provides a Mixin for retrieving metadata and derived coordinate
   information from a Survey object.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.
"""

import numpy as np

class InfoMixin:
    """
    Mixin for methods that retrieve metadata or derived coordinate
    information from a Survey object.
    """

    def get_xvect(self):
        """Returns the grid's x-coordinate vector."""
        if self.grid is None: raise ValueError("Grid does not exist.")
        nx = self.grid.z_image.shape[1]
        return np.linspace(self.info.x_min, self.info.x_max, nx)

    def get_yvect(self):
        """Returns the grid's y-coordinate vector."""
        if self.grid is None: raise ValueError("Grid does not exist.")
        ny = self.grid.z_image.shape[0]
        return np.linspace(self.info.y_min, self.info.y_max, ny)

    def get_xyvect(self):
        """Returns the grid's x and y-coordinate vectors."""
        return self.get_xvect(), self.get_yvect()

    def get_xygrid(self):
        """
        Returns the dataset's x and y-coordinate 2D grids.

        Returns
        -------
        X : np.ndarray
            2D array of the grid's x-coordinates.
        Y : np.ndarray
            2D array of the grid's y-coordinates.
        """
        x, y = self.get_xyvect()
        return np.meshgrid(x, y)

    def get_grid_extent(self):
        """
        Returns the spatial extent of the grid.

        Returns
        -------
        tuple
            A tuple containing (xmin, xmax, ymin, ymax).
        """
        if self.grid is None: raise ValueError("Grid does not exist.")
        return (self.info.x_min, self.info.x_max, 
                self.info.y_min, self.info.y_max)

    def get_grid_corners(self):
        """
        Returns the grid corner coordinates (BL, BR, TL, TR).
        """
        if self.grid is None: raise ValueError("Grid does not exist.")
        xmin, xmax, ymin, ymax = self.get_grid_extent()
        return np.array([[xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax]])

    def get_points_bounding_box(self):
        """
        Returns the bounding box of the ungridded point data.

        Returns
        -------
        np.ndarray
            An array of the corner coordinates: [[xmin, ymin], [xmax, ymin], 
            [xmin, ymax], [xmax, ymax]].
        """
        if self.points is None: raise ValueError("No point data available.")
        xmin, xmax = self.points.x.min(), self.points.x.max()
        ymin, ymax = self.points.y.min(), self.points.y.max()
        return np.array([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
    
    def grid_to_dict(self):
        """
        Exports grid and metadata to a dictionary for file writers.

        This helper method bundles all necessary grid information into a
        dictionary format that is compatible with various external writer
        functions, such as write_surfer6ascii.

        Returns
        -------
        dict
            A dictionary containing key grid parameters.

        Raises
        ------
        ValueError
            If the survey does not contain any grid data.
        """
        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No grid data available to export.")

        # Recalculate zmin and zmax from the actual grid values
        zmin, zmax = np.nanmin(self.grid.z_image), np.nanmax(self.grid.z_image)

        return {
            "values": self.grid.z_image,
            "nrow": self.grid.z_image.shape[0],
            "ncol": self.grid.z_image.shape[1],
            "xmin": self.info.x_min,
            "xmax": self.info.x_max,
            "ymin": self.info.y_min,
            "ymax": self.info.y_max,
            "xsize": self.info.x_gridding_delta,
            "ysize": self.info.y_gridding_delta,
            "zmin": zmin,
            "zmax": zmax,
        }
