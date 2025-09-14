"""
   geophpy.core.stats
   ------------------
   Provides Mixins for statistical and informational methods.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.
"""

import numpy as np
from . import utils

class StatsPointsMixin:
    """
    Mixin for methods that retrieve statistics from ungridded (point) data.
    """
    
    def get_point_stats(self, verbose: bool = False) -> dict:
        """
        Calculates statistics on the ungridded point data values.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the calculated statistics. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing 'min', 'max', 'mean', and 'stdev'.
        
        Raises
        ------
        ValueError
            If no point data is available.
        """

        if self.points is None or self.points.values.size == 0:
            raise ValueError("No point data available to calculate statistics.")

        values = self.points.values
        stats = {
            'min': np.nanmin(values),
            'max': np.nanmax(values),
            'mean': np.nanmean(values),
            'stdev': np.nanstd(values)
        }

        if verbose:
            print(f"Point Data Statistics for '{self.name}':")
            for key, value in stats.items():
                print(f"  - {key.capitalize()}: {value:.2f}")

        return stats
    
    def get_median_xstep(self, prec: int = 2):
        """
        Calculates the median step size for the ungridded x-coordinates.

        This is a high-level method that calls a low-level utility
        function to perform the calculation.

        Parameters
        ----------
        prec : int, optional
            The decimal precision for the returned value. Defaults to 2.

        Returns
        -------
        float
            The calculated median step size.
        """
        if self.points is None or self.points.x.size == 0:
            raise ValueError("No point data available to calculate median x-step.")
        
        # The method calls the low-level "tool" from utils.py
        return utils.get_median_step(self.points.x, prec=prec)


    def get_median_ystep(self, prec: int = 2):
        """Calculates the median step size for the ungridded y-coordinates.

        This is a high-level method that calls a low-level utility
        function to perform the calculation.

        Parameters
        ----------
        prec : int, optional
            The decimal precision for the returned value. Defaults to 2.

        Returns
        -------
        float
            The calculated median step size.
        """

        if self.points is None or self.points.y.size == 0:
            raise ValueError("No point data available to calculate median y-step.")

        return utils.get_median_step(self.points.y, prec=prec)

    def get_points_centroid(self) -> tuple:
        """
        Calculates the centroid of the ungridded point data.

        The centroid is calculated as the mean of the x and y coordinates.

        Returns
        -------
        tuple
            A tuple containing the (x_centroid, y_centroid).

        Raises
        ------
        ValueError
            If no point data is available.
        """
        if self.points is None or self.points.x.size == 0:
            raise ValueError("No point data available to calculate centroid.")
        
        x_centroid = np.nanmean(self.points.x)
        y_centroid = np.nanmean(self.points.y)
        
        return (x_centroid, y_centroid)


class StatsGridMixin:
    """
    Mixin for methods that retrieve statistics from gridded (image-like) data.
    """

    def get_grid_stats(self, verbose: bool = False) -> dict:
        """
        Calculates statistics on the gridded data values.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the calculated statistics. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing 'min', 'max', 'mean', and 'stdev'.
        
        Raises
        ------
        ValueError
            If no gridded data is available.
        """

        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available. Please run .interpolate() first.")

        values = self.grid.z_image
        stats = {
            'min': np.nanmin(values),
            'max': np.nanmax(values),
            'mean': np.nanmean(values),
            'stdev': np.nanstd(values)
        }

        if verbose:
            print(f"Grid Data Statistics for '{self.name}':")
            for key, value in stats.items():
                print(f"  - {key.capitalize()}: {value:.2f}")
        
        return stats

    def get_grid_centroid(self) -> tuple:
        """
        Calculates the geometric center of the gridded data's extent.

        Returns
        -------
        tuple
            A tuple containing the (x_center, y_center) of the grid.

        Raises
        ------
        ValueError
            If the data has not been gridded yet.
        """
        if self.grid is None or self.info.x_min is None:
            raise ValueError("Grid does not exist or has no coordinate information.")
            
        x_center = self.info.x_min + (self.info.x_max - self.info.x_min) / 2.0
        y_center = self.info.y_min + (self.info.y_max - self.info.y_min) / 2.0
        
        return (x_center, y_center)
