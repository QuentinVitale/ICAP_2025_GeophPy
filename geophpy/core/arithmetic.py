# -*- coding: utf-8 -*-
"""
   geophpy.core.arithmetic
   -----------------------

   Provides Mixins for arithmetic operations on Survey objects, for both
   point and gridded data.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.
"""

import numpy as np
import copy

class ArithmeticPointsMixin:
    """
    Mixin for arithmetic operations that work on ungridded (point) data.
    """
    def add_to_points(self, value: float, inplace: bool = False):
        """
        Adds a constant value to the ungridded data points.

        Parameters
        ----------
        value : float
            The value to add to each point in `self.points.values`.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the modified data if `inplace` is False.
        """

        if self.points is None: raise ValueError("No point data available.")
        target = self if inplace else copy.deepcopy(self)
        target.points.values = target.points.values + value
        target.log_step('add', {'value': value, 'target': 'points'})
        if not inplace: return target

    def subtract_from_points(self, value: float, inplace: bool = False):
        """
        Subtracts a constant value from the ungridded data points.

        Parameters
        ----------
        value : float
            The value to subtract from each point in `self.points.values`.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the modified data if `inplace` is False.
        """
        if self.points is None: raise ValueError("No point data available.")
        target = self if inplace else copy.deepcopy(self)
        target.points.values = target.points.values - value
        target.log_step('subtract', {'value': value, 'target': 'points'})
        if not inplace: return target

    def multiply_points_by(self, value: float, inplace: bool = False):
        """
        Multiplies the ungridded data points by a constant value.

        Parameters
        ----------
        value : float
            The value to multiply each point in `self.points.values` by.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the modified data if `inplace` is False.
        """
        if self.points is None: raise ValueError("No point data available.")
        target = self if inplace else copy.deepcopy(self)
        target.points.values = target.points.values* value
        target.log_step('multiply', {'value': value, 'target': 'points'})
        if not inplace: return target

    def divide_points_by(self, value: float, inplace: bool = False):
        """
        Divides the ungridded data points by a constant value.

        Parameters
        ----------
        value : float
            The value to divide each point in `self.points.values` by.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the modified data if `inplace` is False.
        """
        if self.points is None: raise ValueError("No point data available.")
        target = self if inplace else copy.deepcopy(self)
        target.points.values = target.points.values / value
        target.log_step('divide', {'value': value, 'target': 'points'})
        if not inplace: return target

    def set_points_mean(self, value: float, inplace: bool = False):
        """
        Shifts the point data so that its mean is equal to a target value.

        Parameters
        ----------
        value : float
            The target mean for the point data values.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the modified data if `inplace` is False.
        """
        if self.points is None: raise ValueError("No point data available.")
        target = self if inplace else copy.deepcopy(self)
        current_mean = np.nanmean(target.points.values)
        shift = value - current_mean
        target.points.values = target.points.values + shift
        target.log_step('set_mean', {'value': value, 'target': 'points'})
        if not inplace: return target

    def set_points_median(self, value: float, inplace: bool = False):
        """
        Shifts the point data so that its median is equal to a target value.

        Parameters
        ----------
        value : float
            The target median for the point data values.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the modified data if `inplace` is False.
        """
        if self.points is None: raise ValueError("No point data available.")
        target = self if inplace else copy.deepcopy(self)
        current_median = np.nanmedian(target.points.values)
        shift = value - current_median
        target.points.values = target.points.values + shift
        target.log_step('set_median', {'value': value, 'target': 'points'})
        if not inplace: return target


class ArithmeticGridMixin:
    """
    Mixin for arithmetic operations that work on gridded (image-like) data.
    """
    def add_to_grid(self, value: float, inplace: bool = False):
        """Adds a constant value to the gridded data."""
        if self.grid is None: raise ValueError("No gridded data available.")
        target = self if inplace else copy.deepcopy(self)
        target.grid.z_image = target.grid.z_image + value
        target.log_step('add', {'value': value, 'target': 'grid'})
        if not inplace: return target

    def subtract_from_grid(self, value: float, inplace: bool = False):
        """Subtracts a constant value from the gridded data."""
        if self.grid is None: raise ValueError("No gridded data available.")
        target = self if inplace else copy.deepcopy(self)
        target.grid.z_image = target.grid.z_image - value
        target.log_step('subtract', {'value': value, 'target': 'grid'})
        if not inplace: return target

    def multiply_grid_by(self, value: float, inplace: bool = False):
        """Multiplies the gridded data by a constant value."""
        if self.grid is None: raise ValueError("No gridded data available.")
        target = self if inplace else copy.deepcopy(self)
        target.grid.z_image = target.grid.z_image * value
        target.log_step('multiply', {'value': value, 'target': 'grid'})
        if not inplace: return target

    def divide_grid_by(self, value: float, inplace: bool = False):
        """Divides the gridded data by a constant value."""
        if self.grid is None: raise ValueError("No gridded data available.")
        target = self if inplace else copy.deepcopy(self)
        target.grid.z_image = target.grid.z_image / value
        target.log_step('divide', {'value': value, 'target': 'grid'})
        if not inplace: return target

    def set_grid_mean(self, value: float, inplace: bool = False):
        """
        Shifts the gridded data so that its mean is equal to a target value.
        """
        if self.grid is None: raise ValueError("No gridded data available.")
        target = self if inplace else copy.deepcopy(self)
        current_mean = np.nanmean(target.grid.z_image)
        shift = value - current_mean
        target.grid.z_image = target.grid.z_image + shift
        target.log_step('set_mean', {'value': value, 'target': 'grid'})
        if not inplace: return target

    def set_grid_median(self, value: float, inplace: bool = False):
        """
        Shifts the gridded data so that its median is equal to a target value.
        """
        if self.grid is None: raise ValueError("No gridded data available.")
        target = self if inplace else copy.deepcopy(self)
        current_median = np.nanmedian(target.grid.z_image)
        shift = value - current_median
        target.grid.z_image = target.grid.z_image + shift
        target.log_step('set_median', {'value': value, 'target': 'grid'})
        if not inplace: return target
