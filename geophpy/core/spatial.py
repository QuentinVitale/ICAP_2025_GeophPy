# -*- coding: utf-8 -*-
"""
   geophpy.core.spatial
   --------------------

   Provides Mixins for spatial operations on Survey objects, such as
   interpolation and transformations for both point and gridded data.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

import copy
import numpy as np
import dataclasses
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from .datastructures import PointData, GridData, Info
from .utils import get_median_step, get_decimals_nb


# --- Low-Level Functions ---
def create_grid_from_points(
        point_data: PointData, 
        info: Info,
        method: str = 'linear', 
        x_step: float = None, 
        y_step: float = None, 
        x_prec: int = 2, 
        y_prec: int = 2, 
        x_frame_factor: float = 0., 
        y_frame_factor: float = 0.
    ) -> (GridData, Info):
    """
    Creates a 2D grid and updated metadata from scatter point data.

    This pure function contains the core gridding algorithm. It is called by
    the main Survey.interpolate() method.

    Parameters
    ----------
    point_data : PointData
        A dataclass containing the x, y, and values arrays.
    info : Info
        The current Info dataclass from the Survey object. It will be updated.
    method : {'linear', 'nearest', 'cubic', 'none'}, optional
        The interpolation method to use. 'none' uses binning.
        Defaults to 'linear'.
    x_step : float, optional
        Grid step size in the x-direction. If None, it's calculated.
    y_step : float, optional
        Grid step size in the y-direction. If None, it's calculated.
    x_prec : int, optional
        Precision for rounding x-coordinates.
    y_prec : int, optional
        Precision for rounding y-coordinates.
    x_frame_factor : float, optional
        A factor to expand the grid frame in the x-direction.
    y_frame_factor : float, optional
        A factor to expand the grid frame in the y-direction.

    Returns
    -------
    GridData
        A new GridData object containing the `z_image` numpy array.
    Info
        An updated Info object with the new grid metadata.
    """
    x, y, z = point_data.x, point_data.y, point_data.values

    # --- 1. Creating a regular grid ---
    
    # Median step between two distinct x values
    if x_step is None:
        x_step = get_median_step(point_data.x, prec=x_prec)

    else:
        x_prec = get_decimals_nb(x_step)
        pass
    
    # Min and max x coordinates and number of x pixels
    xmin, xmax = x.min(), x.max()
    xmin = (1. + x_frame_factor) * xmin - x_frame_factor * xmax
    xmax = (1. + x_frame_factor) * xmax - x_frame_factor * xmin
    xmin, xmax = round(xmin, x_prec), round(xmax, x_prec)
    nx = int(np.around((xmax - xmin) / x_step) + 1)

    # Median step between two distinct y values
    if y_step is None:
        y_step = get_median_step(point_data.y, prec=y_prec)

    else:
        y_prec = get_decimals_nb(y_step)

    # Determinate min and max y coordinates and number of y pixels
    ymin, ymax = y.min(), y.max()
    ymin = (1. + y_frame_factor) * ymin - y_frame_factor * ymax
    ymax = (1. + y_frame_factor) * ymax - y_frame_factor * ymin
    ymin, ymax = round(ymin, y_prec), round(ymax, y_prec)
    ny = int(np.around((ymax - ymin) / y_step) + 1)

    # Regular grid in both x- and y-direction
    xi = np.linspace(xmin, xmax, nx, endpoint=True)
    yi = np.linspace(ymin, ymax, ny, endpoint=True)
    X, Y = np.meshgrid(xi, yi)

    # --- 2. Gridding data ---
    Z = None

    # No interpolation: Binning method
    if method.lower() == "none":
        statistic, _, _, _ = binned_statistic_2d(x, y, values=z, statistic='mean', bins=[xi, yi])
        Z = statistic.T

    # SciPy interpolation
    elif method in ['linear', 'nearest', 'cubic']:
        Z = griddata((x, y), z, (X, Y), method=method)
        if np.all(np.isnan(Z.flatten())):
            print("Warning: Interpolation with griddata failed, result is all NaNs.")
            return None, None
    
    # Other iterpolation
    else:
        raise ValueError(f"Undefined interpolation method: '{method}'")

    # --- 3. Create and populate the new data objects to be returned ---
    new_grid_data = GridData(z_image=Z)

    updated_info = info # Start with a copy of the old info
    updated_info.x_min = xmin
    updated_info.x_max = xmax
    updated_info.y_min = ymin
    updated_info.y_max = ymax
    updated_info.z_min = np.nanmin(Z)
    updated_info.z_max = np.nanmax(Z)
    updated_info.x_gridding_delta = x_step
    updated_info.y_gridding_delta = y_step
    updated_info.gridding_interpolation = method
    
    return new_grid_data, updated_info


class SpatialPointsMixin:
    """
    Mixin for spatial operations that work on ungridded (point) data.
    """

    def rotate_points(self, angle: float, center: tuple = (0, 0), inplace: bool = False):
        """
        Rotates the ungridded point data around a center point.

        By default, this method returns a new Survey object with the rotated
        point data, leaving the original object unchanged.

        Parameters
        ----------
        angle : float
            The angle of rotation in degrees. Positive values are
            counter-clockwise.
        center : tuple of (float, float), optional
            The (x, y) coordinate of the center of rotation.
            Defaults to the origin (0, 0).
        inplace : bool, optional
            If True, perform the operation in-place and return None.
            If False, return a new, modified Survey object.
            Defaults to False.
        
        Returns
        -------
        Survey or None
            A new Survey object with rotated points if `inplace` is False.
            None if `inplace` is True.
        
        Raises
        ------
        ValueError
            If no point data is available to rotate.
        """
        if self.points is None or self.points.x.size == 0:
            raise ValueError("No point data available to rotate.")

        # --- Choose the target object ---
        target_survey = self if inplace else copy.deepcopy(self)

        # --- Perform the 2D rotation ---
        angle_rad = np.deg2rad(angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])

        # Center the points around the rotation origin
        points = np.vstack((target_survey.points.x - center[0], 
                            target_survey.points.y - center[1]))

        # Apply the rotation
        rotated_points = rotation_matrix @ points

        # Translate the points back and update the target object
        target_survey.points.x = rotated_points[0] + center[0]
        target_survey.points.y = rotated_points[1] + center[1]

        # --- Log the operation and return the result ---
        target_survey.log_step('rotate', {'angle': angle, 'center': center})

        if not inplace:
            return target_survey

    def translate_points(self, x_shift: float = 0, y_shift: float = 0, inplace: bool = False):
        """
        Translates the ungridded point data by a given shift.

        Parameters
        ----------
        x_shift : float, optional
            The value to add to all x-coordinates. Defaults to 0.
        y_shift : float, optional
            The value to add to all y-coordinates. Defaults to 0.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with translated points if `inplace` is False.
        """
        if self.points is None:
            raise ValueError("No point data available to translate.")

        target = self if inplace else copy.deepcopy(self)

        # Apply the translation
        target.points.x += x_shift
        target.points.y += y_shift
        
        target.log_step('translate', {'x_shift': x_shift, 'y_shift': y_shift})

        if not inplace:
            return target

    def crop_points(self, xmin=None, xmax=None, ymin=None, ymax=None, inplace: bool = False):
        """
        Crops the ungridded point data to a specified rectangular area.

        This method filters out points that fall outside the defined
        x and y boundaries. It automatically applies the filter to all compatible
        numpy arrays found within the `self.points` dataclass.

        Parameters
        ----------
        xmin : float, optional
            The minimum x-coordinate to keep.
        xmax : float, optional
            The maximum x-coordinate to keep.
        ymin : float, optional
            The minimum y-coordinate to keep.
        ymax : float, optional
            The maximum y-coordinate to keep.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the cropped points if `inplace` is False.
        """

        if self.points is None:
            raise ValueError("No point data available to crop.")

        target = self if inplace else copy.deepcopy(self)
        
        # Store the original size of the coordinate arrays
        original_size = target.points.x.size
        if original_size == 0:
            return target if not inplace else None # Nothing to crop

        # Build a boolean mask from the primary coordinates
        mask = np.ones(original_size, dtype=bool)
        if xmin is not None: mask &= (target.points.x >= xmin)
        if xmax is not None: mask &= (target.points.x <= xmax)
        if ymin is not None: mask &= (target.points.y >= ymin)
        if ymax is not None: mask &= (target.points.y <= ymax)

        # Iterate over all fields defined in the PointData dataclass
        for field in dataclasses.fields(target.points):
            attr_name = field.name
            attr_value = getattr(target.points, attr_name)

            # Check if the attribute is a numpy array with the correct size
            if isinstance(attr_value, np.ndarray) and attr_value.size == original_size:
                # If it is, apply the mask
                setattr(target.points, attr_name, attr_value[mask])

        target.log_step('crop', {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})

        if not inplace:
            return target

    def get_track(self, track_number: int, attribute: str = 'values'):
        """
        Extracts a specific data attribute for a single survey track.

        This method filters the point data to return the values of a
        specified attribute (e.g., 'values', 'x', 'y') that correspond
        to a single track number.

        Parameters
        ----------
        track_number : int
            The integer ID of the track to extract.
        attribute : {'values', 'x', 'y', 'east', 'north'}, optional
            The name of the data attribute to return for the specified
            track. Defaults to 'values'.

        Returns
        -------
        np.ndarray
            A 1D NumPy array containing the requested data for the track.
            Returns an empty array if the track is not found.

        Raises
        ------
        ValueError
            If the survey does not contain any track information or if the
            track array has a different size than the coordinate arrays.
        AttributeError
            If the requested `attribute` does not exist in the point data.
        """

        if self.points is None or getattr(self.points, 'track', None) is None:
            raise ValueError(
                "No track information is available in this survey.\n"
                "Hint: If your data file does not contain track/profile numbers, "
                "you may need to run a method like `.estimate_tracks_from_coords()` first."
            )
        
        if self.points.track.size != self.points.x.size:
            raise ValueError("Track array size does not match coordinate array size. Data is inconsistent.")

        # The rest of the logic remains the same
        track_indices = np.where(self.points.track == track_number)[0]
        if track_indices.size == 0:
            print(f"Warning: Track number {track_number} not found.")
            return np.array([])

        data_to_extract = getattr(self.points, attribute)
        return data_to_extract[track_indices]


class SpatialGridMixin:
    """
    Mixin for spatial operations that work on gridded (image-like) data.
    """

    def rotate_grid(self, angle: int = 90, inplace: bool = False):
        """
        Rotates the gridded data by a multiple of 90 degrees.

        By default, this method returns a new Survey object with the rotated
        data, leaving the original object unchanged.

        Parameters
        ----------
        angle : {0, 90, 180, 270}, optional
            The angle of rotation in degrees. Defaults to 90.
        inplace : bool, optional
            If True, perform the operation in-place on the current object and
            return None. If False, return a new, modified Survey object.
            Defaults to False.
        
        Returns
        -------
        Survey or None
            A new, rotated Survey object if `inplace` is False.
            None if `inplace` is True.
        
        Raises
        ------
        ValueError
            If the data has not been gridded yet.
        """
        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available to rotate. Please run .interpolate() first.")
        
        if angle not in [0, 90, 180, 270]:
            raise ValueError(f"Invalid angle: {angle}. Must be one of 0, 90, 180, 270.")

        # --- Choose the target object ---
        # If inplace, modify the object itself. Otherwise, work on a deep copy.
        target_survey = self if inplace else copy.deepcopy(self)

        # --- Perform the rotation on the target ---
        k = angle // 90
        target_survey.grid.z_image = np.rot90(self.grid.z_image, k=k)
        
        if self.grid.easting_image is not None:
            target_survey.grid.easting_image = np.rot90(self.grid.easting_image, k=k)
        if self.grid.northing_image is not None:
            target_survey.grid.northing_image = np.rot90(self.grid.northing_image, k=k)

        # --- Log the operation and return the result ---
        target_survey.log_step('rotate', {'angle': angle})

        if not inplace:
            return target_survey

    def translate_grid(self, x_shift: float = 0, y_shift: float = 0, inplace: bool = False):
        """
        Translates the gridded data by updating its coordinate metadata.

        Note: This method does not re-interpolate the data. It only shifts
        the coordinate system of the existing grid.

        Parameters
        ----------
        x_shift : float, optional
            The value to add to the grid's x-coordinates. Defaults to 0.
        y_shift : float, optional
            The value to add to the grid's y-coordinates. Defaults to 0.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the translated grid if `inplace` is False.
        """
        if self.grid is None:
            raise ValueError("No gridded data available to translate.")

        target = self if inplace else copy.deepcopy(self)

        # Update the grid's coordinate metadata
        target.info.x_min += x_shift
        target.info.x_max += x_shift
        target.info.y_min += y_shift
        target.info.y_max += y_shift

        if target.grid.easting_image is not None:
            target.grid.easting_image += x_shift
        if target.grid.northing_image is not None:
            target.grid.northing_image += y_shift
        
        target.log_step('translate', {'x_shift': x_shift, 'y_shift': y_shift})
        
        if not inplace:
            return target

    def crop_grid(self, xmin=None, xmax=None, ymin=None, ymax=None, inplace: bool = False):
        """
        Crops the gridded data to a specified rectangular area.

        This method slices the grid arrays and updates the corresponding
        metadata in `self.info`.

        Parameters
        ----------
        xmin : float, optional
            The new minimum x-coordinate.
        xmax : float, optional
            The new maximum x-coordinate.
        ymin : float, optional
            The new minimum y-coordinate.
        ymax : float, optional
            The new maximum y-coordinate.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the cropped grid if `inplace` is False.
        """
        if self.grid is None:
            raise ValueError("No gridded data available to crop.")

        target = self if inplace else copy.deepcopy(self)

        # Get the current grid vectors
        x_vect = target.get_xvect()
        y_vect = target.get_yvect()

        # Find the pixel indices for the new boundaries
        x_start_index = np.searchsorted(x_vect, xmin) if xmin is not None else 0
        x_end_index = np.searchsorted(x_vect, xmax, side='right') if xmax is not None else len(x_vect)
        y_start_index = np.searchsorted(y_vect, ymin) if ymin is not None else 0
        y_end_index = np.searchsorted(y_vect, ymax, side='right') if ymax is not None else len(y_vect)
        
        # Slice all grid arrays
        target.grid.z_image = target.grid.z_image[y_start_index:y_end_index, x_start_index:x_end_index]
        if target.grid.easting_image is not None:
            target.grid.easting_image = target.grid.easting_image[y_start_index:y_end_index, x_start_index:x_end_index]

        # Update the metadata in the info object
        target.info.x_min = x_vect[x_start_index]
        target.info.x_max = x_vect[x_end_index - 1]
        target.info.y_min = y_vect[y_start_index]
        target.info.y_max = y_vect[y_end_index - 1]

        target.log_step('crop', {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})

        if not inplace:
            return target

    def flip_grid(self, direction: str, inplace: bool = False):
        """
        Flips the gridded data horizontally or vertically.

        By default, this method returns a new Survey object with the flipped
        data, leaving the original object unchanged.

        Parameters
        ----------
        direction : {'horizontal', 'vertical'}
            The direction in which to flip the grid. 'horizontal' flips
            left-to-right, 'vertical' flips up-to-down.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new, flipped Survey object if `inplace` is False.
            None if `inplace` is True.

        Raises
        ------
        ValueError
            If the data has not been gridded yet or if the direction is
            not valid.
        """
        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available to flip. Please run .interpolate() first.")
        
        direction = direction.lower()
        if direction not in ['horizontal', 'vertical']:
            raise ValueError(f"Invalid direction: '{direction}'. Must be 'horizontal' or 'vertical'.")

        target = self if inplace else copy.deepcopy(self)

        # --- Perform the flip on the target ---
        if direction == 'horizontal':
            axis = 1  # Flip along the vertical axis (left-right)
        else: # vertical
            axis = 0  # Flip along the horizontal axis (up-down)
            
        target.grid.z_image = np.flip(self.grid.z_image, axis=axis)
        
        if self.grid.easting_image is not None:
            target.grid.easting_image = np.flip(self.grid.easting_image, axis=axis)
        if self.grid.northing_image is not None:
            target.grid.northing_image = np.flip(self.grid.northing_image, axis=axis)

        target.log_step('flip', {'direction': direction})
        
        if not inplace:
            return target

        

