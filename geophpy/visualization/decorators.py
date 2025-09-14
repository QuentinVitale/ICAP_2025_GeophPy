# -*- coding: utf-8 -*-
"""
   geophpy.visualization.decorators
   --------------------------------

   Provides decorator functions to streamline the creation of plots.

   These decorators handle repetitive tasks such as checking for the required
   data types (points vs. grid) and initializing Matplotlib figures and axes,
   allowing the low-level plotting functions to focus solely on the plotting
   logic itself.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

import matplotlib.pyplot as plt
import numpy as np

from functools import wraps
from . import helpers

def _handle_legacy_args(kwargs):
    """A helper to rename old arguments to their new standard names."""
    if 'cmmin' in kwargs:
        kwargs['vmin'] = kwargs.pop('cmmin')
    if 'cmmax' in kwargs:
        kwargs['vmax'] = kwargs.pop('cmmax')

# def _setup_common_2d_axes(ax):
#     """Helper to apply settings common to all 2D plots."""
#     ax.set_aspect('equal')

def curve_plot_setup(func):
    """
    Decorator for simple 1D curve plots (e.g., profiles, histograms).

    This decorator only handles the initialization of a Matplotlib Figure
    and Axes. It does not perform any data validation or apply 2D-specific
    settings like aspect ratio.
    """
    @wraps(func)
    def wrapper(survey, *args, fig=None, **kwargs):
        # Pop fig and ax from kwargs so they are not passed down twice
        fig = kwargs.pop('fig', None)
        ax = kwargs.pop('ax', None)
        
        # If no axes were provided by the user, create them.
        if ax is None:
            if fig is None: fig = plt.figure()
            else: fig.clf()
            ax = fig.add_subplot(111)
        # If axes were provided, get their parent figure.
        else:
            fig = ax.figure

        # Call the original plotting function with the prepared fig and ax
        func(survey, *args, fig=fig, ax=ax, **kwargs)
        
        # Apply a simplified set of options (no colorbar, no aspect ratio)
        helpers.apply_plot_options(survey, fig, ax, **kwargs)

        return fig, ax
    return wrapper

def point_plot_setup(func):
    """
    Decorator for plotting functions that require ungridded (point) data.

    This decorator performs two main actions:
    1. It verifies that the `survey` object contains valid point data.
    2. It handles the initialization of a Matplotlib `Figure` and `Axes`,
       creating a new figure or clearing an existing one as needed.

    The decorated function must accept `survey`, `fig`, and `ax` as its
    first three arguments, in addition to `**kwargs`.

    Parameters
    ----------
    func : callable
        The plotting function to be decorated.

    Returns
    -------
    callable
        The wrapped function.
    """
    @wraps(func)
    def wrapper(survey, *args, fig=None, **kwargs):
        # 1. Safety check for point data
        if survey.points is None or survey.points.values.size == 0:
            raise ValueError("No point data available for this plot type.")

        # Initialize or clear the figure and axes
        fig = kwargs.pop('fig', None)
        ax = kwargs.pop('ax', None)
        if ax is None:
            if fig is None:
                fig = plt.figure()
            else: 
                fig.clf()
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure # Get the figure from the provided axes
        
        # 2. Define keywords that are for our helpers, not for Matplotlib's core functions.
        helper_keys = ['labeldisplay', 'axisdisplay', 'pointsdisplay', 'gridpointsdisplay',
                       'rects', 'rect_color', 'points', 'point_color', 'title', 'cmapdisplay']
        
        # 3. Separate the kwargs.
        helper_kwargs = {}
        plot_kwargs = {}
        for key, value in kwargs.items():
            if key in helper_keys:
                helper_kwargs[key] = value
            else:
                plot_kwargs[key] = value

        _handle_legacy_args(kwargs)
        

        # # 5. Smart vmin/vmax using percentiles ---
        # if 'vmin' not in kwargs and 'vmax' not in kwargs:
        #     valid_values = survey.points.values[~np.isnan(survey.points.values)]
        #     if valid_values.size > 0:
        #         kwargs['vmin'] = np.percentile(valid_values, 5)
        #         kwargs['vmax'] = np.percentile(valid_values, 95)
    
        # 1. The decorator sets GENERIC default labels FIRST.
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(survey.name)

        # 6. Call the original plotting function with the prepared fig and ax
        # The specialist plotting function, which can now OVERRIDE the defaults.
        mappable = func(survey, *args, fig=fig, ax=ax, **plot_kwargs)
        
        # _setup_common_2d_axes(ax) # Set the aspect ratio to equal to avoid distortion

        # 7. If the function returned a mappable object, create a colorbar
        if mappable:
            helpers.create_colorbar(fig, ax, mappable, **helper_kwargs)
        
        return fig, ax
    return wrapper


def grid_plot_setup(func):
    """
    Decorator for plotting functions that require gridded (image-like) data.

    This decorator performs two main actions:
    1. It verifies that the `survey` object has been interpolated and
       contains valid grid data.
    2. It handles the initialization of a Matplotlib `Figure` and `Axes`.

    The decorated function must accept `survey`, `fig`, and `ax` as its
    first three arguments, in addition to `**kwargs`.

    Parameters
    ----------
    func : callable
        The plotting function to be decorated.

    Returns
    -------
    callable
        The wrapped function.
    """
    @wraps(func)
    def wrapper(survey, *args, fig=None, **kwargs):
        # 1. Safety check for gridded data
        if survey.grid is None or survey.grid.z_image is None:
            raise ValueError("No gridded data available for this plot type. "
                             "Please run the .interpolate() method first.")
                
        # Initialize or clear the figure and axes
        fig = kwargs.pop('fig', None)
        ax = kwargs.pop('ax', None)
        if ax is None:
            if fig is None: fig = plt.figure()
            else: fig.clf()
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure # Get the figure from the provided axes
        
        # 2. Define keywords that are for our helpers, not for Matplotlib's core functions.
        helper_keys = ['labeldisplay', 'axisdisplay', 'pointsdisplay', 'gridpointsdisplay',
                       'rects', 'rect_color', 'points', 'point_color', 'title', 'cmapdisplay']
        
        # 3. Separate the kwargs.
        helper_kwargs = {}
        plot_kwargs = {}
        for key, value in kwargs.items():
            if key in helper_keys:
                helper_kwargs[key] = value
            else:
                plot_kwargs[key] = value

        _handle_legacy_args(kwargs)

        # 1. The decorator sets GENERIC default labels FIRST.
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(survey.name)

        # # # 5. Smart vmin/vmax using percentiles ---
        # if 'cmap' in plot_kwargs and 'vmin' not in plot_kwargs and 'vmax' not in plot_kwargs:
        #     valid_values = survey.grid.z_image[~np.isnan(survey.grid.z_image)]
        #     if valid_values.size > 0:
        #         plot_kwargs['vmin'] = np.percentile(valid_values, 5)
        #         plot_kwargs['vmax'] = np.percentile(valid_values, 95)
        # if 'cmap' in plot_kwargs and 'vmin' not in plot_kwargs and 'vmax' not in plot_kwargs:
        #     valid_values = survey.grid.z_image[~np.isnan(survey.grid.z_image)]
        #     if valid_values.size > 0:
        #         plot_kwargs['vmin'] = np.percentile(valid_values, 5)
        #         plot_kwargs['vmax'] = np.percentile(valid_values, 95)

        # 6. Call the core plotting function with the prepared fig and ax
        # and returns the "mappable" object needed for the colorbar 
        mappable = func(survey, *args, fig=fig, ax=ax, **plot_kwargs)

        # 5.  Apply common options and create the colorbar using helpers
        # _setup_common_2d_axes(ax) # Set the aspect ratio to equal to avoid distortion
        helpers.create_colorbar(fig, ax, mappable, **helper_kwargs)
        helpers.apply_plot_options(survey, fig, ax, **helper_kwargs)

        return fig, ax
    return wrapper
