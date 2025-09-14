# -*- coding: utf-8 -*-
'''
    geophpy.plotting.plot2D
    -----------------------

    Dataset display in 2-dimensions.

    :copyright: Copyright 2014-2020 L. Darras, P. Marty, Q. Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import matplotlib.colors as colors
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# to get matplotlib filled markers list
from matplotlib.lines import Line2D

from six import iteritems

from mpl_toolkits.axes_grid1 import make_axes_locatable

from geophpy.core.utils import *
from .decorators import grid_plot_setup, point_plot_setup 
from . import helpers
#from geophpy.plotting.plot import _init_figure

# to avoid using pyplot to make figure (retained in memory, could be problematic with a GUI)
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib.figure import Figure

#from scipy.interpolate import griddata


# list of matplotlib filled marker (code from matplotlib.org)
UNFILLED_MARKERS = [m for m, func in iteritems(Line2D.markers)
                    if func != 'nothing' and m not in Line2D.filled_markers]
UNFILLED_MARKERS = sorted(UNFILLED_MARKERS,
                          key=lambda x: (str(type(x)), str(x)))[::-1]


@grid_plot_setup
def plot_surface(survey, fig, ax, **kwargs):
    """
    Plots the gridded data as a 2D surface image using `imshow`.

    This is a low-level plotting function that is responsible only for
    drawing the main data image onto the axes.

    Parameters
    ----------
    survey : Survey
        The Survey object containing the `grid` data to be plotted.
        Provided by the decorator.
    fig : matplotlib.figure.Figure
        The figure object to draw on. Provided by the decorator.
    ax : matplotlib.axes.Axes
        The axes object to draw on. Provided by the decorator.
    **kwargs
        Keyword arguments are passed to different functions:
        
        Directly to `matplotlib.axes.Axes.imshow`:
            - cmap (str): The colormap name. Defaults to `survey.info.cmapname`.
            - interpolation (str): e.g., 'bilinear', 'nearest'.
            - vmin, vmax (float): To set the color scale limits.
            - See Matplotlib docs for a full list.
        
        To helper functions via the decorator:
            - labeldisplay (bool): Show or hide titles and labels.
            - axisdisplay (bool): Show or hide the x and y axes.
            - pointsdisplay (bool): Overlay the raw data points on the plot.
            - rects (list): A list of rectangles to draw on the plot.
            - cmapdisplay (bool): Show or hide the colorbar.

    Returns
    -------
    matplotlib.image.AxesImage
        The mappable object created by `imshow`, which is used by the
        decorator's helpers to create a colorbar.

    See Also
    --------
    geophpy.visualization.decorators.grid_plot_setup : The decorator that
        handles figure setup and data validation.
    geophpy.visualization.helpers.apply_plot_options : The helper function
        that processes many of the display-related `**kwargs`.

    Notes
    -----
    This function is designed to be wrapped by the `@grid_plot_setup`
    decorator. The decorator is responsible for creating the `fig` and `ax`
    objects, checking if `survey.grid` data exists, and calling helper

    functions to apply common plot options (like titles, colorbars, etc.)
    after this function has run.
    """

    # Pop cmap from kwargs or use the default from the survey's info
    cmap = kwargs.pop('cmap', survey.info.cmapname)

    default_extent = (survey.info.x_min, survey.info.x_max,
                      survey.info.y_min, survey.info.y_max)
    extent = kwargs.pop('extent', default_extent)  # Use the default extent unless the user provides a custom one.

    
    mappable = ax.imshow(survey.grid.z_image, extent = extent, origin='lower', cmap=cmap, **kwargs)
    
    ax.set_aspect('equal')
    
    # Return the mappable object so the decorator can create a colorbar
    return mappable


@grid_plot_setup
def plot_contour(survey, fig, ax, **kwargs):
    """
    Plots the gridded data as a 2D contour map.

    This function is decorated by @grid_plot_setup.
    """
    cmap = kwargs.pop('cmap', survey.info.cmapname)
    levels = kwargs.pop('levels', 10)
    
    # Use our helper to generate the contour levels
    contour_levels = helpers.make_levels_for_contour(survey, levels=levels, **kwargs)
    
    mappable = ax.contour(survey.grid.z_image, levels=contour_levels, cmap=cmap,
                          extent=(survey.info.x_min, survey.info.x_max,
                                  survey.info.y_min, survey.info.y_max),
                          **kwargs)
    
    ax.set_aspect('equal')
    
    # Return the mappable object so the decorator can create a colorbar
    return mappable


@grid_plot_setup
def plot_contourf(survey, fig, ax, **kwargs):
    """
    Plots the gridded data as a filled contour map.

    This function is decorated by @grid_plot_setup, which handles data
    validation and figure/axes creation.

    Parameters
    ----------
    survey : Survey
        The Survey object containing the `grid` and `info` data.
    fig : matplotlib.figure.Figure
        The figure object to draw on (provided by the decorator).
    ax : matplotlib.axes.Axes
        The axes object to draw on (provided by the decorator).
    **kwargs
        Keyword arguments passed to `matplotlib.axes.Axes.contourf`.
        - levels (int or array-like): The number of contour levels or their values.
        - cmap (str): The colormap name. Defaults to `survey.info.cmapname`.
        - See Matplotlib docs for a full list.
    """
    # Pop specific arguments from kwargs or use defaults
    cmap = kwargs.pop('cmap', survey.info.cmapname)
    levels = kwargs.pop('levels', 10)

    # Use our helper function to generate the contour levels
    contour_levels = helpers.make_levels_for_contour(survey, levels=levels, **kwargs)

    # The core action: draw the filled contours
    mappable = ax.contourf(
        survey.grid.z_image, 
        levels=contour_levels, 
        cmap=cmap,
        extent=(survey.info.x_min, survey.info.x_max,
                survey.info.y_min, survey.info.y_max),
        **kwargs
    )
    
    ax.set_aspect('equal')

    # Return the mappable object so the decorator can create a colorbar
    return mappable


@point_plot_setup
def plot_scatter(survey, fig, ax, **kwargs):
    """
    Plots the ungridded data as a 2D scatter plot.
    
    The color of each point represents its measurement value. This function
    is decorated by @point_plot_setup.
    """
    
    cmap = kwargs.pop('cmap', survey.info.cmapname)

    mappable = ax.scatter(survey.points.x, survey.points.y, c=survey.points.values, cmap=cmap, **kwargs)
    
    ax.set_aspect('equal')
    
    # The decorator's helpers need the mappable to create a colorbar
    return mappable


@point_plot_setup
def plot_postmap(survey, fig, ax, **kwargs):
    """
    Plots the ungridded data point locations (postmap or track plot).

    This function is decorated by @point_plot_setup, which handles data
    validation and figure/axes creation.

    Parameters
    ----------
    survey : Survey
        The Survey object containing the `points` data.
    fig : matplotlib.figure.Figure
        The figure object to draw on (provided by the decorator).
    ax : matplotlib.axes.Axes
        The axes object to draw on (provided by the decorator).
    **kwargs
        Additional keyword arguments passed to `matplotlib.axes.Axes.plot`.
        For example: `marker='.'`, `color='blue'`, `markersize=2`.
    """
    
    # Pop specific arguments or set defaults    
    kwargs.pop('cmap', None)  # This function uses ax.plot(), which does not accept 'cmap' and is safely removed here
    kwargs.pop('vmin', None)  # This function uses Line2D.set(), which does not accept 'vmin' and is safely removed here
    kwargs.pop('vmax', None)  # This function uses Line2D.set(), which does not accept 'vmax' and is safely removed here
    marker = kwargs.pop('marker', '.')
    color = kwargs.pop('color', 'black')
    linestyle = kwargs.pop('linestyle', 'None')
    
    # The core action: plot the x, y coordinates
    ax.plot(survey.points.x, survey.points.y,
            marker=marker, color=color, linestyle=linestyle, **kwargs)
    
    ax.set_aspect('equal')

    # No mappable object is returned for a simple line/marker plot
    return None
