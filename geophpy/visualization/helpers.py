# -*- coding: utf-8 -*-
"""
   geophpy.visualization.helpers
   -----------------------------

   Provides low-level helper functions for creating and customizing plots.

   These functions handle repetitive tasks like creating colorbars, applying
   common plot options, and adding overlays like points or rectangles.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from . import helpers

def get_colormap(cmapname, creversed=False):
    """
    Returns a Matplotlib colormap object from a color map name.

    Parameters
    ----------
    cmapname : str
        The name of the colormap (e.g., 'viridis', 'gray').
    creversed : bool, optional
        If True, the reversed version of the colormap is returned.
        Defaults to False.

    Returns
    -------
    matplotlib.colors.Colormap
        The requested colormap object.
    """
    if creversed:
        cmapname = cmapname + '_r'
    return plt.get_cmap(cmapname)


def get_colorscale(logscale=False):
    """
    Returns a Matplotlib linear or logarithmic colorscale normalizer.
    """
    if logscale:
        return colors.LogNorm()
    return colors.Normalize()


def create_colorbar(fig, ax, mappable, **kwargs):
    """
    Adds a colorbar to the axes for a given mappable object.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    mappable : object
        The object (e.g., an image from `imshow`) to which the colorbar applies.
    **kwargs
        - cmapdisplay (bool): If False, the colorbar is not created.
    """
    if kwargs.get('cmapdisplay', True):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        return fig.colorbar(mappable, cax=cax)
    return None


def apply_plot_options(survey, fig, ax, **kwargs):
    """
    Applies a variety of common plot options to a given axes.

    Parameters
    ----------
    survey : Survey
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    **kwargs
        A dictionary of plot options (see below).

    Keyword Arguments
    -----------------
    title : str
    labeldisplay : bool
    axisdisplay : bool
    pointsdisplay : bool
    gridpointsdisplay : bool
    marker : str
    rects : list
    rect_color : str
    points : list
    point_color : str
    """

    
    # Label Display
    if kwargs.get('labeldisplay', True):
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(survey.name)

    # Title Display
    if 'title' in kwargs:
        ax.set_title(kwargs.get('title'))

    # Axis Visibility
    ax.get_xaxis().set_visible(kwargs.get('axisdisplay', True))
    ax.get_yaxis().set_visible(kwargs.get('axisdisplay', True))

    # Overlay raw data points
    if kwargs.get('pointsdisplay', False):
        marker = kwargs.get('marker', '.')
        ax.plot(survey.points.x, survey.points.y, 'k', ms=1, linestyle='None', marker=marker)

    # Overlay grid node locations
    if kwargs.get('gridpointsdisplay', False):
        if survey.grid is not None and survey.grid.easting_image is not None:
            ax.plot(survey.grid.easting_image, survey.grid.northing_image, 'b+', ms=1)
        else:
            print("Warning: Cannot display grid points. Grid or its coordinates are not available.")

    # Overlay custom rectangles
    rects = kwargs.get('rects', None)
    if rects is not None:
        rect_color = kwargs.get('rect_color', "red")
        for x, y, w, h in rects:
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor=rect_color))
    
    # Overlay custom points
    points = kwargs.get('points', None)
    if points is not None:
        point_color = kwargs.get('point_color', "green")
        marker = kwargs.get('marker', 'o')
        px, py = zip(*points) # Unzip list of (x,y) tuples
        ax.plot(px, py, color=point_color, linestyle='None', marker=marker)


def make_levels_for_contour(survey, levels=10, **kwargs):
    """
    Generates an array of level values for contour plots.

    If `levels` is an integer, it creates that many evenly spaced levels
    between the min and max values of the grid data. If `levels` is already
    an array-like object, it is returned directly.

    Parameters
    ----------
    survey : Survey
        The Survey object containing the gridded data.
    levels : int or array-like, optional
        Either the number of contour levels to generate or a pre-defined
        array of contour levels. Defaults to 10.
    **kwargs
        - vmin (float): A minimum value to use instead of the grid's minimum.
        - vmax (float): A maximum value to use instead of the grid's maximum.

    Returns
    -------
    np.ndarray
        An array of values at which to draw contour lines.
    """
    if isinstance(levels, int):
        vmin = kwargs.get('vmin', np.nanmin(survey.grid.z_image))
        vmax = kwargs.get('vmax', np.nanmax(survey.grid.z_image))
        return np.linspace(vmin, vmax, levels)
    # If levels is already a list or array, return it as is
    return levels
