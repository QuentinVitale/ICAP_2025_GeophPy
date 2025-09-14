# -*- coding: utf-8 -*-
"""
   geophpy.visualization.histo
   ---------------------------

   Provides the low-level function for plotting data histograms.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from .decorators import point_plot_setup, grid_plot_setup
from . import helpers
#import geophpy.plotting.plot as plot1D
#from geophpy.visualization.plot import _init_figure

from mpl_toolkits.axes_grid1 import make_axes_locatable

@point_plot_setup
def plot_points_histogram(survey, fig, ax, **kwargs):
    """
    Plots a histogram of the ungridded point data values.

    Parameters
    ----------
    survey : Survey
        The Survey object containing the `points` data.
    fig : matplotlib.figure.Figure
        The figure object to draw on (provided by the decorator).
    ax : matplotlib.axes.Axes
        The axes object to draw on (provided by the decorator).
    **kwargs
        Keyword arguments for the plot.
        - bins (int): The number of bins for the histogram.
        - range (tuple): The lower and upper range of the bins, e.g., (vmin, vmax).
        - colored (bool): If True, colors the bars according to a colormap.
    """

    # Get the raw data values and remove any NaNs
    values = survey.points.values[~np.isnan(survey.points.values)]
    
    # --- Core Plotting Logic ---
    bins = kwargs.pop('bins', 50)
    mappable = _draw_histogram(values, fig, ax, bins, **kwargs)
    
    ax.set_title("Histogram of Point Data")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return mappable


@grid_plot_setup
def plot_grid_histogram(survey, fig, ax, **kwargs):
    """
    Plots a histogram of the gridded data values.
    
    Parameters
    ----------
    survey : Survey
        The Survey object containing the `points` data.
    fig : matplotlib.figure.Figure
        The figure object to draw on (provided by the decorator).
    ax : matplotlib.axes.Axes
        The axes object to draw on (provided by the decorator).
    **kwargs
        Keyword arguments for the plot.
        - bins (int): The number of bins for the histogram.
        - range (tuple): The lower and upper range of the bins, e.g., (vmin, vmax).
        - colored (bool): If True, colors the bars according to a colormap.
    """

    # Get the gridded data values and remove any NaNs
    values = survey.grid.z_image.flatten()
    values = values[~np.isnan(values)]

    # --- Core Plotting Logic ---
    bins = kwargs.pop('bins', 50)
    mappable = _draw_histogram(values, fig, ax, bins, **kwargs)

    ax.set_title("Histogram of Gridded Data")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return mappable


def _draw_histogram(values, fig, ax, bins, **kwargs):
    """Private helper function to draw the actual histogram bars."""
    colored = kwargs.pop('colored', True)
    
    hist_range = kwargs.pop('range', None)
    if not colored:
        kwargs.pop('cmap', None)  # safely removes the 'cmap' argument so it is not passed to ax.hist()
        kwargs.pop('vmin', None)
        kwargs.pop('vmax', None)
        
        color = kwargs.pop('color', 'black')
        ax.hist(values, bins=bins, range=hist_range, color=color, **kwargs)
        return None
    else:
        cmap_name = kwargs.pop('cmap', 'viridis')
        n, bin_edges = np.histogram(values, bins=bins, range=hist_range)
        cmap = helpers.get_colormap(cmap_name)
        norm = plt.Normalize(bin_edges.min(), bin_edges.max())
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        for count, edge in zip(n, bin_edges):
            width = bin_edges[1] - bin_edges[0]
            ax.bar(edge, count, width=width, color=cmap(norm(edge)), align='edge')
        
        return mappable


# #def plot(dataset, fig=None, filename=None, zmin=None, zmax=None, cmapname=None, creversed=False, cmapdisplay=True, coloredhisto=True, dpi=None, transparent=False, valfilt=True):
# def plot(dataset, **kwargs):
#     ''' Plot the dataset histogram.

#     cf. :meth:`~geophpy.dataset.DataSet.histo_plot`

#     '''

#     # Plot options
#     fig = kwargs.get('fig', None)
#     zmin = kwargs.get('zmin', None)
#     zmax = kwargs.get('zmax', None)
#     cmapname = kwargs.get('cmapname', None)
#     creversed = kwargs.get('creversed', False)
#     cmapdisplay = kwargs.get('cmapdisplay', True)
#     coloredhisto = kwargs.get('coloredhisto', True)
#     valfilt = kwargs.get('valfilt', False)
#     colorbar = None
#     showtitle = kwargs.get('showtitle', True)

#     # Save options
#     filename = kwargs.get('filename', None)
#     dpi = kwargs.get('dpi', None)
#     transparent = kwargs.get('transparent', False)

#     # Colormap options
#     if cmapname is None and coloredhisto is True:
#         cmapname = dataset.info.cmapname

#     if creversed:
#         cmapname = cmapname + '_r'  # adds '_r' at the colormap name to use the reversed colormap

#     cmap = plt.get_cmap(cmapname)  # gets the colormap

#     # Figure initialization
#     #fig, ax = plot1D._init_figure(fig=fig)  # clear an existing figure and add an empty ax
#     fig, ax = _init_figure(fig=fig)  # clear an existing figure and add an empty ax


#     # First display ############################################################
#     #if fig is None:
#     #    fig = plt.figure()
#     #else:
#     #    fig.clf()

#     # Ignoring NaNs
#     if valfilt or dataset.data.z_image is None:
#         nanidx = np.isnan( dataset.data.values.T[0] )  # index of nan
#         Z = dataset.data.values.T[2][~nanidx]

#     else:
#         nanidx = np.isnan( dataset.data.z_image )  # index of nan
#         Z = dataset.data.z_image[~nanidx]

#     # Creating histogram #######################################################
#     # Data range
#     nbins = 100
#     if zmin is None:
#         zmin = Z.min()
#     if zmax is None:
#         zmax = Z.max()

#     # Histogram
#     #n, bins, patches = plt.hist(Z.flatten(), bins=nbins, range=(zmin, zmax), facecolor='black', alpha=1)
#     #hst = plt.gca()
#     n, bins, patches = ax.hist(Z.flatten(), bins=nbins, range=(zmin, zmax), facecolor='black', alpha=1)
#     ax.set_ylabel('Count')

#     if coloredhisto:
#         # Normalizing from 0 to 1 for cmap display
#         bin_centers = 0.5 * (bins[:-1] + bins[1:])
#         color = bin_centers - min(bin_centers)
#         color /= max(color)

#         # Setting individual patch color
#         for clr, patch in zip(color, patches):
#             patch.set_facecolor(cmap(clr))
#             patch.set_edgecolor('None')

#     #  Colorbar display
#     if cmapdisplay:
#         # Creating a Mappable for colorbar
#         ## As the histogram is not a 'Mappable', we have to use ScalarMappable
#         ## to build the colorbar
#         norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)
#         #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#         sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
#         sm._A = []

#         # Colorbar creation
#         #cb = plt.colorbar(sm, orientation='horizontal', pad=0.05)
#         #cb.ax.xaxis.set_ticks_position('top')  # ticks on top
#         #cb.ax.xaxis.set_tick_params(direction='out')  # ticks outside

#         #see cax after
#         cb = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.075)
#         cb.ax.xaxis.set_ticks_position('top')  # ticks on top
#         cb.ax.xaxis.set_tick_params(direction='out')  # ticks outside

#         ### Seems ok now
#         # Histogram ticks in relative axes coordinates (0, 1)
#         ## Colorbar ticks position need to be in axes oordinates (0, 1).
#         ## Histogram ticks (zmin, zmax) are converted using mpl.transforms
#         ## (work with log scale). An additional dummy y-coordinate is added
#         ## to work with mpl.transforms.
#         #hist_ticks_data = np.vstack((hst.get_xticks(), np.zeros_like(hst.get_xticks()))).T  # (x,0)-like data coordinates
#         #hist_ticks_data = np.vstack((ax.get_xticks(), np.zeros_like(ax.get_xticks()))).T  # (x,0)-like data coordinates
#         #hist_ticks = cb.ax.transAxes.inverted().transform(cb.ax.transData.transform(hist_ticks_data))[:,0]  # color bar axes coordinates

#         # Colorbar ticks same as histogram ticks
#         #tick_locator = mpl.ticker.FixedLocator(hist_ticks)
#         #cb.locator = tick_locator
#         cb.update_ticks()
#         cb.ax.xaxis.set_ticklabels([])  # no ticks label displayed
#         cb.ax.set_xlabel(dataset.data.fields[-1])  # colorbar label display

#         colorbar = cb

#     else:
#         ax.set_xlabel(dataset.data.fields[-1])  # colorbar label display

#     if showtitle:
#         ax.set_title(dataset.name)
    
    
#     fig.tight_layout()

#     # Curve display
# #    plt.xlim(bins.min(), bins.max())
#     ax.set_xlim(bins.min(), bins.max())

#     # Saving figure to file
#     if filename is not None:
# ##       plt.savefig(filename, dpi=dpi, transparent=transparent)
# ##        plt.savefig(filename, dpi=dpi, transparent=transparent, facecolor=fig.get_facecolor(), edgecolor='none')
#         plt.savefig(filename, dpi=dpi, transparent=transparent, edgecolor='none')
       
#     return fig, colorbar


def getlimits(self, valfilt=False):
    ''' Get limits values of histogram.'''

    if valfilt or self.data.z_image is None:
        Z = self.data.values.T[2]
    else:
        Z = self.data.z_image

    return np.nanmin(Z.flatten()), np.nanmax(Z.flatten())

##
##def getlimits(self):
##    '''getting limits values of histogram.'''
##    Z = self.data.z_image
##    array = np.reshape(np.array(Z), (1, -1))
##
##    return np.nanmin(array), np.nanmax(array)
