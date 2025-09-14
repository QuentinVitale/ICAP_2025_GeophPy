# -*- coding: utf-8 -*-
"""
   geophpy.visualization.plot
   --------------------------

   Provides the main PlottingMixin for the Survey class, which handles all
   high-level plotting and visualization tasks by dispatching to specialized
   plotting functions.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

import matplotlib.pyplot as plt

# --- Import the low-level plotting functions ---
from . import track
from . import plot2D
from . import plot3D
from . import histo
from . import correlation

import numpy as np
import pandas as pd
from operator import itemgetter
import os
import json

from geophpy.core.json import JSON_Indent_Encoder
# from geophpy.visualization.plot2D import (plot_surface,
#                                      plot_contour,
#                                      plot_contourf,
#                                      plot_scatter,
#                                      plot_postmap)
import geophpy.__config__ as CONFIG

#from geophpy.visualization.plot2D import UNFILLED_MARKERS


class PlottingMixin:
    """
    Mixin containing the main user-facing plotting method for the Survey class.
    """

    # --- Data plots ---
    def plot(self, plot_type: str = '2D-SURFACE', **kwargs):
        """
        Generic plotting method for data that dispatches to a specific plot function.

        This is the main user-facing method for creating visualizations of the
        Survey object's data. It selects the appropriate plotting routine
        based on the `plot_type` argument.

        Parameters
        ----------
        plot_type : {'TRACK', '2D-SCATTER', '2D-SURFACE', '2D-CONTOUR', 'HISTOGRAM'}, optional
            The type of plot to generate. Defaults to '2D-SURFACE'.
        **kwargs
            Additional keyword arguments passed to the underlying plotting
            function. For example, `cmap='viridis'` for colormap, `bins=50`
            for histograms, or `marker='s'` for scatter plots.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated Matplotlib Figure object.
        ax : matplotlib.axes.Axes or array of Axes
            The generated Matplotlib Axes object(s).

        Raises
        ------
        ValueError
            If an unknown `plot_type` is specified or if the required data
            (e.g., gridded data for a surface plot) is not available.

        See Also
        --------
        geophpy.visualization.track.plot_track
        geophpy.visualization.plot2D.plot_scatter
        geophpy.visualization.plot2D.plot_surface
        geophpy.visualization.histo.plot

        Examples
        --------
        >>> # After loading data into a survey object
        >>> # fig, ax = survey.plot('TRACK')
        >>> # fig, ax = survey.plot('HISTOGRAM', bins=100)
        >>> # fig, ax = survey.plot('2D-SURFACE', cmap='terrain')
        
        """

        # The 'self' here is the Survey object instance
        plot_type = plot_type.upper()
        
        # --- Dispatch to the correct low-level plotting function ---
        if plot_type == 'TRACK':
            return plot_survey_track(self, **kwargs)
        
        elif plot_type == '2D-SCATTER':
            return plot2D.plot_scatter(self, **kwargs)
        
        elif plot_type == '2D-POSTMAP':
            return plot2D.plot_postmap(self, **kwargs)
            
        elif plot_type == '2D-SURFACE':
            return plot2D.plot_surface(self, **kwargs)

        elif plot_type == '2D-CONTOUR':
            return plot2D.plot_contour(self, **kwargs)

        elif plot_type == '2D-CONTOURF':
            return plot2D.plot_contourf(self, **kwargs)
       
        elif plot_type == 'POINTS_HISTOGRAM':
            return histo.plot_points_histogram(self, **kwargs)

        elif plot_type == 'GRID_HISTOGRAM':
            return histo.plot_grid_histogram(self, **kwargs)
            
        else:
            valid_options = ['TRACK', '2D-SCATTER', '2D-POSTMAP', '2D-SURFACE', '2D-CONTOUR', 'HISTOGRAM']
            raise ValueError(f"Unknown plot_type: '{plot_type}'. "
                             f"Valid options are: {valid_options}")

    # --- Diagnostic/ helper plots. ---
    def plot_correlation_sum(self, **kwargs):
        """
        Plots the summed cross-correlation of all adjacent profiles.

        This is a high-level shortcut method for this specific diagnostic plot.
        
        """        
        # Call the low-level, decorated function
        return correlation.plot_correlation_sum(self, **kwargs)
    
    def plot_correlation_map(self, **kwargs):
        """
        Plots the profile-to-profile correlation map of the gridded data.

        This is a high-level shortcut method for this specific diagnostic plot.
        """
        # Call the low-level, decorated function
        return correlation.plot_correlation_map(self, cmap='Spectral_r', **kwargs)

    def plot_summary(self, **kwargs):
        """
        Creates a summary figure with multiple views of the survey data.

        This diagnostic plot shows the survey tracks, a scatter plot of the
        point data, the interpolated grid surface, and the grid data histogram
        in a single figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            An existing figure to plot on. If None, a new one is created.
        **kwargs
            Additional keyword arguments passed to the underlying plot calls.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : array of matplotlib.axes.Axes
        """

        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = plt.figure(figsize=(12, 10))
        else:
            fig.clf()

        # Create a 2x2 grid of subplots
        gs = fig.add_gridspec(2, 2)
        ax_postmap = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[0, 1])
        ax_surface = fig.add_subplot(gs[1, 0])
        ax_hist = fig.add_subplot(gs[1, 1])
        
        fig.suptitle(f"Summary for Survey: {self.name}", fontsize=16)

        # --- Use our existing shortcut methods to populate each subplot ---
        # Plot 1: Survey Tracks (Postmap)
        self.plot_postmap(ax=ax_postmap, **kwargs)
        ax_postmap.set_title("Survey Tracks")
    
        # Plot 1: Survey Tracks (Postmap)
 #       self.plot_track(ax=ax_track, **kwargs)
  #      ax_track.set_title("Survey Tracks")

        # Plot 2: Scatter Plot
        self.plot_scatter(ax=ax_scatter, **kwargs)
        ax_scatter.set_title("Scatter Plot of Point Data")

        # Plot 3 & 4: Grid and Grid Histogram (if data is interpolated)
        if self.grid:
            self.plot_surface(ax=ax_surface, **kwargs)
            ax_surface.set_title("Gridded Surface")
            
            self.plot_grid_histogram(ax=ax_hist, colored=False, **kwargs)
            ax_hist.set_title("Histogram of Gridded Data")
        else:
            ax_surface.text(0.5, 0.5, 'No Gridded Data', ha='center', va='center')
            ax_surface.set_title("Gridded Surface")
            self.plot_points_histogram(ax=ax_hist, colored=False, **kwargs)
            ax_hist.set_title("Histogram of Point Data")

        fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
        return fig, fig.axes

    # --- Shortcut Methods for User Convenience ---

    # def plot_track(self, **kwargs):
    #     """
    #     Plots the survey tracks (postmap). A shortcut for .plot('TRACK').
    #     """
    #     return self.plot('TRACK', **kwargs)

    def plot_points_histogram(self, **kwargs):
        """
        Plots a histogram of the ungridded point data.
        A shortcut for .plot('POINTS_HISTOGRAM').
        """
        return self.plot('POINTS_HISTOGRAM', **kwargs)

    def plot_grid_histogram(self, **kwargs):
        """
        Plots a histogram of the gridded data.
        A shortcut for .plot('GRID_HISTOGRAM').
        """
        return self.plot('GRID_HISTOGRAM', **kwargs)
    
    def plot_postmap(self, **kwargs):
        """
        Plots the ungridded data posirions as a 2D scatter plot.
        A shortcut for .plot('2D-POSTMAP').
        """
        return self.plot('2D-POSTMAP', **kwargs)
    
    def plot_scatter(self, **kwargs):
        """
        Plots the ungridded data as a 2D scatter plot.
        A shortcut for .plot('2D-SCATTER').
        """
        return self.plot('2D-SCATTER', **kwargs)

    def plot_surface(self, **kwargs):
        """
        Plots the gridded data as a 2D surface image.
        A shortcut for .plot('2D-SURFACE').
        """
        return self.plot('2D-SURFACE', **kwargs)
    
    def plot_contour(self, **kwargs):
        """
        Plots the gridded data as a 2D contour map.
        A shortcut for .plot('2D-CONTOUR').
        """
        return self.plot('2D-CONTOUR', **kwargs)

    def plot_contourf(self, **kwargs):
        """
        Plots the gridded data as a filled 2D contour map.
        A shortcut for .plot('2D-CONTOURF').
        """
        return self.plot('2D-CONTOURF', **kwargs)
    
    def save_figure(self, fig, filename: str, **kwargs):
        """
        Saves a Matplotlib figure to a file with custom options.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object that you want to save.
        filename : str
            The name and path of the output file (e.g., 'my_map.png').
        **kwargs
            Additional keyword arguments for saving options.

        Keyword Arguments
        -----------------
        axisdisplay : bool, optional
            If False, the plot is saved without axis, labels, or titles,
            cropping the image to the data frame. Defaults to True.
        cmapdisplay : bool, optional
            If False, the colorbar is not included in the tight layout,
            which can affect cropping. Defaults to True.
        dpi : int, optional
            The resolution of the saved file in dots per inch.
            Defaults to 600.
        transparent : bool, optional
            If True, the figure background will be transparent.
            Defaults to True.
        
        Notes
        -----
        This method uses Matplotlib's `savefig` function. Most of its
        keyword arguments (like `facecolor`, `edgecolor`) can also be passed
        through `**kwargs`.
        """
        # Retrieve save options from kwargs with sensible defaults
        axisdisplay = kwargs.get('axisdisplay', True)
        cmapdisplay = kwargs.get('cmapdisplay', True)
        dpi = kwargs.get('dpi', 600)
        transparent = kwargs.get('transparent', True)

        if filename:
            # If both axes and colorbar are hidden, save a tightly cropped image
            if not axisdisplay and not cmapdisplay:
                fig.savefig(filename, dpi=dpi, transparent=transparent, 
                            bbox_inches='tight', pad_inches=0)
            else:
                fig.savefig(filename, dpi=dpi, transparent=transparent,
                            bbox_inches='tight')
            
            print(f"Figure saved to {filename}")
        else:
            print("Warning: No filename provided. Figure not saved.")
    


# list of plot types available
plottype_list = ['2D-SCATTER', '2D-SURFACE', '2D-CONTOUR', '2D-CONTOURF', '2D-POSTMAP']

# list of interpolations available
interpolation_list = ['none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'sinc']

# list of picture format files available
pictureformat_list = ['.eps', '.jpeg', '.jpg', '.pdf', '.pgf', '.png', '.ps', '.raw', 
                      '.rgba', '.svg', '.svgz', '.tif', '.tiff']


def gettypelist():
   ''' Return the list of available plot types. '''

   return plottype_list


def getinterpolationlist():
   ''' Return the list of available interpolations for data display. '''

   return interpolation_list


def getpictureformatlist():
    ''' Return the list of available picture format. '''

    return pictureformat_list


# def plot(dataset, plottype, cmapname, creversed=False, fig=None, filename=None, 
#          cmmin=None, cmmax=None, interpolation='bilinear', levels=None, cmapdisplay=True, 
#          axisdisplay=True, labeldisplay=False, pointsdisplay=False, dpi=None, transparent=False, 
#          logscale=False, rects=None, points=None, marker='+', markersize=None):
#     ''' Dataset display.

#     cf. `:meth:`~geophpy.dataset.DataSet.plot`

#     '''

#     ret = None

#     if filename is None:
#         success = True
#     else:
#         success = ispictureformat(filename)

#     if success:
#         # 2D-SURFACE
#         if plottype.upper() in ['2D-SURFACE', 'SURFACE', 'SURF', 'SF']:

#             # ### display options
#             # filename = kwargs.get('filename', None)
#             # axisdisplay = kwargs.get('axisdisplay', True)
#             # cmapdisplay = kwargs.get('cmapdisplay', True)
#             # dpi = kwargs.get('dpi', 600)
#             # transparent = kwargs.get('transparent', True)

#             # dataset,
#             # cmapname,
#             # creversed = kwargs.get('creversed', False)
#             # fig = kwargs.get('fig', None)
#             # filename = kwargs.get('filename', None)
#             # cmmin = kwargs.get('cmmin', None)
#             # cmmax = kwargs.get('cmmax', None)
#             # interpolation = kwargs.get('interpolation', 'bilinear')
#             # cmapdisplay = kwargs.get('cmapdisplay', True)
#             # axisdisplay = kwargs.get('axisdisplay', True)
#             # labeldisplay = kwargs.get('labeldisplay', False)
#             # pointsdisplay = kwargs.get('pointsdisplay', False)
#             # dpi = kwargs.get('dpi', None)
#             # transparent = kwargs.get('transparent', False)
#             # logscale = kwargs.get('logscale', False)
#             # rects = kwargs.get('rects', None)
#             # points = kwargs.get('points', None)
#             # marker = kwargs.get('marker', '+')
#             # markersize = kwargs.get('markersize', None)

#             fig, cmap = plot_surface(dataset, cmapname,
#                                             creversed=creversed, fig=fig, filename=filename,
#                                             cmmin=cmmin, cmmax=cmmax,
#                                             interpolation=interpolation,
#                                             cmapdisplay=cmapdisplay, axisdisplay=axisdisplay,
#                                             labeldisplay=labeldisplay, pointsdisplay=pointsdisplay,
#                                             dpi=dpi, transparent=transparent,
#                                             logscale=logscale, rects=rects, points=points, marker=marker)

#         # 2D-CONTOUR
#         elif plottype.upper() in ['2D-CONTOUR', 'CONTOUR', 'CONT', 'CT']:
#             fig, cmap = plot_contour(dataset, cmapname, levels=levels, creversed=creversed, 
#                                      fig=fig, filename=filename, cmmin=cmmin, cmmax=cmmax, 
#                                      cmapdisplay=cmapdisplay, axisdisplay=axisdisplay, 
#                                      labeldisplay=labeldisplay, pointsdisplay=pointsdisplay, 
#                                      dpi=dpi, transparent=transparent, logscale=logscale, 
#                                      rects=rects, points=points, marker=marker)

#         # 2D-CONTOUR (Filled)
#         elif plottype.upper() in ['2D-CONTOURF', 'CONTOURF', 'CONTF', 'CF']:
#             fig, cmap = plot_contourf(dataset, cmapname, levels=levels, creversed=creversed, 
#                                       fig=fig, filename=filename, cmmin=cmmin, cmmax=cmmax, 
#                                       cmapdisplay=cmapdisplay, axisdisplay=axisdisplay, 
#                                       labeldisplay=labeldisplay, pointsdisplay=pointsdisplay, 
#                                       dpi=dpi, transparent=transparent, logscale=logscale, 
#                                       rects=rects, points=points, marker=marker)

#         # 2D-SCATTER
#         elif plottype in ['2D-SCATTER', 'SCATTER', 'SCAT', 'SC']:
#             fig, cmap = plot_scatter(dataset, cmapname, creversed=creversed, fig=fig, 
#                                      filename=filename, cmmin=cmmin, cmmax=cmmax, 
#                                      cmapdisplay=cmapdisplay, axisdisplay=axisdisplay, 
#                                      labeldisplay=labeldisplay, dpi=dpi, transparent=transparent, 
#                                      logscale=logscale, rects=rects, markersize=markersize)

#         # 2D-POSTMAP
#         elif plottype.upper() in ['2D-POSTMAP', 'POSTMAP', 'POST', 'PM']:
#             fig, cmap = plot_postmap(dataset, fig=fig, filename=filename, axisdisplay=axisdisplay, 
#                                      labeldisplay=labeldisplay, dpi=600, transparent=transparent, 
#                                      marker=marker, markersize=markersize)

#         # '3D-SURFACE'
#         else:
#             fig, cmap = plot3D.plot_surface(dataset, cmapname, creversed, fig, filename, 
#                                             cmmin, cmmax, cmapdisplay, axisdisplay, dpi, 
#                                             transparent, logscale)

#     else:
#         fig = None
#         cmap = None

#     return fig, cmap


def completewithnan(dataset):
   """
   completes a data set with nan values in empty space
   """
   lbegin = 0
   nanpoints=[]

   x = dataset.data.values[0][0]
   for l in range(len(dataset.data.values)):
      if (dataset.data.values[l][0] != x):
         # treats the previous profile
         lend = l
         delta_y = _profile_mindelta_get(dataset.data.values[lbegin:lend])
         _profile_completewithnan(dataset.data.values[lbegin:lend],delta_y,
                                  dataset.info.y_min, dataset.info.y_max, nanpoints)
         lbegin = l
         # if the previous profile is not next, adds nan profiles to avoid linearisations on the plot
         if ((dataset.data.values[l][0] - x) > dataset.info.x_mindelta):
            # nan profile created just after the previous profile
            _profile_createwithnan(nanpoints, x+(dataset.info.x_mindelta/10),
                                   dataset.info.y_min, dataset.info.y_max)
            # nan profile created just before the current profile
            _profile_createwithnan(nanpoints, dataset.data.values[l][0]-(dataset.info.x_mindelta/10),
                                   dataset.info.y_min, dataset.info.y_max)
         x = dataset.data.values[l][0]

   # treats the last profile
   delta_y = _profile_mindelta_get(dataset.data.values[lbegin:])
   _profile_completewithnan(dataset.data.values[lbegin:],delta_y,dataset.info.y_min, 
                            dataset.info.y_max, nanpoints)

   # adds the nan points to the values array
   # converts in a numpy array, sorted by column 0 then by column 1.
   try:
      dataset.data.values = np.array(sorted(np.concatenate((dataset.data.values, nanpoints)), 
                                            key=itemgetter(0,1)))
   except:
      pass


def ispictureformat(picturefilename):
   '''
   Detects if the picture format is available
   '''
   splitedfilename = os.path.splitext(picturefilename)
   extension = (splitedfilename[-1]).lower()

   ispictureformat = False
   for ext in pictureformat_list:
      if (extension == ext):
         ispictureformat = True
         break

   return ispictureformat


def _profile_completewithnan(profile,delta_y,dataset_ymin, dataset_ymax, nanpoints):
   """
   completes a profile with nan value in empty space
   """
   coef = 20

   if (dataset_ymin != None):             # if ymin is not equal to None
      profile[0][1] = dataset_ymin        # normalizes all begins of profiles with the same value

   yprev = profile[0][1]                  # initialises previous y
   valprev = profile[0][2]                # initialises previous value

   for p in profile:                      # for each point in the profile
      y = p[1]                            # y is the column 1 value
      if (((y-yprev) > coef*delta_y) or np.isnan(p[2]) or np.isnan(valprev)):
         yn = yprev + ((y-yprev)/1000)
         while (yn < y):
            nanpoints.append([p[0], yn, np.nan])
            yn = yn + ((y-yprev)/10)
         nanpoints.append([p[0], y - ((y-yprev)/1000), np.nan])
      yprev = y                           # saves the y value as the previous for profile to be
      valprev = p[2]                      # saves the value as the previous for profile to be

   # treats the last point of profile
   if ((dataset_ymax-yprev) > coef*delta_y):
      y = yprev + ((dataset_ymax-yprev)/1000)
      while (y < dataset_ymax):
         nanpoints.append([p[0], y, np.nan])
         y = y + ((dataset_ymax-yprev)/10)



def _profile_createwithnan(nanpoints, x,dataset_ymin, dataset_ymax):
   """
   creates a profile with nan values
   """

   if ((dataset_ymin != None) and (dataset_ymax != None)):
      for y in range(int(dataset_ymin), int(dataset_ymax)):
         nanpoints.append([x, y, np.nan])


def _profile_mindelta_get(profile):
   '''
   Get Minimal Delta between 2 values in a same profile.
   '''
   l = 0
   deltamin = 0
   while (deltamin == 0):
      deltamin = abs(profile[l+1][1] - profile[l][1])
      l = l+1

   yprev = profile[l][1]

   for p in profile[l+1:]:
      delta = abs(p[1] - yprev)
      if ((delta != 0) and (delta < deltamin)):
         deltamin = delta

   return deltamin


def coord2rec(xmin, xmax, ymin, ymax):
    '''
    Convert rectangle extent coordinates to bottom left corner,
    width and height coordinates.

    '''

    x = min(xmin,xmax)
    y = min(ymin,ymax)
    w = abs(xmax-xmin)
    h = abs(ymax-ymin)

    return [x, y, w, h]


def extents2rectangles(extentlist):
    '''
    Convert a list of rectangle extent coordinates to a list of
    bottom left corner width and height coordinates.

    '''

    rectanglelist = []

    for extent in extentlist:
       rectanglelist.append(coord2rec(*extent))

    return rectanglelist


# def _init_figure(fig=None):
#     ''' Clear the given figure or return a new one and initialize an ax. '''

#     # First display
#     if fig is None:
#         fig = plt.figure()
#         #fig = Figure()
#         #FigureCanvas(fig)

#     # Existing display
#     else:
#         fig = plt.figure(fig.number, clear=True)
#         #fig.clf()

#     # Axes initialization
#     ax = fig.add_subplot(111)

#     return fig, ax


def grid_plot(don,grid_final,ncx,ncy,ext,pxy,nc_data,nb_ecarts,nb_res,output_file=None,
              sep='\t',plot_pts=False,matrix=False):
    """
    Plot the result of ``interp_grid``.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    grid_final : np.ndarray (dim 3) of float
        For each data column, contains the grid values after the chosen method.
    ncx : list of str
        Names of every X columns.
    ncy : list of str
        Names of every Y columns.
    ext : [float, float, float, float]
        Extend of the grid. Contains ``[min_X, max_X, min_Y, max_Y]``.
    pxy : list of 4 floats
        Steps of the grid for each axis. Contains ``[pas_X, pas_Y]``.
    nc_data : list of str
        Names of every Z columns (actual data).
    nb_ecarts : int
        Number of X and Y columns. The number of coils.
    nb_res : int
        The number of data per coil.
    ``[opt]`` output_file : ``None`` or str, default : ``None``
        Name of output file. If ``None``, do not save.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` plot_pts : bool, default : ``False``
        Plots the raw points on top of the grid.
    ``[opt]`` matrix : bool, default : ``False``
        Whether the output should be saved as a dataframe or as the custom 'matrix' format.
    
    Returns
    -------
    none, but plots the final grid and saves the figures.\n
    * ``output_file = None``
        Nothing more
    * ``output_file != None``
        Saves the grid in a dataframe or in the custom 'matrix format'.
    
    Notes
    -----
    Subfunction of ``interp_grid``\n
    Is not called if heatmap was activated.
    
    See also
    --------
    ``interp_grid, df_to_matrix``
    """
    nb_data = len(nc_data)
    
    # Calcule des dimensions de la grille
    pas_X = (ext[1]-ext[0])/pxy[0]
    pas_Y = (ext[3]-ext[2])/pxy[1]
    gridx = [ext[0] + pas_X*(i+0.5) for i in range(pxy[0])]
    gridy = [ext[2] + pas_Y*(j+0.5) for j in range(pxy[1])]
    
    # Cas de plusieurs voies
    try:
        int(ncx[0][-1])
        col_x = ""
        col_y = ""
        for e in range(nb_ecarts):
            col_x += ncx[e]+"|"
            col_y += ncy[e]+"|"
        col_x = col_x[:-1]
        col_y = col_y[:-1]
    # Cas d'une seule voie
    except:
        col_x = ncx[0]
        col_y = ncy[0]
    # Construction du dataframe représentatif de la grille
    don_f = pd.DataFrame({col_x: np.array([[i for j in gridy] for i in gridx]).round(CONFIG.prec_coos).flatten(),
                          col_y: np.array([[j for j in gridy] for i in gridx]).round(CONFIG.prec_coos).flatten()})
    for n in range(nb_data):
        don_temp = pd.DataFrame({nc_data[n]: grid_final[n].flatten().round(CONFIG.prec_data)})
        don_f = pd.concat([don_f, don_temp], axis=1)
    
    # Résultat enregistré dans un fichier (option)
    if output_file != None:
        # Format matrice
        if matrix:
            grid_not_np = []
            for n in range(nb_data):
                grid_not_np.append(grid_final[n].T.round(CONFIG.prec_data).tolist())
            try:
                grid_save = {"grid" : grid_not_np, "ext" : ext, "pxy" : pxy, "step" : [pas_X,pas_Y],
                             "ncx" : ncx.to_list(), "ncy" : ncy.to_list(), "ncz" : nc_data.to_list()}
            except AttributeError:
                grid_save = {"grid" : grid_not_np, "ext" : ext, "pxy" : pxy, "step" : [pas_X,pas_Y],
                             "ncx" : ncx, "ncy" : ncy, "ncz" : nc_data}
            with open(output_file, "w") as f:
                json.dump(grid_save, f, indent=None, cls=JSON_Indent_Encoder)
        # Format dataframe
        else:
            don_f.to_csv(output_file, index=False, sep=sep)
    
    # Plot du résultat
    plt.style.use('_mpl-gallery-nogrid')
    for i in range(nb_ecarts):
        fig,ax = plt.subplots(nrows=1,ncols=nb_res,figsize=(nb_res*CONFIG.fig_width//2,CONFIG.fig_height),\
                              squeeze=False)
        
        for j in range(nb_res):
            Q_l = [z for z in grid_final[i*nb_res+j].flatten() if z == z]
            Q = np.quantile(Q_l,[0.05,0.95])
            ims = ax[0][j].imshow(grid_final[i*nb_res+j].T, origin='lower', cmap='cividis', \
                                  vmin = Q[0], vmax=Q[1], extent=ext)
            ax[0][j].set_title(nc_data[i*nb_res+j])
            ax[0][j].set_xlabel(ncx[i])
            ax[0][j].set_ylabel(ncy[i])
            ax[0][j].set_aspect('equal')
            plt.colorbar(ims,ax=ax[0][j])
            if plot_pts:
                try:
                    ax[0][j].scatter(don[ncx[i]],don[ncy[i]],marker='s',c=don["Num fich"],s=5)
                except:
                    ax[0][j].scatter(don[ncx[i]],don[ncy[i]],marker='s',s=5)
        plt.show(block=False)
        # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
        # pour accélérer la vitesse de l'input
        plt.pause(CONFIG.fig_render_time)
    return don_f


def plot_data(file,col_x=None,col_y=None,col_z=None,sep='\t'):
    """ [TA]\n
    Plots raw data from dataframe.
    
    Parameters
    ----------
    file : str or dataframe
        Dataframe file or loaded dataframe.
    ``[opt]`` col_x : ``None`` or list of int, default : ``None``
        Index of every X coordinates columns. If ``None``, is set to ``[0]``.
    ``[opt]`` col_y : ``None`` or list of int, default : ``None``
        Index of every Y coordinates columns. If ``None``, is set to ``[1]``.
    ``[opt]`` col_z : ``None`` or list of int, default : ``None``
        Index of every Z coordinates columns (actual data). If ``None``, takes every column that is not X nor Y.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    
    Notes
    -----
    Also handles column names with multiple substrings splitted by '|', which correspond to the grid dataframe format.
    
    Raises
    ------
    * File not found.
    * Wrong separator.
    * Column is not numeric.
    """
    if isinstance(file,str):
        try:
            # Chargement des données
            df = pd.read_csv(file, sep=sep)
        except FileNotFoundError:
            raise FileNotFoundError("File {} not found.".format(file))
    else:
        df = file
    
    if len(df.columns) <= 1:
        raise FileNotFoundError("Unable to read {}, is the separator {} correct ?".format(file,repr(sep)))
    
    # Gestion des colonnes
    # Si une colonne contient des "/", on suppose qu'elle a été générée par une grille
    multi_col = False
    if col_x == None:
        col_x = [0]
    ncx = df.columns[col_x]
    ncx_t = df.columns[col_x]
    rx = ncx[0].split("|")
    if len(rx) != 1:
        ncx = rx
        multi_col = True
    if col_y == None:
        col_y = [1]
    ncy = df.columns[col_y]
    ncy_t = df.columns[col_y]
    ry = ncy[0].split("|")
    if len(ry) != 1:
        ncy = ry
        multi_col = True
    if col_z == None:
        col_z = df.columns.drop(ncx_t)
        nc_data = col_z.drop(ncy_t)
    else:
        nc_data = df.columns[col_z]
    nb_data = len(nc_data)
    nb_ecarts = len(ncx)
    nb_res = max(1, nb_data//nb_ecarts)
    
    # Affichage pour chaque voie
    for e in range(nb_ecarts):
        fig,ax=plt.subplots(nrows=1,ncols=nb_res,figsize=(nb_res*CONFIG.fig_width//2,CONFIG.fig_height),squeeze=False)
        if multi_col:
            X = df[ncx_t]
            Y = df[ncy_t]
        else:
            X = df[ncx_t[e]]
            Y = df[ncy_t[e]]
        for r in range(nb_res):
            n = e*nb_res + r
            Z = df[nc_data[n]]
            # Le quantile n'a pas de sens si certaines données ne sont pas numériques
            try:
                Q5,Q95 = Z.dropna().quantile([0.05,0.95])
            except TypeError:
                raise TypeError("Column {} is not numeric or has NaN.".format(nc_data[n]))
            col = ax[0][r].scatter(X,Y,marker='s',c=Z,cmap='cividis',s=6,vmin=Q5,vmax=Q95)
            plt.colorbar(col,ax=ax[0][r])
            ax[0][r].title.set_text(nc_data[n])
            ax[0][r].set_xlabel(ncx[e])
            ax[0][r].set_ylabel(ncy[e])
            ax[0][r].set_aspect('equal')
        plt.show(block=False)
        # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
        # pour accélérer la vitesse de l'input
        plt.pause(CONFIG.fig_render_time) 


def plot_grid(file):
    """ [TA]\n
    Plots raw data from 'matrix' format.
    
    Parameters
    ----------
    file : str
        Matrix file.
    
    Raises
    ------
    * File not found.
    * File is not a JSON.
    """
    try:
        with open(file, 'r') as f:
            # Chargement des données
            grid_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("File {} not found.".format(file))
    except json.JSONDecodeError:
        raise FileNotFoundError("File {} is not a json.".format(file))
    
    # Sélection des positions et de la grille
    nb_ecarts = len(grid_dict["ncx"])
    nb_res = len(grid_dict["ncz"])//nb_ecarts
    grid = np.array(grid_dict["grid"])
    
    # Affichage pour chaque voie
    plt.style.use('_mpl-gallery-nogrid')
    for e in range(nb_ecarts):
        fig,ax = plt.subplots(nrows=1,ncols=nb_res,figsize=(nb_res*CONFIG.fig_width//2,CONFIG.fig_height),squeeze=False)
        
        for r in range(nb_res):
            n = e*nb_res+r
            Q_l = [z for z in grid[n].flatten() if z == z]
            Q = np.quantile(Q_l,[0.05,0.95])
            ims = ax[0][r].imshow(grid[n], origin='lower', cmap='cividis', vmin = Q[0], vmax=Q[1], extent=grid_dict["ext"])
            ax[0][r].set_title(grid_dict["ncz"][n])
            ax[0][r].set_xlabel(grid_dict["ncx"][e])
            ax[0][r].set_ylabel(grid_dict["ncy"][e])
            ax[0][r].set_aspect('equal')
            plt.colorbar(ims,ax=ax[0][r])
        plt.show(block=False)
        # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
        # pour accélérer la vitesse de l'input
        plt.pause(CONFIG.fig_render_time)


def plot_pos(file,sep='\t'):
    """
    Plots positions of each coil.
    
    Parameters
    ----------
    file : str or dataframe
        Dataframe file or loaded dataframe.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    
    Notes
    -----
    Coil position columns must be named as such : ``[X/Y] + _int_ + [coil_id]``.\n
    Output figure can be heavy.
    
    Raises
    ------
    * File not found.
    * Wrong separator.
    * No positions for each coil.
    """
    if isinstance(file,str):
        try:
            # Chargement des données
            df = pd.read_csv(file, sep=sep)
        except FileNotFoundError:
            raise FileNotFoundError('File "{}" not found.'.format(file))
    else:
        df = file
    
    if len(df.columns) <= 1:
        raise OSError("File '{}' does not have '{}' as its separator.".format(file,repr(sep)))
    
    # Solution du pauvre : on propose jusqu'à 8 couleurs pour les différentes voies
    color = ["blue","green","orange","magenta","red","cyan","black","yellow"]
    ncx = []
    ncy = []
    # Va être égal au nombre de voies
    cpt = 0
    while True: # Programmassion is my passion
        # Nom des positions des différentes voies
        new_x = "X_int_"+str(cpt+1)
        new_y = "Y_int_"+str(cpt+1)
        try:
            df[new_x]
            ncx.append(new_x)
            ncy.append(new_y)
        except KeyError:
            break
        cpt += 1
    if cpt == 0:
        raise KeyError("channels positions should be named as \"X_int_1\" [...]")
    
    # Affichage des positions
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(CONFIG.fig_width,CONFIG.fig_height))
    # Trait noir reliant les positions d'un même point
    for index, row in df.iterrows():
        ax.plot(row[ncx],row[ncy],'-k')
    # Point de la couleur de sa voie
    for i in range(cpt):
        ax.plot(df[ncx[i]],df[ncy[i]],'o',color=color[i%8],label="Coil "+str(i+1))
    ax.set_title("Coil position")
    ax.set_xlabel("X_int")
    ax.set_ylabel("Y_int")
    ax.set_aspect('equal')
    plt.legend()
    plt.show(block=False)
    # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
    # pour accélérer la vitesse de l'input
    plt.pause(CONFIG.fig_render_time)