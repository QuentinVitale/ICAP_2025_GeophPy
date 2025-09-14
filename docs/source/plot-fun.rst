.. _chap-plot-fun-geophpy:

Plotting functions
******************

The different plotting functions are based on `Matplotlib`_ and will return a Matplotlib figure object. 
The dataset can hence be plotted using commands of the form: 

   >>> fig = dataset.PlotFunction(option1=1, option2=True)
   >>> fig.show()

Each plot types can be stored into a new or existing figure object with the :meth:`~geophpy.dataset.DataSet.plot` method of the :class:`~geophpy.dataset.DataSet` class: 
 
   >>> # Plot in a new figure 
   >>> fig = dataset.PlotType(option1=1, ...)
   >>> fig.show()
 
   >>> # Plot in an existing figure 
   >>> fig = plt.figure()
   >>> dataset.PlotType(fig=fig)
   >>> fig.show()

and/or saved into a file providing a file name to the :meth:`~geophpy.dataset.DataSet.plot` method: 

    >>> # Saving plot in a file
    >>> dataset.PlotType(filename='MyPlot.png')

The different available plotting functions are listed in the sections below.

.. _`Matplotlib`: https://matplotlib.org/

Map plotting
============

All map displays are done through the :meth:`~geophpy.dataset.DataSet.plot` method of the :class:`~geophpy.dataset.DataSet` class:

>>> fig, cmap = dataset.plot(plottype='2D-SURFACE', ...)

The :meth:`~geophpy.dataset.DataSet.plot` method returns both a :class:`~matplotlib.figure.Figure` and the associated :class:`~matplotlib.colorbar.Colorbar` so that the typical display sequence is:

    >>> # Classic display command
    >>> fig, cmap = dataset.plot(plottype='2D-SURFACE', ...)
    >>> fig.show()

    >>> # One-line command
    >>> dataset.plot(plottype='2D-SURFACE', ...)[0].show()

Interpolation
-------------

To be displayed as a map, most of the plot types need a gridded dataset (i.e. :attr:`~geophpy.dataset.Data.z_image`). 
Use the :meth:`~geophpy.dataset.DataSet.interpolate` method of the  :class:`~geophpy.dataset.DataSet` class to grid the dataset:

    >>> # Linear interpolation 
    >>> dataset.interpolate(interpolation='linear')

    >>> # No interpolation
    >>> dataset.interpolate(interpolation='none')

To get the list of the available plotting interpolations use the command:

    >>> interpolation_getlist()
    ['none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'sinc']

see :ref:`chap-dataset-op-geophpy`

Color maps
----------

The list of the available color maps can be obtained with the command: 

   >>> colormap_getlist()
   ['Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys',
    'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuOr', 'PuRd', 'Purples',
    'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'Wistia',
    'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 
    'bone', 'bwr', 'copper', 'gist_earth', 'gist_gray', 'gist_heat',
    'gist_yarg', 'gnuplot', 'gray', 'hot', 'hsv', 'jet', 'ocean', 'pink',
    'spectral', 'terrain'] 


You can plot a specific color map using the following command: 

   >>> # Plotting a color map
   >>> fig = colormap_plot('BrBG')
   >>> fig.show()

   >>> # Plotting the reverse color map
   >>> colormap_plot('BrBG', creversed=True).show()

+------------------------------------------------+-------------------------------------------------------+
| .. figure:: _static/figPlotColorMapBrBG.png    | .. figure:: _static/figPlotColorMapBrBG_r.png         |
|    :align: center                              |    :align: center                                     |
|                                                |                                                       |
|    Plot -  BrBG color map.                     |    Plot -  BrBG reverse color map.                    |
+------------------------------------------------+-------------------------------------------------------+

And you can save it to a file using: 

   >>> # Saving the color map as a .png file
   >>> cmapfilename = 'colormap.png'
   >>> colormap_plot(cmname, filename=cmapfilename)

You can also manually add "_r" (instead of using ``creversed=True``) to the color map name to obtain the reversed color map: 

   >>> cmapnb = 6
   >>> cmaplist = colormap_getlist()
   >>> for i in range (cmapnb):
   >>>     colormap_plot(cmaplist[i-1], filename="CMAP_" + str(i) + ".png")


+-------------------------------------------+-------------------------------------------+-------------------------------------------+-------------------------------------------+-------------------------------------------+-------------------------------------------+
|                 Blues                     |                   BrBG                    |                   BuGn                    |                   BuPu                    |                   CMRmap                  |                   GnBu                    |	
|                                           |                                           |                                           |                                           |                                           |                                           |
| .. figure:: _static/figPlotColorMap_0.png | .. figure:: _static/figPlotColorMap_1.png | .. figure:: _static/figPlotColorMap_2.png | .. figure:: _static/figPlotColorMap_3.png | .. figure:: _static/figPlotColorMap_4.png | .. figure:: _static/figPlotColorMap_5.png |	
|    :align: center                         |    :align: center                         |    :align: center                         |    :align: center                         |    :align: center                         |    :align: center                         |
|                                           |                                           |                                           |                                           |                                           |                                           |
|                                           |                                           |                                           |                                           |                                           |                                           |
+-------------------------------------------+-------------------------------------------+-------------------------------------------+-------------------------------------------+-------------------------------------------+-------------------------------------------+

or you can build figure and plot objects to display them in a new window:

    >>> cm_fig = None
    >>> first_time = True
    >>> for cmapname in cmaplist:
    >>>     cm_fig = colormap_plot(cmapname, fig=cm_fig)
    >>>     if (first_time == True):
    >>>         fig.show()
    >>>         first_time = False
    >>>     fig.draw()

Map types
---------

It is possible to plot the dataset as a map using different plot types. 
To display the dataset as a map simply use the :meth:`~geophpy.dataset.DataSet.plot` method of the :class:`~geophpy.dataset.DataSet` class:

   >>> dataset.plot(plottype=PlotType, option1=..., option2=...)

The list of the a available plot types can be obtained by the command:

   >>> from geophpy.dataset import plottype_getlist
   >>> plottype_getlist()
   ['2D-SCATTER', '2D-SURFACE', '2D-CONTOUR', '2D-CONTOURF', '2D-POSTMAP']

You can overlays the measured data points to the map using the comand:

   >>> # Data points display
   >>> dataset.plot(plottype='2D-SURFACE', pointsdisplay=True)

**2-D scatter and postmap plots**

Display your raw data as a scatter plot to get a quick look at it or  your data point position as a postmap: 

   >>> dataset.info.cmapname = 'gist_ncar'
   >>> fig, cmap = dataset.plot(plottype='2D-SCATTER', cmmin=-20, cmmax=20)
   >>> fig.show()

   >>>  fig, cmap = dataset.plot(plottype='2D-POSTMAP')
   >>> fig.show()

   +------------------------------------------------+-------------------------------------------------------+
   | .. figure:: _static/figPlotScatter.png         | .. figure:: _static/figQuickStartPostmap.png          |
   |    :align: center                              |    :align: center                                     |
   |    :height: 6cm                                |    :height: 6cm                                       | 
   |                                                |                                                       |
   |    Plot - Displaying data as scatter plot.     |    Plot - Displaying data points position.            |
   |                                                |                                                       |
   +------------------------------------------------+-------------------------------------------------------+

**2-D surface plot**

You can plot a dataset as a 2-D surface map using different interpolation for the display: 

   >>> # Dataset 2-D surface plot
   >>> dataset.plot('2D-SURFACE', 'gray_r', plot.png,
   interpolation='bilinear', transparent=True, dpi=400)

   >>> dataset.plot('2D-SURFACE', 'gray_r', plot.png,
   interpolation='bicubic', transparent=True, dpi=400)

   +---------------------------------------------------+-------------------------------------------------------+
   | .. figure:: _static/figCarto4.png                 | .. figure:: _static/figCarto5.png                     |
   |    :align: center                                 |    :align: center                                     |
   |    :height: 6cm                                   |    :height: 6cm                                       | 
   |                                                   |                                                       |
   |    Plot - Map using the 'bilinear' interpolation  |    Plot - Map using the 'bilinear' interpolation.     |
   |                                                   |                                                       |
   +---------------------------------------------------+-------------------------------------------------------+


**2-D Contour plots**

   You can plot a dataset as a 2-D (filled or unfilled) contour plot using: 

   >>> # Dataset 2-D contour plot
   >>> fig, cmap = dataset.plot('2D-CONTOUR', levels=100, cmmin=-20, cmmax=20)
   >>> fig.show()

   >>> # Dataset 2-D Filled-contour plot
   >>> fig, cmap = dataset.plot('2D-CONTOURF', levels=100, cmmin=-20, cmmax=20)
   >>> fig.show()

   +------------------------------------------------+-------------------------------------------------------+
   | .. figure:: _static/figPlotContour.png         | .. figure:: _static/figPlotContourF.png               |
   |    :align: center                              |    :align: center                                     |
   |    :height: 6cm                                |    :height: 6cm                                       | 
   |                                                |                                                       |
   |    Plot - Displaying data as a contour plot.   |    Plot - Displaying data as a filled contour plot.   |
   +------------------------------------------------+-------------------------------------------------------+

Plot options
------------

**Axis, label and color bar display**

You can customize the display by enabling/disabling the axis, label and color bar display:

   >>> # Custom axis, label and color map display
   >>> dataset.plot(plottype='2D-SURFACE', labeldisplay=False)
   >>> dataset.plot(plottype='2D-SURFACE', axisdisplay=False, cmapdisplay=False)

   +------------------------------------------------+----------------------------------------------------------------+
   | .. figure:: _static/figPlotOptions1.png        | .. figure:: _static/figPlotOptions2.png                        |
   |    :align: center                              |    :align: center                                              |
   |    :height: 6cm                                |    :height: 6cm                                                |
   |                                                |                                                                | 
   |    Plot -         Disabling the label display. |    Plot - Disabling the axis and color bar display.            |
   +------------------------------------------------+----------------------------------------------------------------+

**Adding points and rectangles**

Overlay some specific points or add rectangles:

   >>> import geophpy.plotting.plot as gplt
   
   >>> # Custom rectangles display
   >>> xmin, xmax, ymin, ymax = dataset.get_gridextent()
   >>> area_extents = [[15, 25, 15, 20], [35, xmax-0.5, 5, ymax-0.5]]
   >>> rectangles = gplt.extents2rectangles(area_extents)
   >>> dataset.plot(plottype='2D-SURFACE', rects=rectangles)

   >>> # Custom points display
   >>> points = [[22, 17], [45, 25]]
   >>> dataset.plot(plottype='2D-SURFACE', rects=rectangles, points=points)

   +-------------------------------------------------+----------------------------------------------------------------+
   | .. figure:: _static/figPlotOverlays1.png        | .. figure:: _static/figPlotOverlays2.png                       |
   |    :align: center                               |    :align: center                                              |
   |    :height: 6cm                                 |    :height: 6cm                                                |
   |                                                 |                                                                | 
   |    Plot - Overlaying custom rectangles.         |    Plot - Overlaying custom points.                            |
   +-------------------------------------------------+----------------------------------------------------------------+

.. note:: 

   You can used the :meth:`~geophpy.plotting.plot.extents2rectangles` method of the :mod:`~geophpy.plotting.plot` module to convert area extent (xmin, xmax, ymin, ymax) to rectangle (x, y, width, height) to be displayed:

   >>> import geophpy.plotting.plot as gplt
   >>> area_extents = [ [xmin, xmax ymin, ymax], ...]
   >>> rectangles = gplt.extents2rectangles(area_extents)
   >>> dataset.plot(plottype='2D-SURFACE', rects=rectanges)

Histogram
=========

To adjust the limits of color map you must view the limits of the data set:

    >>> zmin, zmax = dataset.histo_getlimits()

You can plot the histogram curve in black, or respecting the dataset color map and with or without a color bar:

    >>> # Black histogram
    >>> valmin= -20, valmax = 20
    >>> dataset.histo_plot(zmin=valmin, zmax=valmax, cmapdisplay=False, coloredhisto=False)

    >>> # Colored histogram
    >>> dataset.histo_plot(zmin=valmin, zmax=valmax, cmapdisplay=True, coloredhisto=True)

+-------------------------------------------------+----------------------------------------------------------------+
| .. figure:: _static/figPlotHisto1.png           | .. figure:: _static/figPlotHisto2.png                          |
|    :align: center                               |    :align: center                                              |
|    :height: 6cm                                 |    :height: 6cm                                                |
|                                                 |                                                                | 
|    Plot - Dataset Histogram.                    |    Plot - Dataset colored histogram.                           |
+-------------------------------------------------+----------------------------------------------------------------+

    >>> # Saving histogram to a file
    >>> dataset.histo_plot(filename='histogram.png')

Correlation
===========

You can plot the correlation map between a profile and mean of its surrounding profiles  (see :ref:`chap-gen-proc-festoon-geophpy`):

    >>> dataset.correlation_plotmap(method="Crosscorr")

or the mean correlation profile is used as correlation map (see see :ref:`chap-gen-proc-festoon-geophpy`):

    >>> dataset.correlation_plotsum(method="Crosscorr")

+-------------------------------------------------+----------------------------------------------------------------+
| .. figure:: _static/figCorrelationMap.png       | .. figure:: _static/figCorrelationSum.png                      |
|    :align: center                               |    :align: center                                              |
|    :height: 6cm                                 |    :height: 6cm                                                |
|                                                 |                                                                | 
|    Plot - Dataset correlation map.              |    Plot - Dataset mean correlation profile.                    |
+-------------------------------------------------+----------------------------------------------------------------+

Mean cross-track profile
========================

Before and after destriping mean cross-track profiles can be displayed with the following commands:

    >>> dataset.meantrack_plot(Nprof=4, method='additive', Ndeg=None, plotflag='raw')

    >>> dataset.meantrack_plot(Nprof=4, method='additive', Ndeg=None, plotflag='both')

+-------------------------------------------------+----------------------------------------------------------------+
| .. figure:: _static/figDestriping3.png          | .. figure:: _static/figDestriping4.png                         |
|    :align: center                               |    :align: center                                              |
|    :height: 6cm                                 |    :height: 6cm                                                |
|                                                 |                                                                | 
|    Plot - Dataset raw mean cross-track profile. |    Plot - Dataset destripped mean cross-track profile..        |
+-------------------------------------------------+----------------------------------------------------------------+
