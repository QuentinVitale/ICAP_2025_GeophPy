.. _chap-quick-start-geophpy:

Getting started
***************

Quick start
===========

First, import the :class:`~geophpy.dataset.DataSet` class from the :class:`~geophpy.dataset` module using the ``import`` command at the top of your script: 

    >>> from geophpy.dataset import DataSet

You can a so import the complete dataset module content using (not recommended for script readability):

    >>> from geophpy.dataset import *


**Reading data from files**

    Read your data from an ASCII delimiter-separated values files using the :meth:`~geophpy.dataset.DataSet.from_file` method from the :class:`~geophpy.dataset.DataSet` class:
   
    >>> # Opening data from an ascii file
    >>> dataset = DataSet.from_file(["test.dat"], fileformat='ascii', 
    x_colnum=1, y_colnum=2, z_colnum=5)

    If not provided, the file format will be estimated from the file extension and the first three columns will be considered as the data (X, Y and Z respectively).

    You can optionally give the file delimiter, the number of header lines to skip and the header lines that contains the different field names.

**Displaying the dataset**

    Display your raw data as a scatter plot or interpolate it to display 2-D surface plot:

    >>> # Raw data scatter plot
    >>> fig, cmap = dataset.plot(plottype='2D-SCATTER', cmmin=-20, cmmax=20)
    >>> fig.show()

    >>> # Raw data surface plot
    >>> dataset.interpolate(interpolation='linear')  # or 'none'
    >>> fig, cmap = dataset.plot(plottype='2D-SURFACE', cmmin=-20, cmmax=20)
    >>> fig.show()

    +-----------------------------------------------------+-----------------------------------------------------+
    | .. figure:: _static/figQuickStartScatterPlot.png    | .. figure:: _static/figQuickStartSurfacePlot.png    |
    |    :height: 6cm                                     |    :height: 6cm                                     |
    |    :align: center                                   |    :align: center                                   |
    |                                                     |                                                     |
    |    Quick Start - Raw dataset scatter plot.          |    Quick Start - Raw dataset surface plot.          |
    +-----------------------------------------------------+-----------------------------------------------------+

    You can also display the position of the measurement points in a plot or directly onto the 2-D surface plot:

    >>> # Data point position (postmap plot)
    >>> fig, cmap = dataset.plot(plottype='2D-POSTMAP')
    >>> fig.show()

    >>> # Data point position onto the 2-D surface
    >>> fig, cmap = dataset.plot(plottype='2D-SURFACE', pointsdisplay=True)
    >>> fig.show()

    +-----------------------------------------------------+-------------------------------------------------------------+
    | .. figure:: _static/figQuickStartPostmap.png        | .. figure:: _static/figQuickStartSurfacePlotPoints.png      |
    |    :height: 6cm                                     |    :height: 6cm                                             |
    |    :align: center                                   |    :align: center                                           |
    |                                                     |                                                             |
    |    Quick Start - Raw dataset postmap plot.          |    Quick Start - Raw dataset surface plot with data points. |
    +-----------------------------------------------------+-------------------------------------------------------------+

    See :ref:`chap-plot-fun-geophpy` for the complete list of available plot possibilities.

**General processing**

    Use the available processing (despike, destaggering, destriping etc.) to filter your :class:`~geophpy.dataset.DataSet`:

    >>> # Dataset destaggering
    >>> dataset.festoonfilt(corrmin=0.5, setmin=-20, setmax=20)

    >>> # Dataset destriping
    >>> dataset.destripecon(Nprof='all', setmin=-20, setmax=20)

    +-----------------------------------------------------+-----------------------------------------------------+
    | .. figure:: _static/figQuickStartFestoonfilter.png  | .. figure:: _static/figQuickStartDestrip.png        |
    |    :height: 6cm                                     |    :height: 6cm                                     |
    |    :align: center                                   |    :align: center                                   |
    |                                                     |                                                     |
    |    Quick Start - Dataset destaggering.              |    Quick Start - Dataset destriping.                |
    +-----------------------------------------------------+-----------------------------------------------------+

    See :ref:`chap-gen-proc-geophpy` for the complete list of available processing.

**Method-specific processing**

    Use the method-specific available processing (for magnetic survey for instance):

    >>> # Dataset reduction to the pole
    >>> dataset.polereduction()

    See :ref:`chap-mag-proc-geophpy` for the complete list of available magnetic-survey-specific processing.

**Saving data in a file**

    Save your data to a file using the :meth:`~geophpy.dataset.DataSet.to_file` method from the :class:`~geophpy.dataset.DataSet` class:

    >>> # Saving to a NetCDF file
    >>> dataset.to_file("save.nc", description='My Processed Data')

    >>> # Saving to a Surfer Grid file
    >>> dataset.to_file("save.grd")

    >>> # Saving to an ascii file
    >>> dataset.to_file("save.dat", fileformat='ascii', delimiter=',')

    .. warning::

       If you only processed the gridded :class:`~geophpy.dataset.DataSet` (:attr:`~valfilt=False`), 
       use the :meth:`~geophpy.dataset.DataSet.sample` method to re-sample the gridded dataset at the original dataset value positions. 
       Otherwise you will be saving the imported raw data value.

**Georeferencing grid nodes**

    Georeference your data using Ground Control Points (survey nodes known in both local and geographic coordinate systems). 
    Use the :meth:`~geophpy.dataset.DataSet.setgeoref` from the :class:`~geophpy.dataset.DataSet` class

    >>> # Reading GCPs file
    >>> import geophpy.geoposset as pset
    >>> gpos = pset.GeoPosSet.from_file("GCPs.csv")
    >>> gpos.plot(long_label=True)
    
    >>> # Georeferencing ungridded values
    >>> dataset.setgeoref('UTM', gpos.points_list, 'T', 32)

    >>> # Georeferencing gridded values
    >>> dataset.interpolate(x_step=0.5, y_step=0.25)
    >>> dataset.setgeoref('UTM', gpos.points_list, 'T', 32)

    +-----------------------------------------------+----------------------------------------+---------------------------------------------+
    | .. figure:: _static/figGeorefLocalSystem.png  | .. figure:: _static/figGeorefGCPs.png  | .. figure:: _static/figGeorefRefSystem.png  |
    |    :height: 4cm                               |    :height: 4cm                        |    :height: 4cm                             |
    |    :align: center                             |    :align: center                      |    :align: center                           |
    |                                               |                                        |                                             |
    +-----------------------------------------------+----------------------------------------+---------------------------------------------+

DataSet overview
================

All imported data are stored into a :class:`~geophpy.dataset.DataSet` class that contains the different plotting and processing methods.

The :class:`~geophpy.dataset.DataSet` class is composed of 3 objects:

**DataSet.info**

    The :class:`~geophpy.dataset.Info` class that contains the informations about the dataset:

    * :attr:`~geophpy.dataset.Info.x_min` = minimal x coordinate of the data set.
    * :attr:`~geophpy.dataset.Info.x_max` = maximal x coordinate of the data set.
    * :attr:`~geophpy.dataset.Info.y_min` = minimal y coordinate of the data set.
    * :attr:`~geophpy.dataset.Info.y_max` = maximal y coordinate of the data set.
    * :attr:`~geophpy.dataset.Info.z_min` = minimal z value of the data set.
    * :attr:`~geophpy.dataset.Info.z_max` = maximal z value of the data set.
    * :attr:`~geophpy.dataset.Info.x_gridding_delta` = delta between 2 x values in the interpolated image grid.
    * :attr:`~geophpy.dataset.Info.y_gridding_delta` = delta between 2 y values in the interpolated image grid.
    * :attr:`~geophpy.dataset.Info.gridding_interpolation` = interpolation name used for the building of the image grid.

**DataSet.data**

    The :class:`~geophpy.dataset.Data` class that contains:

    * :attr:`~geophpy.dataset.Data.fields` = the names of the fields (columns): ['X', 'Y', 'Z'].
    * :attr:`~geophpy.dataset.Data.values` = 2D array of raw values before interpolation (array with (x, y, z) values).
    * :attr:`~geophpy.dataset.Data.z_image` = 2D array of the current gridded data values.
    * :attr:`~geophpy.dataset.Data.easting_image` = None    # easting grid (to use with z_image)
    * :attr:`~geophpy.dataset.Data.northing_image` = None   # northing grid(to use with z_image)
    * :attr:`~geophpy.dataset.Data.easting` = None    # easting array (to use with values)
    * :attr:`~geophpy.dataset.Data.northing` = None   # northing array (to use with values)

.. warning::
 
   The :attr:`~geophpy.dataset.DataSet.data.z_image` object is NOT AUTOMATICALLY BUILT after opening a file but by explicitly using the gridding interpolation method :meth:`~geophpy.dataset.DataSet.interpolate`. 
   See :ref:`DataSetOperation` for details.

**DataSet.georef**

    The :class:`~geophpy.dataset.GeoRefSystem` class object contains:

    * :attr:`~geophpy.dataset.GeoRefSystem.active` = A flag for the georeferencing status.
    * :attr:`~geophpy.dataset.GeoRefSystem.refsystem` = The georeferencing system ('UTM', 'WGS84', ...).
    * :attr:`~geophpy.dataset.GeoRefSystem.utm_zoneletter` = The optional UTM letter.
    * :attr:`~geophpy.dataset.GeoRefSystem.utm_zonenumber` = The optional UTM zone number.
    * :attr:`~geophpy.dataset.GeoRefSystem.points_list` = the list of the Ground Control Points coordinates in both the local and georeferenced system.

Opening files
=============

All the reading possibilities are available through the :meth:`~geophpy.dataset.DataSet.from_file` method of the :class:`~geophpy.dataset.DataSet` class.

:program:`GeophPy` manages different types of files. You can obtain the list of accepted file formats with the command:

    >>> from geophpy.dataset import fileformat_getlist
    >>> fileformat_getlist()
    ['ascii', 'netcdf', 'surfer']

The file format is automatically recognized from the file extension using an internal dictionary:

    >>> from geophpy.dataset import format_chooser
    >>> format_chooser
    {'.cdf': 'netcdf', '.nc': 'netcdf', 
    '.grd': 'surfer',
    '.xyz': 'ascii', '.csv': 'ascii', '.txt': 'ascii', '.dat': 'ascii'}

If the file format is not in the dictionary or is not properly recognized from the extension, it can be forced to a specific format using the ``fileformat`` keyword of the :meth:`~geophpy.dataset.DataSet.from_file` method.

**ASCII files**

    You can open comma-separated values (CSV) files, or any other delimiter-separated values files, 
    by indicating the number and type of the columns of interest for the dataset to be processed:

    >>> # Opening a ".dat" file from Geometrics Magnetometer G-858
    >>> ## (format 'ascii' with delimiter ' ')
    >>> dataset = DataSet.from_file(["test.dat"], fileformat='ascii',
    delimiter=' ', z_colnum=5)

    Geometrics Magnetometer G-858 *.dat* file example:

    >>>           X           Y     TOP_RDG  BOTTOM_RDG    VRT_GRAD
    >>>      50.000       0.059   46406.028   46390.698     -23.585
    >>>      50.000       0.178   46407.275   46394.028     -20.380
    >>>      50.000       0.296   46409.165   46397.987     -17.197
    >>>      ...          ...     ...         ...           ...

    >>> # Opening an *.xyz* file 
    >>> ## (column titles on the first line,
    >>> ## X, Y and data values on the others lines,
    >>> ## separated by a delimiter)
    >>> dataset = DataSet.from_file(["test.xyz"], fileformat='ascii',
    delimiter='\t', fields_row=1)


    *.xyz* file examples:

    >>> X      Y      Z
    >>> 0      0      0.34
    >>> 0      1      -0.21
    >>> 0      2      2.45
    >>> ...    ...    ...

    >>> X,Y,Z
    >>> 0,0,0.34
    >>> 0,1,-0.21
    >>> 0,2,2.45
    >>> ...,...,...

    You can also specify the number of header lines to skip or the specific columns for x, y and values:

    >>> # Number of header lines to skip
    >>> dataset = DataSet.from_file(["test.txt"], fileformat='ascii',
    delimiter=',', skip_rows=4)

    >>> # Specific x, y and value column numbers
    >>> dataset = DataSet.from_file(["test.txt"], fileformat='ascii',
    delimiter=',', x_colnum=xcol, y_colnum=ycol, z_colnum=zcol)

    .. note::

       If unspecified, the ``delimiter`` is estimated directly from the file content and the ``fileformat`` is determined from the file extension. 

**Surfer Grid files**

    :program:`GeophPy` manages Golden Software :program:`Surfer` Grid files (Surfer 7 binary grids, Surfer 6 binary grids and Surfer 6 ASCII grids). 
    The grid type is automatically determined from the `.grd` file. 
    To open a Surfer Grid simply use:

    >>> # Opening a Surfer Grid file
    >>> dataset = DataSet.from_file(["test.grd"])

**NetCDF files**

    Previously processed dataset are by default save in NetCDF format (*.nc*). To open previously processed datasets, simply use:

    >>> # Opening previously processed dataset (.nc)
    >>> dataset = DataSet.from_file(["dataset1.nc"])

**Concatenating Multiple files**

    It is possible to build a dataset from a concatenation of severals ASCII files of the same format:
    
    >>> # Opening several selected files
    >>> dataset = DataSet.from_file(["file1.dat","file2.dat"],
                  format='ascii', delimiter=' ', z_colnum = 5)

    >>> # Opening all files beginning by "file"
    >>> dataset = DataSet.from_file(["file*.dat"], format='ascii',
		  delimiter=' ', z_colnum = 5)

    .. note::

       When reading multiple files directly using the :meth:`~geophpy.dataset.DataSet.from_file` method, no edge-matching method are used so the original limits of the datasets in the mosaic maybe highly visible. 


**Checking files compatibility**

    Opening several files to build a data set needs to make sure that all files selected are in the same format.

    It's possible to check it by reading the headers of each files:

    >>> compatibility = True
    >>> columns_nb = None
    >>> for file in fileslist :
    >>>    col_nb, rows = getlinesfrom_file(file)
    >>>    if ((columns_nb != None) and (col_nb != columns_nb)) :
    >>>        compatibility = False
    >>>        break
    >>>    else :
    >>>        columns_nb = col_nb


.. _DataSetOperation:

.. _chap-dataset-op-geophpy:

Dataset operation
=================

Besides actual :class:`~geophpy.dataset.DataSet` processing, basic :class:`~geophpy.dataset.DataSet` operations (geometrical transformation, math operations, interpolation etc.) are available through simple commands.

**Duplicating dataset**

    Duplicate a :class:`~geophpy.dataset.DataSet` before processing it to save the raw data:

    >>> rawdataset = dataset.copy()

**Dataset interpolation**

    Interpolate the data value with severals gridding interpolation methods ('none', 'nearest', 'linear', 'cubic') to build the dataset :obj:`z_image` object:

    >>> dataset.interpolate(interpolation="none")

    +-----------------------------------------------------+
    | .. figure:: _static/figCarto2.png                   |
    |    :height: 6cm                                     |
    |    :align: center                                   |
    |                                                     |
    |    Quick Start - Dataset interpolation ('none')     |
    +-----------------------------------------------------+

    See :ref:`chap-hlvl-api-geophpy` for calling details.

**Retrieving grid coordinates**

    The :class:`~geophpy.dataset.DataSet` grid (:obj:`z_image`) coordinate vectors and matrices can be retrieved with the following commands:
    
    >>> # Grid coordinate matrices 
    >>> dataset.get_xgrid()  # x-coordinate matrix
    >>> dataset.get_ygrid()  # x-coordinate matrix
    >>> dataset.get_xygrid()  # both x and y-coordinate matrices

    >>> # Grid coordinate vectors
    >>> dataset.get_xvect()  # x-coordinate vector
    >>> dataset.get_yvect()  # y-coordinate vector
    >>> dataset.get_xyvect()  # both x and y-coordinate vectors
    >>> dataset.get_gridextent()  # xmin, xmax, ymin, ymax
    >>> dataset.get_gridcorners() # corners x and y-coordinates

    The :class:`~geophpy.dataset.DataSet` ungridded values (:obj:`values`) can be retrieved with the following commands:
    
    >>> # Data sample coordinates 
    >>> dataset.get_xvalues()  # x-coordinates
    >>> dataset.get_yvalues()  # y-coordinates
    >>> dataset.get_yvalues()  # x, y-coordinates
    >>> dataset.get_values()  # data values
    >>> dataset.get_xyzvalues()  # both x, y-coordinates and data values

    >>> # Bounding box
    >>> dataset.get_boundingbox() # corners coordinates (equivalent get_gridcorners() for a gridded dataset)

    See :ref:`chap-hlvl-api-geophpy` for calling details.

**Basic math operations**

    You can add or multiply the :class:`~geophpy.dataset.DataSet` values by a constant with the following commands:
    
    >>> # Dataset addition/subtraction
    >>> dataset.add(val=14, valfilt=True, zimfilt=True)
    >>> dataset.add(val=-14, valfilt=True, zimfilt=True)

    >>> # Dataset multiplication/division
    >>> dataset.times(val=30, valfilt=True, zimfilt=True)
    >>> dataset.add(val=1/30, valfilt=True, zimfilt=True)

    See :ref:`chap-hlvl-api-geophpy` for calling details.

**Basic geometrical operations**

    You can translate or rotate the :class:`~geophpy.dataset.DataSet` with the following commands:

    >>> # Dataset translation
    >>> dataset.translate(shiftx=20, shifty=-19)

    >>> # Dataset rotation
    >>> dataset.rotate(angle=90, center='BL')

    See :ref:`chap-hlvl-api-geophpy` for calling details.


Saving DataSet 
==============

You can save the :class:`~geophpy.dataset.DataSet` in different a file formats using the :meth:`~geophpy.dataset.DataSet.to_file` method of the :class:`~geophpy.dataset.DataSet` class.

For the time being, only three formats are available. The list of the available file formats can be obtained with the command:

    >>> from geophpy.dataset import fileformat_getlist
    >>> fileformat_getlist()
    ['ascii', 'netcdf', 'surfer']

When saving a file, the file format is automatically recognized from the file extension using an internal dictionary:

    >>> from geophpy.dataset import format_chooser
    >>> format_chooser
    {'.cdf': 'netcdf', '.nc': 'netcdf', 
    '.grd': 'surfer',
    '.xyz': 'ascii', '.csv': 'ascii', '.txt': 'ascii', '.dat': 'ascii'}

If the file format is not in the dictionary or is not properly recognized from the extension, it can be forced to a specific format using the ``fileformat`` keyword of the :meth:`~geophpy.dataset.DataSet.from_file()` method.

**NetCDF files**

    Saving your data in NetCDF file format allow the conservation of the gridded dataset, the dataset values and the georeferencing system.
    An optional description can be added using the ``description`` keyword.

    >>> dataset.to_file('save.nc')
    >>> dataset.to_file('save.nc', description='My dataset example')
    >>> dataset.to_file('save.extension', fileformat='netcdf')

**Surfer Grid files**

    Saving your data in Surfer Grid format only conserves the gridded dataset.  
    The different available grid types can be obtained using the command:

    >>> gridtype_getlist()
    ['surfer7bin', 'surfer6bin', 'surfer6ascii']

    By default, the Surfer 7 binary grid type is used but you can use the ``gridtype`` keyword of the :meth:`~geophpy.dataset.DataSet.from_file` method to choose another grid type:

    >>> dataset.to_file('save.grd')
    >>> dataset.to_file('save.extension', fileformat='surfer')
    >>> dataset.to_file('save.grd', gridtype='surfer6ascii')

**Ascii files**

    Saving your data in Surfer Grid format only conserves the dataset (ungridded) values. 

    >>> dataset.to_file('save.csv')
    >>> dataset.to_file('save.csv', delimiter='\t')
    >>> dataset.to_file('save.extension', delimiter='\t', fileformat='ascii')

    .. warning::

       If you only processed the gridded dataset (``valfilt=False``), 
       use the :meth:`~geophpy.dataset.DataSet.sample` method to re-sample the gridded :class:`~geophpy.dataset.DataSet` at the original value positions. 
       Otherwise you will be saving the imported raw data value.
