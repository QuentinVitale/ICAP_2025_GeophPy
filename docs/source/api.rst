.. _chap-api-geophpy:

API
***

.. _chap-hlvl-proc-fun-geophpy:

High level API
==============

.. automodule:: geophpy.dataset
    :members: getlinesfrom_file, fileformat_getlist, plottype_getlist, interpolation_getlist, colormap_getlist, colormap_plot, pictureformat_getlist, rasterformat_getlist, correlmap, griddinginterpolation_getlist, festooncorrelation_getlist, sensorconfig_getlist

.. autoclass:: geophpy.dataset.Info
    :members: 

.. autoclass:: geophpy.dataset.Data
    :members:

.. autoclass:: geophpy.dataset.GeoRefSystem
    :members:

.. autoclass:: geophpy.dataset.DataSet
    :members: from_file, to_file, plot, histo_plot, meantrack_plot, histo_getlimits, copy, interpolate, sample, regrid, threshold, peakfilt, medianfilt, festoonfilt, correlmap, regtrend, wallisfilt, ploughfilt, zeromeanprofile, destripecon, destripecub, polereduction, logtransform, continuation, analyticsignal, magconfigconversion, get_xgrid, get_ygrid, get_xygrid, get_grid_values, get_gridcorners, get_gridextent, get_xvect, get_yvect, get_xyvect, get_xvalues, get_yvalues, get_xyvalues, get_values, get_xyzvalues, get_median_xstep, get_median_ystep, get_median_xystep

.. automodule:: geophpy.geoposset
   :members: refsys_getlist, filetype_getlist, utm_to_wgs84, wgs84_to_utm, utm_getzonelimits, isvalid_utm_letter, isvalid_utm_number

.. autoclass:: geophpy.geoposset.GeoPosSet
    :members: from_ascii_file, from_file, to_ascii, plot, to_kml

High level processing functions
===============================

The calling protocol of these functions is described in the end of this document (see :ref:`chap-hlvl-api-geophpy`) but about the detailed source code of is available in this section.

.. automodule:: geophpy.processing.general
    :members: threshold, peakfilt, medianfilt, festoonfilt, detrend, regtrend, wallisfilt, ploughfilt, zeromeanprofile, destripecon, destripecub

.. automodule:: geophpy.processing.magnetism
    :members: logtransform, polereduction, continuation, eulerdeconvolution, analyticsignal, magconfigconversion, susceptibility

.. automodule:: geophpy.operation.general
    :members: apodisation2d
 
.. _chap-hlvl-plot-fun-geophpy:

High level plotting functions
=============================

The calling protocol of these functions is described in the end of this document (see :ref:`chap-hlvl-api-geophpy`) but about the detailed source code of is available in this section.

.. automodule:: geophpy.plotting.histo
    :members:

.. automodule:: geophpy.plotting.correlation
    :members:

.. automodule:: geophpy.plotting.destriping
    :members:

.. automodule:: geophpy.plotting.spectral
    :members:

.. automodule:: geophpy.plotting.plot
    :members: plot, extents2rectangles

.. _chap-hlvl-api-geophpy:
