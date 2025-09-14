# -*- coding: utf-8 -*-
'''
   geophpy.core.operation
   -------------------------

   DataSet Object general operations routines.

   :copyright: Copyright 2014-2025 Lionel Darras, Philippe Marty, Quentin Vitale and contributors, see AUTHORS.
   :license: GNU GPL v3.
'''

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import griddata, RectBivariateSpline, interp2d, RBFInterpolator
from scipy.stats import linregress
from sklearn.linear_model import (HuberRegressor,TheilSenRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import gstlearn as gl
import gstlearn.plot as gp
from copy import deepcopy
import re

import geophpy.core.utils as genut
from geophpy.visualization.heatmap import heatmap_plot
import geophpy.__config__ as CONFIG




class PointOperationsMixin:
    """
    Mixin for simple operations that work on ungridded (point) data.
    """
    pass


class GridOperationsMixin:
    """
    Mixin for simple operations that work on gridded (image-like) data.
    """
    pass

#---------------------------------------------------------------------------#
# User defined parameters                                                   #
#---------------------------------------------------------------------------#

# list of "griddata" interpolation methods available for wumappy interface
gridding_interpolation_list = ['none', 'nearest', 'linear', 'cubic']

# List of allowed rotation angle (for z_image) for wumappy interface
rotation_angle_list = [0, 90, 180, 270]


#---------------------------------------------------------------------------#
# DataSet Basic Interpolations                                              #
#---------------------------------------------------------------------------#
def getgriddinginterpolationlist():
    '''
    cf. dataset.py
    '''
    return gridding_interpolation_list


### Interpolated moved to spatial
# # def interpolate(dataset, interpolation="none", x_step=None, y_step=None, x_prec=2, \
#                 y_prec=2, x_frame_factor=0., y_frame_factor=0.):
#    ''' Dataset gridding.

#    cf. :meth:`~geophpy.dataset.DataSet.interpolate`

#    '''

#    x, y, z = dataset.data.values.T[:3]

#    # Creating a regular grid ###################################################
#    # Distinct x, y values
#    x_list = np.unique(x)
#    y_list = np.unique(y)

#    # Median step between two distinct x values
#    if x_step is None:
#        x_step = get_median_xstep(dataset, prec=x_prec) # add prec is None is this function
#        # ...TBD... why not take the min diff value instead of the median ?
#        ## Because on all parallel profiles, min can be smaller than actuall step size

#    else:
#        x_prec = genut.getdecimalsnb(x_step)

#    # Min and max x coordinates and number of x pixels
#    xmin = x.min()
#    xmax = x.max()

#    xmin = (1.+x_frame_factor)*xmin - x_frame_factor*xmax
#    xmax = (1.+x_frame_factor)*xmax - x_frame_factor*xmin

#    xmin = round(xmin, x_prec)
#    xmax = round(xmax, x_prec)

#    nx = int(np.around((xmax-xmin)/x_step) + 1)

#    # Median step between two distinct y values
#    if y_step is None:
#        y_step = get_median_ystep(dataset, prec=y_prec)
#        # ...TBD... why not take the min diff value instead of the median ?

#    else:
#        y_prec = genut.getdecimalsnb(y_step)

#    # Determinate min and max y coordinates and number of y pixels
#    ymin = y.min()
#    ymax = y.max()

#    ymin = (1.+y_frame_factor)*ymin - y_frame_factor*ymax
#    ymax = (1.+y_frame_factor)*ymax - y_frame_factor*ymin

#    ymin = round(ymin, y_prec)
#    ymax = round(ymax, y_prec)

#    ny = int(np.around((ymax - ymin)/y_step) + 1)

#    # Regular grid
#    xi = np.linspace(xmin, xmax, nx, endpoint=True)
#    yi = np.linspace(ymin, ymax, ny, endpoint=True)
#    X, Y = np.meshgrid(xi, yi)

#    # Gridding data #############################################################
#    # No interpolation
#    if interpolation.lower() == "none":
#       ## just project data into the grid
#       ## if several data points fall into the same pixel, they are averaged
#       ## don't forget to "peakfilt" the raw values beforehand to avoid averaging bad data points

#       ### attempt using scipy.stats.binned_statistic_2d
#       x, y, val = dataset.get_xyzvalues()
#       statistic, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(
#           x, y, values=val, statistic='mean',bins=[xi, yi])
#       Z = statistic.T
#       #Z = np.flipud(Z)
      
#       #print(type(Z))
#       #Z = dataset.data.z_image
#       #Z[np.isnan(Z)] = 0  # replacing nan with zero, waiting for better
#       #print('*** interpolate - sat', Z)
#       ###

# ##      Z = np.zeros(X.shape)
# ##      Count = np.zeros(X.shape)  # nb of data points in the pixel initialization
# ##      #print('*** interpolate - count ', Count)
# ##
# ##      for x, y, z in dataset.data.values:
# ##         indx = np.where(xi+x_step/2. > x)
# ##         indy = np.where(yi+y_step/2. > y)
# ##         Z[indy[0][0], indx[0][0]] += z
# ##         Count[indy[0][0], indx[0][0]] += 1
# ##
# ##      idx0 = Count == 0  # index of pixel with no data
# ##      #print('*** interpolate - idx0', idxo)
# ##      Z[~idx0] = Z[~idx0]/Count[~idx0]

#    # SciPy iterpolation
#    elif interpolation in getgriddinginterpolationlist():
#       ## perform data interpolation onto the grid
#       ## the interpolation algorithm will deal with overlapping data points
#       ## nevertheless don't forget to "peakfilt" the rawvalues beforehand to avoid
#       ## interpolation being too much influenced by bad data points
#       '''
#       # Fill holes in each profiles with "nan" #######################
#       ## this is to avoid filling holes with interpolated values
#       nan_array = []
#       for x in x_list:
#          profile = np.unique(y_array[np.where(x_array == x)])
#          nan_array = profile_completewithnan(x, profile, nan_array, y_step, factor=2, ymin=ymin, ymax=ymax)
#       if (len(nan_array) != 0):
#          completed_array = np.append(dataset.data.values, np.array(nan_array), axis=0)
#          T = completed_array.T
#          x_array = T[0]
#          y_array = T[1]
#          z_array = T[2]
#       '''

#       Z = griddata((x, y), z, (X, Y), method=interpolation)

#       if np.all(np.isnan(Z.flatten())):  # interpolation failled
#           print('griddata', Z, len(Z))
#           print('griddataAllNan', np.all(np.isnan(Z)))
#           return dataset

#    # Other iterpolation
#    else:
#       # Undefined interpolation method ###############################
#       # ...TBD... raise an error here !
#       pass

#    # Fill the DataSet Object ###################################################
#    dataset.data.z_image = Z
#    dataset.info.x_min = xmin
#    dataset.info.x_max = xmax
#    dataset.info.y_min = ymin
#    dataset.info.y_max = ymax
#    dataset.info.z_min = np.nanmin(Z)
#    dataset.info.z_max = np.nanmax(Z)
#    dataset.info.x_gridding_delta = x_step
#    dataset.info.y_gridding_delta = y_step
#    dataset.info.gridding_interpolation = interpolation

#    return dataset


def sample(dataset):
    ''' Re-sample data at ungridded sample position from gridded Z_image.

    cf. :meth:`~geophpy.dataset.DataSet.sample`

    '''

    # Current gridded values
    X, Y = get_xygrid(dataset)
    Z = dataset.data.z_image

    idx_colan = np.isnan(Z)
    xy = np.stack((X[~idx_colan].flatten(),Y[~idx_colan].flatten())).T
    z = Z[~idx_colan].flatten()

    # ungridded values coordinates at which resample
    xiyi = dataset.data.values.T[:2].T  # [[x0, y0], [x1, y1], ...]
    zi = dataset.data.values.T[2]

    # Re-sampling
    z_interp = griddata(xy, z, xiyi, method='cubic')
    zi *= 0.
    zi += z_interp

    return dataset


def regrid(dataset, x_step=None, y_step=None, method='cubic'):
    ''' Re-grid dataset grid.

    cf. :meth:`~geophpy.dataset.DataSet.regrid`

    '''

    datasetOld = dataset.copy()
    dataset.sample()

##    # New grid step, resample the old grid by a factor 2.
##    if x_step is None:
##        x_step_old = dataset.info.x_gridding_delta
##        prec = getdecimalsnb(x_step_old)
##        x_step = np.around(x_step_old, prec)
##
##    if y_step is None:
##        y_step_old = dataset.info.y_gridding_delta
##        prec = getdecimalsnb(y_step_old)
##        y_step = np.around(y_step_old, prec)

    # Re-gridding dataset
    dataset.interpolate(x_step=x_step, y_step=y_step, interpolation=method)

    # Filling DataSet Object
    dataset.data.values = datasetOld.data.values

    return dataset


def histo_fit(dataset, valfilt=False):
    ''' Fit dataset histogram distribution. '''

    # Fit on ungridded dataset values
    if valfilt or dataset.data.z_image is None:
        data = dataset.get_values()

    # Fit on gridded dataset values
    else:
        data = dataset.get_grid_values()

    # Normal (gaussian) fit
    #    data = scipy.stats.norm.fit(ser)
    m, s = scipy.stats.norm.fit(ser) # get mean and standard deviation
    #pdf_g = scipy.stats.norm.pdf(lnspc, m, s)
    return m, s


# #---------------------------------------------------------------------------#
# # DataSet Grid manipulation                                                 #
# #---------------------------------------------------------------------------#
# def get_xvect(dataset):
#     ''' Return dataset x-coordinate grid vector. '''

#     is_grid = (dataset.info is not None
#                and dataset.data.z_image is not None)

#     if not is_grid:
#         return None

#     xmin = dataset.info.x_min
#     xmax = dataset.info.x_max
#     nx = dataset.data.z_image.shape[1]

#     return np.array([np.linspace(xmin, xmax, nx)])


# def get_yvect(dataset):
#     ''' Return dataset y-coordinate grid vector. '''

#     is_grid = (dataset.info is not None
#                and dataset.data.z_image is not None)

#     if not is_grid:
#         return None

#     ymin = dataset.info.y_min
#     ymax = dataset.info.y_max
#     ny = dataset.data.z_image.shape[0]

#     return np.array([np.linspace(ymin, ymax, ny)])


# def get_xyvect(dataset):
#     ''' Return dataset x- and y-coordinate grid vectors. '''

#     x = get_xvect(dataset)
#     y = get_yvect(dataset)

#     return x, y

##def zimage_xcoord(dataset):
##    '''
##    Return dataset x-coordinate array of a Z_image.
##    '''
##    return np.array([np.linspace(dataset.info.x_min, dataset.info.x_max, dataset.data.z_image.shape[1])])
##
##
##def zimage_ycoord(dataset):
##    '''
##    Return dataset y-coordinate array of a Z_image.
##    '''
##    return np.array([np.linspace(dataset.info.y_min, dataset.info.y_max, dataset.data.z_image.shape[0])])


# def get_xygrid(dataset):
#     ''' Return dataset x and y-coordinate  grid. '''

#     is_grid = (dataset.info is not None
#                and dataset.data.z_image is not None)

#     if not is_grid:
#         return None, None

#     x, y = get_xyvect(dataset)
#     X, Y = np.meshgrid(x, y)

#     return X, Y


# def get_xgrid(dataset):
#     ''' Return dataset x-coordinate  grid. '''

#     return get_xygrid(dataset)[0]


# def get_ygrid(dataset):
#     ''' Return dataset y-coordinate  grid. '''

#     return get_xygrid(dataset)[1]


# def get_gridextent(dataset):
#     ''' Return dataset grid extent. '''

#     is_grid = (dataset.info is not None
#                and dataset.data.z_image is not None)

#     if not is_grid:
#         return None, None, None, None

#     xmin = dataset.info.x_min
#     xmax = dataset.info.x_max
#     ymin = dataset.info.y_min
#     ymax = dataset.info.y_max

#     return  xmin, xmax, ymin, ymax


# def get_gridcorners(dataset):
#     ''' Return dataset grid corners coordinates (BL, BR, TL, TR). '''

#     is_grid = (dataset.info is not None
#                and dataset.data.z_image is not None)

#     if not is_grid:
#         return None

#     xmin = dataset.info.x_min
#     xmax = dataset.info.x_max
#     ymin = dataset.info.y_min
#     ymax = dataset.info.y_max

#     return np.array([[xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax]])


# def get_boundingbox(dataset):
#     ''' Return the coordinates (BL, BR, TL, TR) of the box bounding the data values. '''

#     x, y = get_xyvalues(dataset)

#     xmin = x.min()
#     xmax = x.max()
#     ymin = y.min()
#     ymax = y.max()

#     return np.array([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])


# def get_median_xstep(dataset, prec=None):
#     ''' Return the median step between two distinct x values rounded to the given precision.

#     Profiless are considered parallel to the y-axis. '''

#     x = get_xvalues(dataset)

#     if prec is None:
#         prec = max(genut.getdecimalsnb(x))

#     x_list = np.unique(x)
#     x_step = np.median(np.around(np.diff(x_list), prec))

#     return x_step


# def get_median_ystep(dataset, prec=None):
#     ''' Return the median step between two distinct y values rounded to the given precision.

#     Profiles are considered parallel to the y-axis. '''

#     y = get_yvalues(dataset)

#     if prec is None:
#         prec = max(genut.getdecimalsnb(y))

#     #y_list = np.unique(y)
#     #y_step = np.median(np.around(np.diff(y_list), prec))
#     # Profiles are considered parallel to the y-axis, so y values have more variations than x values.
#     ## using np.unique() can result in an underestimation off the median y_step.
#     y_step = np.median(np.around(np.abs(np.diff(y)), prec))

#     return y_step


# def get_median_xystep(dataset, x_prec=None, y_prec=None):
#     ''' Return the median steps between two distinct x and y values rounded to the given precisions. '''

#     x_step = get_median_xstep(dataset, prec=x_prec)
#     y_step = get_median_ystep(dataset, prec=y_prec)

#     return x_step, y_step


def get_track(dataset, num, attr='values'):
    ''' Return the values corresponding to the track number.

    Parameters
    ----------
    dataset : 1-D array-like
        Dataset object or any object having a track and values attributes.

    num : int or sequence of int
        Track number.

    attr : {'values', 'x', 'y', 'east', 'north', 'long', 'lat'}
        The named attribute of object.

    Return
    ------
    profiles : 1-D array-like

    '''

    if dataset.data.track is None:
        return

    # sequence of tracks
    if hasattr(num, '__iter__') and not isinstance(num, str):
        idx = np.any(np.asarray([dataset.data.track==int(track_num) for track_num in num]), axis=0)

    # single track
    else:
        idx = dataset.data.track==int(num)

    if attr == 'index':
        return np.where(idx)[0]
        
    values = getattr(dataset.data, attr, None)

    if values is not None:
        return values[idx]

    return values

    
#def del_track(dataset, num):
#    ''' Delete the values corresponding to the track number.
#
#    Parameters
#    ----------
#    dataset : 1-D array-like
#        Dataset object or any object having a track and values attributes.
#
#    num : int
#        Track number.
#
#    Return
#    ------
#    Dataset cleared of track num related attributes
#
#    '''
#
#    if dataset.data.track is None:
#        return
#
#    idx = np.where(dataset.data.track==int(num))
#
#    if dataset.data.track is None:
#        return
#    return dataset.data.values[np.where(dataset.data.track==int(num))]



##
#####
##def get_min_xstep(dataset, prec=None):
##    ''' Return the min step between two distinct x values rounded to the given precision. '''
##
##    x = get_xvalues(dataset)
##
##    if prec is None:
##        prec = max(getdecimalsnb(x))
##
##    x_list = np.unique(x)
##    x_step = np.min(np.around(np.diff(x_list), prec))
##
##    return x_step
##
##
##def get_min_ystep(dataset, prec=None):
##    ''' Return the min step between two distinct y values rounded to the given precision. '''
##
##    y = get_yvalues(dataset)
##
##    if prec is None:
##        prec = max(getdecimalsnb(y))
##
##    y_list = np.unique(y)
##    y_step = np.min(np.around(np.diff(y_list), prec))
##
##    return y_step
##
##
##def get_min_xystep(dataset, x_prec=None, y_prec=None):
##    ''' Return the min steps between two distinct x and y values rounded to the given precisions. '''
##
##    x_step = get_min_xstep(dataset, prec=x_prec)
##    y_step = get_min_ystep(dataset, prec=y_prec)
##
##    return x_step, y_step
#####

#def apodisation2d(val, apodisation_factor):
##   '''
##   2D apodisation, to reduce side effects
##
##   Parameters :
##
##   :val: 2-Dimension array
##
##   :apodisation_factor: apodisation factor in percent (0-25)
##
##   '''
#   if (apodisation_factor > 0):
#      # apodisation in the x direction
#      for profile in val.T:
#         _apoisation1d(profile, apodisation_factor)

      # apodisation in the y direction
#      for profile in val:
#         _apodisation1d(profile, apodisation_factor)



#def _apodisation1d(array1D, apodisation_factor):
##   '''
##   1D apodisation, to reduce side effects
##
##   Parameters :
##
##   :array1D: 1-Dimension array
##
##   :apodisation_factor: apodisation factor in percent (0-25)
##
##   '''
#   na = len(array1D)                                  # n is the number of array elements
#   napod = int(np.around((na * apodisation_factor)/100))     # napod is the number of array elements to treat
#   if (napod <= 1):                                   # one element at least must be treated
#      napod = 1
#   pi2 = np.pi/2.
#   for n in range(napod):                             # for napod first data
#      array1D[n] = array1D[n]*np.cos((napod-n)*pi2/napod)

#   for n in range(na-napod, na):                      # for napod last data
#      array1D[n] = array1D[n]*np.cos((n+1-na+napod)*pi2/napod)

###
##
# In the Future, MOVE TO operation.spectral
##
###
#---------------------------------------------------------------------------#
# Fourier Transform tools                                                   #
#---------------------------------------------------------------------------#
def fillnanvalues(val, indexout=False):
    '''
    Fill 'nan' values of each profile (row) using simple linear interpolation.

    Parameters
    ----------
    val: array_like
        Array where to replace the NaNs.

    indexout: bool,
        Flag to return the index (boolean indexing) of the NaNs in the original array.

    Returns
    -------
    The completed array (and the index of the NaNs in the original array).

    '''

    val_valid = np.copy(val)

    # Index of NaNs in the array
    nan_idx = np.isnan(val)

    # Data interpolation
    ## if there are NaNs in the profile, the value at the NaNs locations
    ## will be estimated using linear interpolation.
    if nan_idx.any():
        nprof = 0
        for profile in val_valid:
            nprof += 1

            # All-nan profile
            if np.isnan(profile).all():
                nan_idx = np.isnan(profile)
                profile[nan_idx] = 1  # -999

            # Missing data in the profile
            elif np.isnan(profile).any():
                nan_idx = np.isnan(profile)
                val_idx = ~nan_idx
                valid_data = profile[val_idx]
                val_interp = np.interp(nan_idx.nonzero()[0], val_idx.nonzero()[0], valid_data)
                profile[nan_idx] = val_interp

    # Return both data and nan index
    if indexout:
        return val_valid, nan_idx

    # Return data alone
    return val_valid

##def wavenumber(nx, ny, dx, dy):
##    '''
##    Computes the grid wavenumber coordinates.
##
##    Parameters
##    ----------
##    nx, ny : int
##        Dimension of grid in x (col) and y (line) directions.
##
##    dx, dy : float
##        Sample intervals in the x and y directions.
##
##    Returns
##    -------
##    kx, ky : array_like
##        The wavenumbers coordinate in the kx and ky directions.
##
##    Examples
##    --------
##    >>> ny, nx = grid.shape
##    >>> dy, dx = 0.1, 1  # grid spatial interval in m
##    >>> fourier = np.fft.fft2(grid)
##    >>> kx, ky = wavenumber(nx, ny, dx, dy)
##    >>> kx
##    array([[ 0.  ,  0.01694915,  0.03389831,  ..., -0.05084746, -0.03389831, -0.01694915]
##    [ 0.  ,  0.01694915,  0.03389831,  ..., -0.05084746, -0.03389831, -0.01694915]
##    ...
##    [ 0.  ,  0.01694915,  0.03389831,  ..., -0.05084746, -0.03389831, -0.01694915]
##    [ 0.  ,  0.01694915,  0.03389831,  ..., -0.05084746, -0.03389831, -0.01694915]])
##    >>> ky
##    array([[   0.    ,    0.    ,    0.     ,    ... ,   0.      ,   0.    ,   0.    ]
##    [ 0.02003606,  0.02003606,  0.02003606,  ..., 0.02003606, 0.02003606, 0.02003606]
##    [ 0.04007213,  0.04007213,  0.04007213,  ..., 0.04007213, 0.04007213, 0.04007213]
##    ...
##    [ -0.04007213,  -0.04007213,  -0.04007213,  ..., -0.04007213, -0.04007213, -0.04007213]
##    [ -0.02003606,  -0.02003606,  -0.02003606,  ..., -0.02003606, -0.02003606, -0.02003606]])
##
##    '''
##
##    # x-directed wavenumber
##    kx = np.fft.fftfreq(nx, d=dx)  # x-directed wavenumber vector
##    kx.shape = [-1,nx]  # ensuring line vector
##    kx = np.matlib.repmat(kx, ny, 1)  # x-directed wavenumber matrix
##
##    # y-directed wavenumber
##    ky = np.fft.fftfreq(ny, d=dy)  # y-directed wavenumber vector
##    ky.shape = [ny,-1]  # ensuring column vector
##    ky = np.matlib.repmat(ky, 1, nx)  # y-directed wavenumber matrix
##
##    return 2*np.pi*kx, 2*np.pi*ky


def wavenumber(nx, ny, dx, dy, indx=None, indy=None):
    '''
    Computes the grid wavenumber coordinates.

    Parameters
    ----------

    nx, ny : int
        Dimension of grid in x (col) and y (line) directions.

    dx, dy : float
        Sample intervals in the x and y directions.

    indx, indy : int, optional
        Index in the kx and ky directions.
        If ix or iy are None, the whole matrix is returned.

    Returns
    -------
    kx, ky : array_like
        The wavenumbers coordinate in the kx and ky directions.

    Notes
    -----
    This function is a direct adaptation from the Subroutine B.20.
    "Subroutine to calculate the wavenumber coordinates of elements
    of grids" in (Blakely, 96)[#]_.

    References
    ----------
     .. [#] Blakely R. J. 1996.
         Potential Theory in Gravity and Magnetic Applications.
         Appendix B, p396.
         Cambridge University Press.

    '''

    # Nyquist frequencies in the kx and ky directions
    nyqx = nx/2 + 1
    nyqy = ny/2 + 1

    # Index determination
    if indx is None or indy is None:
        indx = range(nx)
        indy = range(ny)

    # Wavenumbers computation
    #kx = np.empty([len(indy), len(indx)])
    #ky = np.empty([len(indy), len(indx)])
    kx = np.zeros([len(indy), len(indx)])
    ky = np.zeros([len(indy), len(indx)])
    for ix in indx:
        for iy in indy:

            # kx direction
            if ix <= nyqx:
                kx[iy][ix] = float(ix) / ((nx-1)*dx)
            else:
                kx[iy][ix] = float(ix-nx) / ((nx-1)*dx)

            # ky direction
            if iy <= nyqy:
                ky[iy][ix] = float(iy) / ((ny-1)*dy)
            else:
                ky[iy][ix] = float(iy-ny) / ((ny-1)*dy)

    return 2*np.pi*kx, 2*np.pi*ky


def apodisation2d(val, apodisation_factor):
   '''
   2D apodisation, to reduce side effects

   Parameters :

   :val: 2-Dimension array

   :apodisation_factor: apodisation factor in percent (0-25)

   Returns :
      - apodisation pixels number in x direction
      - apodisation pixels number in y direction
      - enlarged array after apodisation
   '''

   array2DTemp = []
   array2D = []

   if apodisation_factor > 0:
      # apodisation in the x direction
      nx = len(val.T[0])                                       # n is the number of array elements
      napodx = int(np.around((nx * apodisation_factor)/100))   # napod is the number of array elements to treat
      if napodx <= 1:                                        # one element at least must be treated
         napodx = 1
      for profile in val.T:
         array2DTemp.append(_apodisation1d(profile, napodx))
      array2DTemp = (np.array(array2DTemp)).T

      # apodisation in the y direction
      ny = len(array2DTemp[0])                                 # n is the number of array elements
      napody = int(np.around((ny * apodisation_factor)/100))   # napod is the number of array elements to treat
      if napody <= 1:                                        # one element at least must be treated
         napody = 1
      for profile in array2DTemp:
         array2D.append(_apodisation1d(profile, napody))
   else:                                                       # apodisation factor = 0
      array2D = val

#   return napodx, napody, np.array(array2D)
   return np.array(array2D)



def _apodisation1d(array1D, napod):
   '''
   1D apodisation, to reduce side effects

   Parameters :

   :array1D: 1-Dimension array

   :napod: apodisation pixels number

   Returns : 1-Dimension array of len(array1D) + napod elements

   '''

   pi2 = np.pi/2.

   na = len(array1D)                                 # n is the number of array elements
   nresult = na + 2*napod
   array1Dresult = []
   for n in range(napod):
      array1Dresult.append(array1D[n]*np.cos((napod-n)*pi2/napod))
   for n in range(na):
      array1Dresult.append(array1D[n])
   for n in range(na-napod, na):                      # for napod last data
      array1Dresult.append(array1D[n]*np.cos((n+1-na+napod)*pi2/napod))

   return array1Dresult


def apodisation2Dreverse(val, valwithapod, napodx, napody):
   '''
   To do the reverse apodisation
   '''
   na = len(val)
   nb = len(val[0])
   for n in range(na):
      for m in range(nb):
         val[n][m] = valwithapod[n+napody][m+napodx]

#---------------------------------------------------------------------------#
# DataSet Basic Math Operations                                             #
#---------------------------------------------------------------------------#
def stats(dataset, valfilt=False, valmin=None, valmax=None):
    '''
    cf. dataset.py
    '''

    # Statistics on dataset values or Z_images #################################
    if valfilt:
        val = dataset.data.values[:, 2]

    else:
        val = dataset.data.z_image

    # Limiting data range ######################################################
    if valmin is None:
        valmin = np.nanmin(val)

    if valmax is None:
        valmax = np.nanmax(val)

    idx = (val >= valmin) & (val <= valmax)
    val = val[idx]

    # Dataset statistics #######################################################
    return genut.arraygetstats(val)


def multidatasetstats(datasets, valfilt=True, valmin=None, valmax=None):
    '''
    Returns basic statistics for each dataset in the Sequence of DataSet Objects
    '''

    mean, std, median, Q1, Q3, IQR = [], [], [], [], [], []

    for dataset in datasets:
        datasetstats = stats(dataset, valfilt=valfilt, valmin=valmin, valmax=valmax)
        mean.append(datasetstats[0])
        std.append(datasetstats[1])
        median.append(datasetstats[2])
        Q1.append(datasetstats[3])
        Q3.append(datasetstats[4])
        IQR.append(datasetstats[5])

    return mean, std, median, Q1, Q3, IQR


def add_constant(dataset, constant=0, valfilt=True, zimfilt=True):
    '''
    cf. dataset.py
    '''
    # Data values ##############################################################
    if valfilt:
        x, y, z = dataset.data.values.T
        z += constant

    # Z_image ##################################################################
    if zimfilt and dataset.data.z_image is not None:
        dataset.data.z_image += constant
        dataset.info.z_min += constant
        dataset.info.z_max += constant

    return dataset


def times_constant(dataset, constant=1, valfilt=True, zimfilt=True):
    '''
    cf. dataset.py
    '''

    # Data values ##############################################################
    if valfilt:
        x, y, z = dataset.data.values.T
        z *= constant

    # Z_image ##################################################################
    if zimfilt and dataset.data.z_image is not None:
        dataset.data.z_image *= constant
        dataset.info.z_min *= constant
        dataset.info.z_max *= constant

    return dataset


#---------------------------------------------------------------------------#
# DataSet Basic Manipulations                                               #
#---------------------------------------------------------------------------#
def copy(dataset):
    '''
    cf. dataset.py
    '''
    return deepcopy(dataset)


def setmedian(dataset, median=None, profilefilt=False, valfilt=False, setmethod='additive'):
    '''
    cf. dataset.py
    '''

    # No value provided for the median #########################################
    if median is None:
        return dataset

    # Set each profile's median ################################################
    if profilefilt:
        # Setting data values
        if valfilt:
            # ...TBD...
            pass

        # Setting data Z_image (if any)
        elif dataset.data.z_image is not None:
            zimage = dataset.data.z_image
            Zfilt = np.empty(zimage.shape)
            colnum = 0
            for col in zimage.T:
                Zfilt[:, colnum] = genut.arraysetmedian(col, val=median, method=setmethod)
                colnum += 1

            dataset.data.z_image = Zfilt

    # Set global dataset median ################################################
    else:
        # Setting median  for data values
        x, y, z = dataset.data.values.T
        z = genut.arraysetmedian(z, val=median, method=setmethod)
        xyz = np.vstack((x, y, z))
        dataset.data.values = xyz.T

        # Setting median for data Z_image if any
        if dataset.data.z_image is not None:
            zimage = dataset.data.z_image
            zimage = genut.arraysetmedian(zimage, val=median, method=setmethod)
            dataset.data.z_image = zimage

    return dataset


def setmean(dataset, mean=None, profilefilt=False, valfilt=False, setmethod='additive'):
    '''
    cf. dataset.py
    '''
    # No value provided for the median #########################################
    if mean is None:
        return dataset

    # Set each profile's mean ##################################################
    if profilefilt:
        # Setting data values
        if valfilt:
            # ...TBD...
            pass

        # Setting data Z_image (if any)
        elif dataset.data.z_image is not None:
            zimage = dataset.data.z_image
            Zfilt = np.empty(zimage.shape)
            colnum = 0
            for col in zimage.T:
                Zfilt[:, colnum] = genut.arraysetmean(col, val=mean, method=setmethod)
                colnum += 1

            dataset.data.z_image = Zfilt

    # Set global dataset mean ##################################################
    else:

        # Setting mean  for data values
        x, y, z = dataset.data.values.T
        z = genut.arraysetmean(z, val=mean, method=setmethod)
        xyz = np.vstack((x, y, z))
        dataset.data.values = xyz.T

        # Setting mean  for data Z_image if any
        if dataset.data.z_image is not None:
            zimage = dataset.data.z_image
            zimage = genut.arraysetmean(zimage, val=mean, method=setmethod)
            dataset.data.z_image = zimage

    return dataset


#---------------------------------------------------------------------------#
# DataSet Compatibility checks                                              #
#---------------------------------------------------------------------------#
def check_georef_compat(dataset_list):
    '''
    Check the coordinates system compatibility of a list of datasets before merging.

    Prameters
    ---------

    dataset_list: sequence of DataSet Objects.
    '''

    active, refsystem, utm_letter, utm_number = [], [], [], []

    for dataset in dataset_list:
        active.append(dataset.georef.active)
        refsystem.append(dataset.georef.refsystem)
        utm_letter.append(dataset.georef.utm_zoneletter)
        utm_number.append(dataset.georef.utm_zonenumber)

    compat = [genut.isidentical(active), genut.isidentical(refsystem),
              genut.isidentical(utm_letter), genut.isidentical(utm_number)]

    return all(compat)


def check_gridstep_compat(dataset_list):
    '''
    Check the grid step compatibility of a list of datasets before merging.

    Prameters
    ---------
    dataset_list: tuple or list
        Sequence of DataSet Objects.

    '''

    dx, dy = [], []

    for dataset in dataset_list:
        dx.append(dataset.info.x_gridding_delta)
        dy.append(dataset.info.y_gridding_delta)

    compat = [genut.isidentical(dx), genut.isidentical(dy)]

    return all(compat)


def check_zimage_compat(dataset_list):
    '''
    Check the Z_image compatibility (i.e presence of) of a list of datasets before merging.

    Prameters
    ---------
    dataset_list: tuple or list
        Sequence of DataSet Objects.

    '''

    iszimage = []

    for dataset in dataset_list:
        iszimage.append(dataset.data.z_image is not None)

    return genut.isidentical(iszimage)

#---------------------------------------------------------------------------#
# DataSet Merging Tools                                                     #
#---------------------------------------------------------------------------#
def overlapmatching(datasets, tol=0.1, valfilt=True):
    '''
    '''

    # Mismatch symetrical matrix ###############################################
    ## The mismatch matrix is symetrical, only the upper-triangle is computed
    n = len(datasets)
    misma = np.zeros((n, n))  # dataset mismatch matrix
    triu = np.triu_indices(n, k=1)  # index matrix upper-triangle with (diagonal offset of 1)
    idxi, idxj = triu[0], triu[1]
    for k in range(len(idxi)):
        misma[idxi[k], idxj[k]] = dataset_mismatch(datasets[idxi[k]], datasets[idxj[k]], tol=tol, valfilt=True)

    tril = np.tril_indices(n, k=-1)
    misma[tril] = -misma[triu]
    print(misma)


def dataset_mismatch(dataset1, dataset2, tol=0.1, valfilt=True):
    '''
    Return the mismatch between overlapping element the two dataset.
    '''

    xyz1, xyz2, dist = dataset_overlap(dataset1, dataset2, tol=tol, valfilt=True)
    mismatch = genut.arraymismatch(xyz1[:, 2], xyz2[:, 2], weighted=True, discardspurious=True)
    
    return mismatch


def dataset_overlap(dataset1, dataset2, tol=0.1, valfilt=True):
    '''
    Return overlapping element of the two dataset.
    '''

    if valfilt:
        xyz1 = dataset1.data.values.T
        xyz2 = dataset2.data.values.T
    #...TBD...
    else:
        return [], [], []

    arr = genut.array1D_getoverlap(xyz1, xyz2, tol=tol)

    arr1 = arr[:, 0:3]  # x,y,z from array 1
    arr2 = arr[:, 4:7]  # x,y,z from array 2
    dist = arr[:, 8]  # actual distance between ovelapping points

    return arr1, arr2, dist


def matchedges(datasets, matchmethod='equalmedian', setval=None, valfilt=True, \
               setmethod='additive', meanrange=None, tol=0.1):
    '''
    Match the different datasets egdes (used before datasets merging).

    Parameters
    ----------

    :datasets: tuple or list
        Sequence of DataSet Objects. Each DataSet Object must have the same coordinates system.

    Reference
    ---------
    Eder-Hinterleitner A., Neubauer W. and Melichar P., 1996.
        Restoring magnetic anomalies.
        Archaeological Prospection, vol.3, no. 4, p185-197.

    '''

    # Ensuring datasets is a list of dataset ###################################
    datasets = list(datasets)

    # Median equalization for all the sub-datasets #############################
    if matchmethod.lower() == 'equalmedian':

        # Using the mean of the sub-datasets medians
        if setval is None:
            # Basic statistics for all sub-datasets
            datasets_stat = multidatasetstats(datasets)
            medians = datasets_stat[2]
            setvalue = np.mean(medians)  # mean of the sub-datasets medians

            ###
            ###...TDB... Should we propose the
            ## Value that gives minimum variation in every subgrid ?
            ###

        # Setting all sub-datasets to a common median value
        for dataset in datasets:
            setmedian(dataset, median=setval, profilefilt=False, valfilt=False, setmethod=setmethod)

    # Mean equalization for all the sub-datasets ###############################
    elif matchmethod.lower() == 'equalmean':

        # Using the mean of the sub-datasets means
        if setval is None:

            # 'Selective mean'
            ## Mean calculated over a specific range
            ## (see Eder-Hinterleitner et al., 1996)
            if meanrange is not None:

                # Concatenation of all sub-datasets values
                valglobal = np.array([]).reshape(-1,)

                if valfilt:
                    for dataset in datasets:
                        val = dataset.data.values[:, 2]
                        valglobal = np.hstack([valglobal, val])

                else:
                    for dataset in datasets:
                        val = dataset.data.z_image.reshape(-1,)
                        valglobal = np.vstack([valglobal, val])

                # Mid XX percent data range
                valmin, valmax = genut.arraygetmidXpercentinterval(valglobal, percent=meanrange)
                datasets_stat = multidatasetstats(datasets, valmin=valmin, valmax=valmax)

            # Classic mean of all sub-datasets
            else:
                datasets_stat = multidatasetstats(datasets)

            # mean of the sub-datasets means
            means = np.asarray(datasets_stat[0])
            setval = np.mean(means)
            ###
            ###...TDB... Should we propose the
            ## Value that gives minimum variation in every subgrid ?
            ###

        # Setting all sub-datasets to a common mean
        for dataset in datasets:
            #setmean(dataset, mean=setval, profilefilt=False, valfilt=False, setmethod=setmethod)
            ##...TDB...
            ## Is the selective mean compared to the global mean on the global medians ?
            ## In the latter case, setmedian() should be used for the offeset calculation.
            setmedian(dataset, median=setval, profilefilt=False, valfilt=False, setmethod=setmethod)


    # Edge mismatch adjustment #################################################
    ## Edge mismatch is computed between each of the sub-dataset and minimized
    ## following (Haigh J.G.B., 1992).
    else:
        pass

###...TBD...
## ??? Separate into
## MergeValues / MergeZimage / MergeHeaderso ????
###
def merge(datasetout, datasets, matchmethod=None, setval=None, setmethod='additive', \
          meanrange=None, commonprofile=False, valfilt=False):
    ''' Merge datasets together.

    cf. :meth:`~geophpy.dataset.DataSet.merge`

    '''

    # Filter ungridded values ##################################################
##    if valfilt or dataset.data.z_image is None:
##        datasets_to_merge = []
##
##        for dataset in datasets:
##            datasets_to_merge.append(dataset.copy())
##
##            # Matching dataset edges
##
##            # Merging values
##            for dataset in datasets_to_merge:
##                val = dataset.data.values
##                valmerged = np.vstack([valmerged, val])
##
##        #values = dstmp.data.values
##        #profiles = genop.arrange_to_profile(values)
##        pass

    # Filter gridded values ####################################################
##    elif not (valfilt or dataset.data.z_image is None):
##        pass

    # Checking dataset compatibilty ############################################
    compatible = all([check_gridstep_compat(datasets),
                      check_georef_compat(datasets),
                      check_zimage_compat(datasets)])
    if not compatible:
        return

    iszimage = datasets[0].data.z_image is not None  # Z_image presence flag

    # copying datasets to not alter the original data if matching needed #######
    datasets_to_merge = []
    for dataset in datasets:
        datasets_to_merge.append(dataset.copy())

    # Matching datasets edges using specific method ############################
    if matchmethod is not None:
        matchedges(datasets_to_merge, matchmethod=matchmethod, setval=setval, \
                   setmethod=setmethod, meanrange=meanrange)
    else:
        pass

    # Merging dataset values ###################################################
    ## So far the values are simply stacked together
    ## All duplicate point are kept
    ## ...TDBD... average/supress/other duplicate
    nc = datasets_to_merge[0].data.values.shape[1]
    valmerged = np.array([]).reshape(0, nc)  # empty array with nc=3 columns
    for dataset in datasets_to_merge:
        val = dataset.data.values
        valmerged = np.vstack([valmerged, val])

    datasetout.data.values = valmerged

    # Merged DataSet Object Initialization #####################################
    ## The values from the 1st dataset are used for the parameters
    ## that are common to all datasets

    # Info parameters common to all datasets
    dx = datasets_to_merge[0].info.x_gridding_delta
    dy = datasets_to_merge[0].info.y_gridding_delta
    grid_interp = datasets_to_merge[0].info.gridding_interpolation
    plot_type = datasets_to_merge[0].info.plottype
    cmap_name = datasets_to_merge[0].info.cmapname

    datasetout.info.x_gridding_delta = dx
    datasetout.info.y_gridding_delta = dy
    datasetout.info.gridding_interpolation = grid_interp
    datasetout.info.plottype = plot_type
    datasetout.info.cmapname = cmap_name

    # Data & GeoRefSystem parameters common to all datasets
    datasetout.data.fields = datasets_to_merge[0].data.fields
    datasetout.georef.active = datasets_to_merge[0].georef.active

    datasetout.georef.active = datasets_to_merge[0].georef.active
    datasetout.georef.refsystem = datasets_to_merge[0].georef.refsystem
    datasetout.georef.utm_zoneletter = datasets_to_merge[0].georef.utm_zoneletter
    datasetout.georef.utm_zonenumber = datasets_to_merge[0].georef.utm_zonenumber

    # Info & GeoRefSystem parameters different for each dataset
    xmin, xmax, ymin, ymax, zmin, zmax = [], [], [], [], [], []
    points_list = []

    # Merged DataSet Object spatial limits #####################################
    # Retreiving value for data limits
    for dataset in datasets_to_merge:
        xmin.append(dataset.info.x_min)
        xmax.append(dataset.info.x_max)
        ymin.append(dataset.info.y_min)
        ymax.append(dataset.info.y_max)
        zmin.append(dataset.info.z_min)
        zmax.append(dataset.info.z_max)
        points_list.append(dataset.georef.points_list)

    # Z_image is present
    if  iszimage:
        xmin = min(xmin)
        xmax = max(xmax)
        ymin = min(ymin)
        ymax = max(ymax)
        zmin = min(zmin)
        zmax = max(zmax)

    # No Z_image, i.e. xmin = None etc.
    ## value are kept to None
    else:
        pass

    datasetout.georef.points_list = points_list
    datasetout.info.x_min = xmin
    datasetout.info.x_max = xmax
    datasetout.info.y_min = ymin
    datasetout.info.y_max = ymax
    datasetout.info.z_min = zmin
    datasetout.info.z_max = zmax

    # No Z_image in the datasets, merge is done ! ##############################
    if not iszimage:
        return

##    # Merging dataset Info etc.
##    # Different
##    ## Data:
##    easting_image = None    # easting array
##    northing_image = None   # northing array

    # Merging dataset Z_images #################################################
    ## So far, if several data points fall into the same pixel,
    ## they are averaged.
    ## ...TBD... possibility to keep, min max or ??? if overlaping point

    # Regular grid for merged dataset
    nx = np.around((xmax - xmin)/dx) + 1
    ny = np.around((ymax - ymin)/dy) + 1

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    X, Y = np.meshgrid(x, y)

    # Initialization of the Merged grid
    Z = X * 0.
    P = Z.copy()  # number of data points in the pixel initialization

    for dataset in datasets_to_merge:
        # Current grid
        Xi, Yi = dataset.get_xygrid()
        Zi = dataset.data.z_image
        nl, nc = Zi.shape
        for i in range(nl):
            for j in range(nc):
                # Current point coordinates
                xi = Xi[i][j]
                yi = Yi[i][j]
                zi = Zi[i][j]

                # Index of the current point in the merged grid
                indx = np.where(x + dx/2. > xi)
                indy = np.where(y + dy/2. > yi)

                # Filling merged grid
                Z[indy[0][0], indx[0][0]] += zi
                P[indy[0][0], indx[0][0]] += 1

    # Averaging data points in the pixel initialization
    Z = Z/P
    datasetout.data.z_image = Z

###
##
# in the future, MOVE TO operation.spatial
##
###
#---------------------------------------------------------------------------#
# DataSet Profiles' detection Tools                                         #
#---------------------------------------------------------------------------#
def arrange_to_profile(values, constant=None):
    '''
    Re-arrange a list of points by profile based on constant coordinates.

    Only x, y values are used to re-arrange the data into profiles.
    Any additionnal values will be kept and managed properly.

    Parameters
    ----------
    values : 1-D array-like
        Array of points coordinates (and extra informations)
        >>> values = [[x1, y1, ...], [x2, y2, ...], ..., [xn, yn, ...]]

    constant : {None, 'x','y'}
        Profile's constant coordinates.
        If None, the constant coordinated id determined from the data as
        the coordinates with the less changes
        If 'x', the profile is parallel to the y-direction.
        If 'y', the profile is parallel to the x-direction.

    Return
    ------
    prof : 1-D array-like
        Array of x or y-constant profiles.
        >>> prof = [[ [x1, y1,...], [x2, y2,...] ],      # profile 1
                    [ [x21, y21,...], [x22, y22,...] ],  # profile 2
                    [ [...], ..., [xn, yn, ...] ]        # last profile
                       ]

    '''

    if constant is None:
        constant = estimate_constant_coordinate(values)

    profile_chooser = {'x' : arrange_to_xprofile,
                       'y' : arrange_to_yprofile}

    profiles = profile_chooser[constant](values)

    return profiles


def arrange_to_xprofile(values):
    ''' Re-arrange a list of values by profile based on constant x-coordinate.

    Only x values are used to re-arrange the data into profiles.
    Any additionnal values will be kept and managed properly.

    Parameters
    ----------
    values : 1-D array-like
        List of points coordinates (and extra informations)
        >>> values = [[x1, y1, ...], [x2, y2, ...], ..., [xn, yn, ...]]

    Return
    ------
    profiles : 1-D array-like
        Array of x-constant profiles.
        >>> profiles = [[ [x1, y1,...], [x2, y2,...] ],      # profile 1
                        [ [x21, y21,...], [x22, y22,...] ],  # profile 2
                        [ [...], ..., [xn, yn, ...] ]        # last profile
                       ]

    '''

    # Input as a list
    val = [list(point) for point in values]

    # Adding a fake point as last point marker
    fake_pts = [-999 for _ in range(len(val[0]))]  # [-999, ..., -999]
    val.append(fake_pts)

    # Rearranging into a profile list
    npts, profiles, profile_points = [], [], []
    xinit = val[0][0]
    for point in val:
        x, y = point[0:2]  # current point coordinates

        # Add point to the current profile point list
        if x == xinit:
            profile_points.append(point)  # coordinates + extra values

        # or create new profile
        elif x != xinit:

            # End of survey, storing last profile
            if x == -999:
                #profiles.append(point_list[:-1])
                profiles.append(profile_points)
                npts.append(len(profile_points))

            # New profile
            else:

                # Storing previous profile
                profiles.append(profile_points)
                npts.append(len(profile_points))

                # Creating a new profile
                profile_points = [] # new empty profile
                xinit = x  # new initial point
                profile_points.append(point) # adding current point to the new profile

    return profiles # np.asarray(profiles)


def arrange_to_yprofile(values):
    ''' Re-arrange points by profile based on constant y-coordinate.

    Only y values are used to re-arrange the data into profiles.
    Any additional values will be kept and managed properly.

    Parameters
    ----------
    values : 1-D array-like
        Array of points coordinates (and extra informations)
        >>> values = [[x1, y1, ...], [x2, y2, ...], ..., [xn, yn, ...]]

    Return
    ------
    profiles : 1-D array-like
        Array of y-constant profiles.
        >>> profiles = [[ [x1, y1,...], [x2, y2,...] ],      # profile 1
                        [ [x21, y21,...], [x22, y22,...] ],  # profile 2
                        [ [...], ..., [xn, yn, ...] ]        # last profile
                       ]

    '''

    # Input as a list
    val = [list(point) for point in values]

    # Adding a fake point as last point marker
    fake_pts = [-999 for _ in range(len(val[0]))]  # [-999, ..., -999]
    val.append(fake_pts)

    # Rearranging into a profile list
    npts, profiles, profile_points = [], [], []
    yinit = val[0][1]
    for point in val:
        x, y = point[0:2]  # current point

        # Add point to the current profile point list
        if y == yinit:
            profile_points.append(point)  # coordinates + extra values

        # or create new profile
        elif y != yinit:

            # End of survey, storing last profile
            if y == -999:
                #profiles.append(point_list[:-1])
                profiles.append(profile_points)
                npts.append(len(profile_points))

            # New profile
            else:

                # Storing previous profile
                profiles.append(profile_points)
                npts.append(len(profile_points))

                # Creating a new profile
                profile_points = [] # new empty profile
                yinit = y  # new initial point
                profile_points.append(point) # adding current point to the new profile

    return profiles # np.asarray(profiles)


def arrange_to_profile_from_track(values, track):
    ''' Re-arrange points by profile based on track number.

    Parameters
    ----------
    values : 1-D array-like
        Array of points coordinates (and extra informations)
        >>> values = [[x1, y1, ...], [x2, y2, ...], ..., [xn, yn, ...]]

    track : 1-D array-like
        Array of track number for each data point.
        >>> values = [1, 1, 1, ..., 2, 2, 2, ... n, n, n]

    Return
    ------
    profiles : 1-D array-like
        Array of y-constant profiles.
        >>> profiles = [[ [x1, y1,...], [x2, y2,...] ],      # profile 1
                        [ [x21, y21,...], [x22, y22,...] ],  # profile 2
                        [ [...], ..., [xn, yn, ...] ]        # last profile
                       ]

    '''

    track = np.asarray(track)
    values = np.asarray(values)
    
    profiles = []
    for num in np.unique(track):
        idx = np.where(track==num)
        profiles.append(values[idx])

    return profiles # np.asarray(profiles)


def estimate_constant_coordinate(values):
    ''' Estimate the constant coordinated of a list of points based on
    the coordinates with the less unique values.

    Parameters
    ----------
    values : 1-D array-like
        List of points coordinates (and extra informations)
        >>> values = [[x1, y1, ...], [x2, y2, ...], ..., [xn, yn, ...]]

    Return
    ------
    constant : {'x','y'}
        Estimated profile's constant coordinates.

    '''

    constant = ['x', 'y']
    xlist = np.unique(values.T[0])
    ylist = np.unique(values.T[1])
    idx = np.argmin([xlist.size, ylist.size])

    return constant[idx]

###
##
# in the future, MOVE TO operation.spatial
##
###
#---------------------------------------------------------------------------#
# DataSet Basic Affine Transformations                                      #
#---------------------------------------------------------------------------#
# def translate(dataset, shiftx=0, shifty=0):
#     ''' Dataset translation.

#     cf. :meth:`~geophpy.dataset.DataSet.translate`

#     The z value here is the actual dataset values  and not the elevation.
#     It is kept in the transformation (in place of the elevation) for
#     convenience but unchanged by the transformations.

#     '''

#     # Data values translation #################################################
#     xyz = dataset.data.values.T
#     vect = np.stack((shiftx, shifty, 0))
#     xyz = genut.array1D_translate(xyz, vect) ### in the future us spatial.array_translate
#     dataset.data.values = xyz.T


#     # Updating dataset.info (if any) ##########################################
#     if dataset.data.z_image is not None:
#         dataset.info.x_min += shiftx
#         dataset.info.x_max += shiftx
#         dataset.info.y_min += shifty
#         dataset.info.y_max += shifty

#     return dataset


def get_rotation_angle_list():
    '''
    cf. dataset.py
    '''
    return rotation_angle_list


# def rotate(dataset, angle=0, center=None):
#     '''  Dataset rotation.

#     cf. :meth:`~geophpy.dataset.DataSet.rotate`

#     The z value here is the actual dataset values  and not the elevation.
#     It is kept in the transformation (in place of the elevation) for
#     convenience but unchanged by the transformations.

#     '''

#     # Authorized rotation angle ###############################################
#     angle = np.mod(angle, 360)  # positive angle (-90->270)
#     if angle not in [0, 90, 180, 270]:
#        return dataset

#     # Data values rotation ####################################################
#     xyz = dataset.data.values.T
#     xyz = genut.array1D_rotate(xyz, angle=angle, center=center) ### in the future us spatial.array_translate
#     dataset.data.values = xyz.T

#     # Data zimage rotation ####################################################
#     if dataset.data.z_image is not None:

#         # zimage rotation
#         angleClockWise = np.mod(angle, 360)
#         k = angleClockWise//90  # number of 90° rotation (return int)
#         dataset.data.z_image = np.rot90(dataset.data.z_image, k)
#         ### ??? in the future use scipy.ndimage.rotate ???

#         # updating dataset info (xmi, ymin, ...)
#         xy = dataset.get_gridcorners()
#         xy = genut.array1D_rotate(xy, angle=angle, center=center)

#         xmin, ymin = xy.min(axis=1)
#         xmax, ymax = xy.max(axis=1)

#         dataset.info.x_min = xmin
#         dataset.info.x_max = xmax
#         dataset.info.y_min = ymin
#         dataset.info.y_max = ymax

#         x, y = dataset.get_xyvect()
# #        dx = np.median(np.diff(x))
# #        dy = np.median(np.diff(y))
# #        x_gridding_delta = dx
# #        y_gridding_delta = dy

#     return dataset


# def array_centroid(array):
#     ''' Centroid of an array of coordinates.

#     Returns the centroid of an array containing x, y and z coordinates.

#     x, y and z must be on separate lineswise
#     array([x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...]])

#     Examples:
#     --------
#     >>> x = np.random.rand(1,10)*100
#     >>> y = np.random.rand(1,10)*100
#     >>> z = np.random.rand(1,10)*100
#     >>> xyz = np.array(x, y, z)
#     >>> center = array_centroid(xyz)

#     '''

#     return np.nanmean(array, axis=1)


def translate_array(array, vect):
    ''' Translation of an array of coordinates.

    Translation of an array containing x, y and eventually z coordinates.

    Array must be ?line-wise? [[x1, x2, x3, ..., xn],
                                [y1, y2, y3, ..., xn],
                                [z1, z2, z3, ..., zn]]

    Parameters
    ----------

    array :

    vect : vector containing the shift for each dimension.

    Returns translated array.

    '''

    xyz_flag = array.shape[0] == 3 # True if array contains x, y and z

    # Checking z-dimension shift
    shiftx = vect[0]
    shifty = vect[1]
    if vect.size == 3:
        shiftz = vect[2]
    else:
        shiftz = 0

    # Adding 'false' z=0 value to the xy-array type
    xyz = array
    if not xyz_flag:
        z = np.zeros(xyz.shape[1])
        xyz = np.vstack((xyz, z))

    # Homogeneous coordinates matrix
    idvect = np.ones(xyz.shape[1])
    xyz = np.vstack((xyz, idvect))

    # Homogeneous translation matrix
    M = np.eye(4)
    M[:, -1] = np.stack((shiftx, shifty, shiftz, 1))

    # Translating data.values
    xyz = M.dot(xyz)

    # Getting rid of homogeneous coordinates
    xyz = np.delete(xyz, -1, 0)
    if not xyz_flag:
        xyz = np.delete(xyz, -1, 0)  # 'false' z=0

    return xyz


def rotate_array(array, angle=90, center=None):
    ''' Rotation of an array of coordinates.

    Clockwise rotation about the z-axis of an array containing
    x, y and eventually z coordinates.

    Array must be ?line-wise? [[x1, x2, x3, ..., xn],
                                [y1, y2, y3, ..., xn],
                                [z1, z2, z3, ..., zn]]

    Parameters
    ----------
    array : 2-D array-like

    angle : scalar
        Rotation angle in degree.

    center : {tuple of coordinates, 'BL', 'BR', 'TL', 'TR', 'centroid' or None}
        Coordinates of the center of rotation.
        If ``None`` or 'centroid', the array centroid will be used.
        If 'BL', 'BR', 'TL' or 'TR', the BottomLeft, Bottom Right,
        Top Left or Top Right To corner of the array bounding box will be used.

    Returns
    -------
    rotated array trough an angle `angle` about the point `center`.

    '''

    angle = np.mod(angle, 360)  # positive angle (-90->270)
    xyz_flag = array.shape[0] == 3  # True if array contains x, y and z

    # Center of rotation ######################################################
    # array centroid
    if center is None or center == 'centroid':
        if xyz_flag:
            center = array_centroid(array[:-1, :])
        else:
            center = array_centroid(array[:, :])

    # Bottom Left as center of rotation
    elif center.upper() in ['BL']:
        center = np.append(np.nanmin(array[0, :]), np.nanmin(array[1, :]))

    # Bottom Right as center of rotation
    elif center.upper() in ['BR']:
        center = np.append(np.nanmax(array[0, :]), np.nanmin(array[1, :]))

    # Top Left as center of rotation
    elif center.upper() in ['TL']:
        center = np.append(np.nanmin(array[0, :]), np.nanmax(array[1, :]))

    # Top Right as center of rotation
    elif center.upper() in ['TR']:
        center = np.append(np.nanmax(array[0, :]), np.nanmax(array[1, :]))

    # Given center vector of coordinates
    else:
        pass

    # Homogeneous coordinates matrix ##########################################
    # Rotation center homogeneous coordinates
    if center.size == 2:
        center = np.append(center, 0) # adding false 0 z value
    elif center.size == 3:
        center[-1] = 0 # ensuring false 0 z value

    # Adding 'false' z=0 value to the xy-array type
    xyz = array
    if not xyz_flag:
        z = np.zeros(xyz.shape[1])
        xyz = np.vstack((xyz, z))

    # Adding homogeneous coordinates to the xyz-array type
    idvect = np.ones(xyz.shape[1])
    xyz = np.vstack((xyz, idvect))

    # Homogeneous matrix of rotation (clockwise)
    A = np.radians(angle)  # from degrees to radians
    cosA = np.cos(A)
    sinA = np.sin(A)
    M = np.array([[cosA, sinA, 0, 0],
                  [-sinA, cosA, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Array rotation ##########################################################
    # Moving the center of rotation at the array centroid
    Mc = np.eye(4)
    Mc[:, -1] = np.append(-center, 1)
    xyz = Mc.dot(xyz)

    # Rotating array
    xyz = M.dot(xyz)

    # Translation to origin back to data centroid
    Mc[:, -1] = np.append(center, 1)
    xyz = Mc.dot(xyz)

    # Getting rid of homogeneous coordinates
    xyz = np.delete(xyz, -1, 0)
    if not xyz_flag:
        xyz = np.delete(xyz, -1, 0)  # 'false' z=0

    return xyz


def set_time(don,dep_0=False,sep=':'):
    """
    Convert time from a string to seconds.\n
    Input can be the full dataframe from the CMD file or simply the ``"Time"`` column.\n
    Format must be a string separated by ``sep``.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    ``[opt]`` dep_0 : bool, default : ``False``
        If first time is set to 0.
    ``[opt]`` sep : str, default : ``':'``
        Time string separator.
    
    Returns
    -------
    ls_tps_sec : dataframe column
        Updated time column.
    """ 
    ls_tps_sec=list()
    premier=0.
    if type(don)==type(pd.DataFrame()):
        # On ne prend pas la date en compte, car pas nécessaire pour départager des profils d'un même fichier
        for temps in don['Time'] :
            if type(temps)!=type('str') :
                ls_tps_sec.append(np.nan)
            else:
                Ht,Mt,St=temps.split(sep)
                h_sec=int(Ht)*3600+int(Mt)*60+float(St)
                if (dep_0 and premier==0.) :
                    premier=h_sec
                                
                ls_tps_sec.append(round(h_sec-premier,3))
            pass
        pass
    else :
        for temps in don:
            if type(temps)!=type('str') :
                ls_tps_sec.append(np.nan)
            else:
                Ht,Mt,St=temps.split(':')
                h_sec=int(Ht)*3600+int(Mt)*60+float(St)
                if (dep_0 and premier==0.) :
                    premier=h_sec
                                
                ls_tps_sec.append(round(h_sec-premier,3))
            pass
        pass
    pass
    return(ls_tps_sec)


def detect_chgt(don,colXY,verif=False):
    """
    Detect profiles (or bases) from time difference (gap means new profile).
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    colXY : [str, str]
        Name of X and Y position columns
    ``[opt]`` verif : bool, default : ``False``
        Enables plotting.
    
    Returns
    -------
    don : dataframe
        Output dataframe.
    """ 
    X,Y=don[colXY[0]],don[colXY[1]]
   
    if 'Seconds' in don.columns :
        pass
    else :
        don['Seconds']=set_time(don)
        
        
    T=don['Seconds'].copy()
    
    for indc in T.index[T.isna()]:
        T.loc[indc]=T.loc[indc-1]
        
    DT=T.diff()
    # La différence de temps donne les débuts de profils ou de base
    ind_chgtd=DT.index[DT>5*DT.median()]
    
    # La fin est l'indice avant le début donc - 1 par rapport au précédent 
    ind_chgtf=ind_chgtd-1
    
    ind_chgtd=ind_chgtd.insert(0,0)
    ind_chgtf=ind_chgtf.append(DT.index[[-1,]])
           
    don['B+P']=0
    for ic,(ind_d,ind_f) in enumerate(zip(ind_chgtd,ind_chgtf)):
        don.loc[ind_d:ind_f,'B+P']=ic+1
    
    # Plot du résultat
    if verif==True:    
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(7,7))
        ax.scatter(X.loc[ind_chgtd],Y.loc[ind_chgtd],marker='s',color='green')
        ax.scatter(X.loc[ind_chgtf],Y.loc[ind_chgtf],marker='s',color='red')
        
        ax.scatter(X,Y,marker='+',c=don['B+P'],cmap='cividis')
        ax.set_aspect('equal')
    
    return(don.copy())


def detect_base_pos(don_c, thr=2, trace=False):
    """
    Separate bases and profiles.
    
    Parameters
    ----------
    don_c : dataframe
        Active dataframe.
    thr : float
        Threshold of acceptance for bases / profiles.
    ``[opt]`` trace : bool, default : ``False``
        Enables plotting.
    
    Returns
    -------
    don_int : dataframe
        Output dataframe.
        
    Raises
    ------
    * Dataframe not interpolated
    """ 
    don_int=don_c.copy()
    if 'X_int' in don_int.columns:
        ls_coord=['X_int','Y_int']
    else:
        warnings.warn('Profile detection is better with interpolated data.')
        if 'Northing' in don_int.columns:
            ls_coord=['Northing','Easting']
        else:
            ls_coord=['x[m]','y[m]']
    if 'B+P' in don_int.columns:
        nom_col='B+P'
    else : 
        raise KeyError("Call 'detect_chgt' before this procedure.")
    
    # Un seul profil en continu
    if max(don_int['B+P']) < 2:
        don_int['Base']=0
        don_int['Profil']=1
        return(don_int.copy())
    
    don_aux=don_int.groupby(nom_col)[ls_coord].mean().round(CONFIG.prec_data)

    if trace : 
        fig, ax=plt.subplots(nrows=1,ncols=1,figsize=(CONFIG.fig_width,CONFIG.fig_height))
    
    ls_X=[]
    ls_Y=[]
    for ind_c in don_aux.index :
        X,Y=don_aux.loc[ind_c]
        if trace : 
            ax.text(X,Y,ind_c)
        ls_X.append(X)
        ls_Y.append(Y)

    if trace :
        ax.scatter(ls_X,ls_Y,marker='+',c='k')
        ax.set_aspect('equal')
    
    
    xb,yb=ls_X[0],ls_Y[0]
    ls_base,ls_prof=[1,],[]
    for ic,xyc in enumerate(zip(ls_X[1:],ls_Y[1:])):
        r=np.sqrt((xyc[0]-xb)**2+(xyc[1]-yb)**2)
        if r<=thr :
            ls_base.append(ic+2)
        else:
            ls_prof.append(ic+2)
    
    # Création des colonnes "Base" et "Profil" avec les valeurs qu'on a trouvé
    don_int['Base']=0
    for ic,ib in enumerate(ls_base):
        ind_c=don_int.index[don_int[nom_col]==ib]
        don_int.loc[ind_c,'Base']=ic+1
    don_int['Profil']=0        
    for ic,ip in enumerate(ls_prof):
        ind_c=don_int.index[don_int[nom_col]==ip]
        don_int.loc[ind_c,'Profil']=ic+1
    
    return(don_int.copy())


def num_prof(don):
    """
    Numbers each profile (or base) in chronological order.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    
    Returns
    -------
    don : dataframe
        Output dataframe.
    """ 
    if not('Base' in don.columns) :
        raise KeyError("Call 'detect_base_pos' before this procedure.")
        
    ind_mes=don.index[don['Base']==0]
    don['Profil']=0
    num_Pent=don.loc[ind_mes,'B+P'].unique()
    for ic,val in enumerate(num_Pent) :
        ind_c=don.index[don['B+P']==val]
        don.loc[ind_c,'Profil']=ic+1
    return(don.copy())
    
                
def sep_BP(don):
    """
    Split dataframe between profiles and bases.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    
    Returns
    -------
    don_base : dataframe
        Output dataframe of all bases.
    don_mes : dataframe
        Output dataframe of all profiles.
    
    Raises
    ------
    * Profiles not detected.
    """
    if not('Profil' in don.columns):
        raise KeyError("Call 'num_prof' before this procedure.")
    else :
        ind_p=don.index[don['Profil']!=0]
        ind_b=don.index[don['Base']!=0]
        return(don.loc[ind_b],don.loc[ind_p])


def detect_profile_square(don):
    """
    Detect profiles (or bases) from X coordinates (data without GPS only).\n
    Bases must be marked with negative coordinates.
    
    Notes
    -----
    ``"Seconds"`` column is created but set to ``-1`` (placeholder).
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    
    Returns
    -------
    don : dataframe
        Output dataframe.
    """
    don["B+P"] = 0
    don["Base"] = 0
    don["Profil"] = 0
    don["Seconds"] = -1
    colname = don.columns[0]
    x = np.nan
    base_nb = 0
    prof_nb = 0
    # On cherche simplement à séparer les points selon leur coordonnée x
    # (peut être remplacé par un groupby ?)
    for index, row in don.iterrows():
        x_l = don[colname].iloc[index]
        if x != x_l and x_l == x_l:
            x = x_l
            if x < 0:
                base_nb += 1
            else:
                prof_nb += 1
        if x < 0:
            don.loc[index, "Base"] = base_nb
            don.loc[index, "Profil"] = 0
        else:
            don.loc[index, "Base"] = 0
            don.loc[index, "Profil"] = prof_nb
        don.loc[index, "B+P"] = base_nb + prof_nb
    return don.copy()


def detect_pseudoprof(don,x_col,y_col,l_p=None,tn=10,tn_c=20,min_conseq=8,plot=False):
    """
    Given a database with continuous timestamps, estimate profiles by finding one point
    (called `center`) per profile, possibly at the center.\n
    Each prospection point is then assigned to the closest center in term of index to form pseudo-profiles.\n
    By default, perform a linear regression and select the closest points as centers.
    For more flexibility, one can set a range of points which will create segments with ``l_p``.\n
    This procedure is to be used only if no other profile detection method has been successful.
    
    Notes
    -----
    It is advised to check if the result is coherent by setting ``verif = True``.\n
    If many centers are missing, raise ``tn`` and / or ``tn_c``.
    On the contrary, lowering them also works.\n
    If you can expect profiles of less than 8 points, try lowering ``min_conseq``.
    If some profiles are detected twice, you can raise it.\n
    Profiles found this way are not neccesarely straight.
    
    Parameters
    ----------
    don : dataframe
        Profile dataframe.
    x_col : str
        Name of X column.
    y_col : str
        Name of Y column.
    ``[opt]`` l_p : ``None`` or list of [float, float], default : ``None``
        List of points coordinates for segments. If ``None``, perform a linear regression instead.
    ``[opt]`` tn : int, default : ``10``
        Number of nearest points used to determinate the max distance treshold.
    ``[opt]`` tn_c : int, default : ``20``
        Multiplier of median distance of the ``tn`` nearest points used to determinate the max distance treshold.
    ``[opt]`` min_conseq : int, default : ``8``
        Minimal index distance that is allowed between two found centers.
    ``[opt]`` plot : bool, default : ``False``
        Enables plotting.

    Returns
    -------
    don : dataframe
        Updated profile dataframe
    
    Raises
    ------
    * ``l_p`` vector too small.
    
    See also
    --------
    ``init, detect_profile_square, detect_chgt`` 
    """
    # Pour être propre, on enlève les anciennes colonnes obsolètes
    for label in ["Profile","Base","B+P"]:
        try:
            don.drop(columns=label)
        except:
            pass
    
    # Colonnes position manquantes
    try:
        don[x_col], don[y_col]
    except KeyError:
        raise KeyError('Columns "{}" and/or "{}" do not exist.'.format(x_col,y_col))
    
    nb_pts = len(don)
    # Si aucun point n'est spécifié, on prendra la droite de régression comme référence
    regr = (l_p == None)
    
    dist_list = []
    
    # Régression
    if regr:
        lin_tab_i = np.array(don.index)
        lin_tab_x = np.array(don[x_col])
        lin_tab_y = np.array(don[y_col])
        # X par rapport à l'indice
        lin_reg_x = linregress(lin_tab_i,lin_tab_x)
        # Y par rapport à l'indice
        lin_reg_y = linregress(lin_tab_i,lin_tab_y)
        
        # Coefficients trouvés par les régressions
        eq = [lin_reg_x.slope, lin_reg_x.intercept, lin_reg_y.slope, lin_reg_y.intercept]
        
        # Expression de la distance entre un point et la droite,
        # à une constante multiplicative près (bon pour les rapports)
        for index, row in don.iterrows():
            dist_list.append(np.abs(eq[2]*(row[x_col]-eq[1]) - eq[0]*(row[y_col]-eq[3])))
    # Distance aux segments
    else:
        l_p = np.array(l_p)
        # Il faut au moins deux poins pour construire un segment
        if len(l_p) < 2:
            raise ValueError("Length of 'l_p' must be at least 2 to define one segment ({}).".format(len(l_p)))
        # Pour chaque point c/p3 du jeu de données...
        for index, row in don.iterrows():
            d_l = []
            # Pour chaque segment, on prend ses deux extrémités a/p1 et b/p2
            for p1,p2 in zip(l_p,l_p[1:]):
                p3 = np.array([row[x_col],row[y_col]])
                ba = p1 - p2
                lba = np.linalg.norm(ba)
                bc = p3 - p2
                lbc = np.linalg.norm(bc)
                angle_1 = np.degrees(np.arccos(np.dot(ba, bc) / (lba * lbc)))
                # Si l'angle cba >= 90°, alors la distance est simplement égale à [cb]
                if angle_1 >= 90.0:
                    d_l.append(lbc)
                    continue
                ac = p3 - p1
                lac = np.linalg.norm(ac)
                ab = -ba
                lab = lba
                angle_2 = np.degrees(np.arccos(np.dot(ab, ac) / (lab * lac)))
                # Si l'angle cab >= 90°, alors la distance est simplement égale à [ca]
                if angle_2 >= 90.0:
                    d_l.append(lac)
                    continue
                # Sinon, on calcule la distance entre c et la droite (ab)
                d_l.append(np.abs(np.cross(ab,ac)/lba))
            dist_list.append(min(d_l))
    
    ### Sélection des centres des pseudo-profils ###
    # Filtre 1 : Sélection des minimums locaux de la droite des distances
    m1_list = []
    up = True
    for ic,d in enumerate(dist_list[:-1]):
        new_up = (d < dist_list[ic+1])
        if new_up and not up:
            m1_list.append(ic)
        up = new_up
    #print(m1_list)
    
    # Filtre 2 : Suppression des minimums locaux trop éloignés, à partir de la distance des meilleurs points
    m2_list = []
    top_n = sorted([dist_list[i] for i in m1_list], key = lambda x: x, reverse = False)[:tn]
    min_med = sum(top_n)/tn * tn_c
    
    for m in m1_list:
        #print(dist_list[m], " ", min_med)
        if dist_list[m] <= min_med:
            m2_list.append(m)
    #print(m2_list)
    
    # Filtre 3 : Suppression des points doubles (quasi-adjacentsdans l'ordre),
    # et ajout de points pour découper les profils trop gros (l_p == None uniquement)
    min_list = []
    if regr:
        max_conseq = (2*nb_pts)//len(m2_list)
    else:
        max_conseq = np.inf
    l_min = len(m2_list)
    i = 0
    j = 1
    while j < l_min:
        d = m2_list[j] - m2_list[i]
        if d > max_conseq:
            min_list.append(m2_list[i])
            nb_new_points = int(d*2//max_conseq)-1
            for n in range(1,nb_new_points+1):
                min_list.append(m2_list[i]+((m2_list[j]-m2_list[i])*n)//(nb_new_points+1))
            i = j
            j += 1
        if d < min_conseq:
            if dist_list[m2_list[j]] < dist_list[m2_list[i]]:
                min_list.append(m2_list[j])
                i = j
            elif m2_list[i] not in min_list:
                min_list.append(m2_list[i])
            j += 1
        else:
            if m2_list[i] not in min_list:
                min_list.append(m2_list[i])
            i = j
            j += 1
    if i == l_min-1:
        min_list.append(m2_list[-1])
    
    # Pour chaque point du jeu, attribue le centre le plus proche par index
    # Chaque centre correspond à un pseudo-profil
    don["Profil"] = 0
    don["Base"] = 0
    l_min = len(min_list)
    for index, row in don.iterrows():
        ind = -1
        for ic, m in enumerate(min_list[1:]):
            if m > index:
                if m-index > index-min_list[ic]:
                    ind = ic+1
                else:
                    ind = ic+2
                break
        if ind == -1:
            ind = l_min
        don.loc[index,"Profil"] = ind   
    don["B+P"] = don["Profil"]
    
    # Plot du résultat
    if plot:
        #print(min_list)
        index_list = range(nb_pts)
        fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(CONFIG.fig_width,CONFIG.fig_height),squeeze=False)
        # Axe 1 : Évolution de la distance en fonction de l'index et affichage des centres trouvés
        ax[0][0].plot(index_list,dist_list,'-')
        ax[0][0].plot([index_list[i] for i in min_list],\
                      [dist_list[i] for i in min_list],'xr')
        ax[0][0].set_xlabel("Index")
        ax[0][0].set_ylabel("Distance")
        ax[0][0].set_title("Distance from line/segments")
        # Axe 2 : Affichage du nuage de point des données, droite/segments et des centres trouvés
        ax[0][1].plot(don[x_col],don[y_col],'x')
        if regr:
            ax[0][1].plot([eq[0]*i+eq[1] for i in index_list],[eq[2]*i+eq[3] for i in index_list],'-k')
        else:
            ax[0][1].plot(l_p[:,0],l_p[:,1],'-k')
        ax[0][1].plot([don[x_col][index_list[i]] for i in min_list],\
                      [don[y_col][index_list[i]] for i in min_list],'xr')
        ax[0][1].set_aspect('equal')
        ax[0][1].set_xlabel(x_col)
        ax[0][1].set_ylabel(y_col)
        ax[0][1].set_title("Centers of pseudo-profiles")
        ax[0][1].ticklabel_format(useOffset=False)
        # Axe 2 : Affichage des pseudo-profils finaux, par couleur
        ax[0][2].scatter(don[x_col],don[y_col],marker='x',c=don["Profil"]%8, cmap='nipy_spectral')
        ax[0][2].set_aspect('equal')
        ax[0][2].set_xlabel(x_col)
        ax[0][2].set_ylabel(y_col)
        ax[0][2].set_title("Found division of pseudo-profiles")
        ax[0][2].ticklabel_format(useOffset=False)
        plt.show(block=False)
        # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser
        # pour accélérer la vitesse de l'input
        plt.pause(CONFIG.fig_render_time)
    
    return don.copy()


def intrp_prof(don_mes,x_col,y_col):
    """
    Interpolate groups of points of same coordinates by linear regression.
    Used if the GPS refresh time is slower than the actual prospection.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    x_col : str
        Name of X column.
    y_col : str
        Name of Y column.
    
    Returns
    -------
    don_mes : dataframe
        Output dataframe of all profiles.
    
    Raises
    ------
    * Profiles not detected.
    """
    colXY=[x_col,y_col]
        
    don_mes['X_int']=0.
    don_mes['Y_int']=0.
    if  'B+P' in don_mes.columns:
        num_profs=don_mes['B+P'].unique()
        nom_col='B+P'
    elif 'Profil' in don_mes.columns :
        num_profs=don_mes['Profil'].unique()
        nom_col='Profil'
    else : 
        raise KeyError("Call 'detect_base_pos' before this procedure.")
        # on parcours chaque profil
    for num_p in num_profs :
        ind_prof=don_mes.index[don_mes[nom_col]==num_p]
        prof_c=don_mes.loc[ind_prof,colXY]
        prof_c.columns=['X','Y']
  
        dxdy=prof_c.diff()
        dxdy['dr']=np.sqrt((dxdy**2).sum(axis=1))
        dxdy.loc[:,'dr']=np.round(dxdy.loc[:,'dr'],CONFIG.prec_data)
        # pour récupérer le premier index sous forme d'index (et pas d'entier)
        # il faut indiquer .index[0:1]intrp
        ind_ancre=dxdy.index[0:1].append(dxdy.index[dxdy['dr']!=0])
        if ind_ancre[-1]!=dxdy.index[-1] :
            ind_ancre=ind_ancre.append(dxdy.index[-1:])
        ind_ancd=ind_ancre[0:-1]
        ind_ancf=ind_ancre[1:]
        nbpts=(ind_ancf-ind_ancd).array
        ls_dX,ls_dY=[],[]
        # On crée les listes de décalage à appliquer à chaque points en X et Y
        # la fin du profil est gérée en prenant les décalages précédents et en faisant
        for ic,nbp in enumerate(nbpts):
            dernier=len(nbpts)
            fin=dxdy.loc[ind_ancf[ic:ic+1],['X','Y']].to_numpy().flatten()
            
            if np.array_equal(fin,np.array([0.,0.])):
                fin=dxdy.loc[ind_ancf[ic-1:ic],['X','Y']].to_numpy().flatten()
                int_c=np.linspace([0,0],fin,nbp+1)
            else:
                int_c=np.linspace([0,0],fin,nbp+1)
                int_c[-1,:]=np.array([0.,0.])
                   
            if ic>0 :
                if ic<dernier-1 :
                    ls_dX+=int_c[:-1,0].tolist()
                    ls_dY+=int_c[:-1,1].tolist()
                else :
                    ls_dX+=int_c[:,0].tolist()
                    ls_dY+=int_c[:,1].tolist()
            else :
                ls_dX=int_c[:-1,0].tolist()
                ls_dY=int_c[:-1,1].tolist()
        try:
            prof_i=(prof_c+np.array([ls_dX,ls_dY]).T).to_numpy()
        except:
            prof_i = don_mes.loc[ind_prof,colXY]

        don_mes.loc[ind_prof,['X_int','Y_int']] = prof_i
    return(don_mes)


def XY_Nan_completion(don,x_col="X_int",y_col="Y_int"):
    """
    Estimates ``NaN`` points coordinates by linear regression on associated profile.
    Others are left unchanged.
    
    Notes
    -----
    Procedure is cancelled if NaN on X and Y are from different positions
    (should be an issue of column splitting).\n
    Profiles of only one known point can't be interpolated.
    Profiles of only two known point are handled by creating a third middle point (otherwise glitchy).\n
    Interpolation also corrects time.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    ``[opt]`` x_col : str, default : ``"X_int"``
        X column label
    ``[opt]`` y_col : str, default : ``"Y_int"``
        Y column label
        
    
    Returns
    -------
    don : dataframe
        Output dataframe.
    
    See Also
    --------
    ``init, pts_rectif``
    """
    X = don[x_col]
    Y = don[y_col]
    indXNan=X.index[X.isna()]
    indYNan=Y.index[Y.isna()]
    if len(indXNan)<1:
        return don.copy()
    if np.all(indXNan==indYNan):
        indc=indXNan.copy()
    else:
        raise ValueError("NaNs on X and Y are not of same indexes.")
    
    ind_aux = indc[indc.diff()!=1.]-1
    
    for i_d in ind_aux:
        prof = don.loc[i_d,'B+P']
        bloc = don.loc[don['B+P'] == prof]
        bloc_notna = bloc.dropna(subset = [x_col,y_col])
        bloc_na = bloc.loc[bloc.index.difference(bloc.dropna(subset = [x_col,y_col]).index)]
        bloc_notna_l = bloc_notna.shape[0]
        
        if bloc_notna_l == 1:
            warnings.warn("One profile has only one known point : no regression possible.")
        else:
            lin_tab_i = np.array(bloc_notna.index)
            lin_tab1 = np.array(bloc_notna["Seconds"])
            lin_tab2 = np.array(bloc_notna[x_col])
            lin_tab3 = np.array(bloc_notna[y_col])
            # La régression ne marche pas avec deux points, mais on peut en créer un troisième
            if bloc_notna_l == 2:
                lin_tab1 = np.concatenate([lin_tab1,[sum(lin_tab1[:,1])/len(lin_tab1[:,1])]])
                lin_tab2 = np.concatenate([lin_tab2,[sum(lin_tab2[:,1])/len(lin_tab2[:,1])]])
                lin_tab3 = np.concatenate([lin_tab3,[sum(lin_tab3[:,1])/len(lin_tab3[:,1])]])
            
            # On fait toutes les régressions par rapport à l'indice,
            # car il est représentatif d'un pas de temps régulier sur un même profil
            lin_reg1 = linregress(lin_tab_i,lin_tab1)
            lin_reg2 = linregress(lin_tab_i,lin_tab2)
            lin_reg3 = linregress(lin_tab_i,lin_tab3)

            for index, row in bloc_na.iterrows():
                c = lin_reg1.intercept + lin_reg1.slope*index
                don.loc[index, "Seconds"] = c
                c = lin_reg2.intercept + lin_reg2.slope*index
                don.loc[index, x_col] = c
                c = lin_reg3.intercept + lin_reg3.slope*index
                don.loc[index, y_col] = c
        
    
    print('NaNs completed : {}'.format(len(indXNan)))
    return don.copy()


def XY_Nan_completion_solo(don,x_col="X_int",y_col="Y_int"):
    """
    Estimates ``NaN`` points coordinates by linear regression from neighbors.
    Others are left unchanged.
    
    Notes
    -----
    Procedure is cancelled if NaN on X and Y are from different positions 
    (should be an issue of column splitting).\n
    Is meant to be used if the prospection is not splitted in profiles.
    Otherwise, see ``XY_Nan_completion``.\n
    Interpolation also corrects time.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    ``[opt]`` x_col : str, default : ``"X_int"``
        X column label
    ``[opt]`` y_col : str, default : ``"Y_int"``
        Y column label
    
    Returns
    -------
    don : dataframe
        Output dataframe.
    
    See Also
    --------
    ``init, XY_Nan_completion``
    """
    X = don[x_col]
    Y = don[y_col]
    indXNan=X.index[X.isna()]
    indYNan=Y.index[Y.isna()]
    if len(indXNan)<1:
        print('No interpolation needed.')
        return don.copy()
    if not np.all(indXNan==indYNan):
        raise ValueError("NaNs on X and Y are not of same indexes.")
    
    full_na = don.iloc[indXNan,:]
    bloc_na_list = [d for _, d in full_na.groupby(full_na.index - np.arange(len(full_na)))]
    #print(bloc_na_list)
    
    # Complétion faite pour chaque bloc de donnée manquant
    for bloc_na in bloc_na_list:
        bnai = bloc_na.index
        l_n = bnai.size
        if bnai[0] == 0: # Bloc au début
            row1 = don.iloc[bnai[-1]+1]
            row2 = don.iloc[bnai[-1]+2]
            pas_x = row2[x_col] - row1[x_col]
            pas_y = row2[y_col] - row1[y_col]
            new_x = [row1[x_col] - pas_x*i for i in range(1,l_n+1,-1)]
            new_y = [row1[y_col] - pas_y*i for i in range(1,l_n+1,-1)]
        elif bnai[-1] == len(don)-1: # Bloc à la fin
            row1 = don.iloc[bnai[0]-2]
            row2 = don.iloc[bnai[0]-1]
            pas_x = row2[x_col] - row1[x_col]
            pas_y = row2[y_col] - row1[y_col]
            new_x = [row2[x_col] + pas_x*i for i in range(1,l_n+1)]
            new_y = [row2[y_col] + pas_y*i for i in range(1,l_n+1)]
        else: # Cas général
            row1 = don.iloc[bnai[0]-1]
            row2 = don.iloc[bnai[-1]+1]
            pas_x = (row2[x_col] - row1[x_col]) / (l_n+1)
            pas_y = (row2[y_col] - row1[y_col]) / (l_n+1)
            new_x = [row1[x_col] + pas_x*i for i in range(1,l_n+1)]
            new_y = [row1[y_col] + pas_y*i for i in range(1,l_n+1)]
        don.loc[bnai, x_col] = new_x
        don.loc[bnai, y_col] = new_y
    
    print('Valeurs remplacées : {}'.format(len(indXNan)))
    return don.copy()


def pts_rectif(don,x_col="X_int",y_col="Y_int",ind_deb=None,ind_fin=None):
    """
    Estimates all points coordinates of same profile by linear regression.\n
    To be used if the GPS error is too important.
    
    Notes
    -----
    Profiles of only one known point can't be interpolated.
    Profiles of only two known point are handled by creating a third middle point (otherwise glitchy).
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    ``[opt]`` x_col : str, default : ``"X_int"``
        X column label
    ``[opt]`` y_col : str, default : ``"Y_int"``
        Y column label
    ``[opt]`` ind_deb : ``None`` or int, default : ``None``
        Index of first profile to interpolate. ``None`` for all.
    ``[opt]`` ind_fin : ``None`` or int, default : ``None``
        Index of last profile to interpolate. ``None`` for all.
    
    Returns
    -------
    don : dataframe
        Output dataframe.
    
    See Also
    --------
    ``XY_Nan_completion``
    """
    ind_aux = []
    cpt = -1
    for index, row in don.iterrows():
        if row["B+P"] != cpt:
            cpt = row["B+P"]
            ind_aux.append(index)

    for i_d in ind_aux[ind_deb:ind_fin]:
        prof = don.loc[i_d,'B+P']
        bloc = don.loc[don['B+P'] == prof]
        bloc_l = bloc.shape[0]
        
        if bloc_l == 1:
            warnings.warn("One profile has only one point : no regression needed.")
        else:
            lin_tab_i = np.array(bloc.index)
            # lin_tab1 = np.array(bloc["Seconds"])
            lin_tab2 = np.array(bloc[x_col])
            lin_tab3 = np.array(bloc[y_col])
            # La régression ne marche pas avec deux points, mais on peut en créer un troisième
            if bloc_l == 2:
                # lin_tab1 = np.concatenate([lin_tab1,[sum(lin_tab1[:,1])/len(lin_tab1[:,1])]])
                lin_tab2 = np.concatenate([lin_tab2,[sum(lin_tab2[:,1])/len(lin_tab2[:,1])]])
                lin_tab3 = np.concatenate([lin_tab3,[sum(lin_tab3[:,1])/len(lin_tab3[:,1])]])
            # On fait toutes les régressions par rapport à l'indice, 
            # car il est représentatif d'un pas de temps régulier sur un même profil
            # lin_reg1 = linregress(lin_tab1)
            lin_reg2 = linregress(lin_tab_i,lin_tab2)
            lin_reg3 = linregress(lin_tab_i,lin_tab3)
            
            for index, row in bloc.iterrows():
                # c = lin_reg1.intercept + lin_reg1.slope*index
                # don.loc[index, "Seconds"] = c
                c = lin_reg2.intercept + lin_reg2.slope*index
                don.loc[index, x_col] = c
                c = lin_reg3.intercept + lin_reg3.slope*index
                don.loc[index, y_col] = c
    
    return don.copy()


def decal_posLT(X,Y,profs,decL=0.,decT=0.):
    """
    Shifts X and Y according to the GPS and coil position from the center.
    
    Notes
    -----
    If no GPS, only the coil shift will be taken into account.
    
    Parameters
    ----------
    X : dataframe column
        X coordinates.
    Y : dataframe column
        Y coordinates.
    profs : dataframe column
        Indexes of profiles.
    ``[opt]`` decL : float, default : ``0.0``
        Total shift on device axis, with direction from first to last coil.
    ``[opt]`` decT : float, default : ``0.0``
        Total shift on device perpendicular axis, with direction from behind to front.
    
    Returns
    -------
    Xc : dataframe column
        Shifted X.
    Yc : dataframe column
        Shifted Y.
    
    See Also
    --------
    ``dec_channels``
    """
    ls_Xc=[]
    ls_Yc=[]
    ls_prof=profs.unique()
    #print(ls_prof)
    for prof in ls_prof:
        ind_c=profs.index[profs==prof]
        XX=X.loc[ind_c].copy()
        YY=Y.loc[ind_c].copy()
        DX=XX.diff()
        DY=YY.diff()
        DX1=DX.copy()
        DY1=DY.copy()
        if len(DX1) == 1:
            ls_Xc.append(XX)
            ls_Yc.append(YY)
            continue
    # pour avoir quelque chose de pas trop moche on fait la moyenne des 
    # décalages avec mouvement avant le point et après le point   
        DX.iloc[0:-1]=DX.iloc[1:]      
        DY.iloc[0:-1]=DY.iloc[1:]
        DX1.iloc[0]=DX1.iloc[1]
        DY1.iloc[0]=DY1.iloc[1]
        
        DR=np.sqrt(DX*DX+DY*DY)       
        DR1=np.sqrt(DX1*DX1+DY1*DY1)
        
        CDir=(DX/DR+DX1/DR1)/2.
        SDir=(DY/DR+DY1/DR1)/2.
        
        decX=CDir*decL-SDir*decT
        decY=SDir*decL+CDir*decT
        
        X1=XX+decX
        Y1=YY+decY
        ls_Xc.append(X1)
        ls_Yc.append(Y1)
       
    Xc=pd.concat(ls_Xc) 
    Yc=pd.concat(ls_Yc)   
    return(Xc,Yc)


def dec_channels(don,ncx,ncy,nb_channels,TR_l,TR_t,gps_dec):
    """
    Shifts X and Y according to the GPS and coil position from the center, for each coil.
    
    Notes
    -----
    If no GPS, only the coil shift will be taken into account.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    ncx : list of str
        Names of every X columns.
    ncy : list of str
        Names of every Y columns.
    nb_channels : int
        Number of X and Y columns. The number of coils.
    TR_l : list of float
        Distance between each coil and the transmitter coil, on lateral axis (m).
    TR_t : list of float
        Distance between each coil and the transmitter coil, on transversal axis (m).
    gps_dec : [float, float]
        Shift vector between the GPS antenna and the device center, on both axis (m). Should be ``[0,0]`` if none.
    
    Returns
    -------
    don : dataframe
        Output dataframe.
    
    See Also
    --------
    ``decal_posLT``
    """
    # Pour chaque voie, on décale la position
    for e in range(nb_channels):
        decx = gps_dec[0]-(TR_l[e]-TR_l[-1])/2
        decy = gps_dec[1]-(TR_t[e]-TR_t[-1])/2
        X, Y = decal_posLT(don["X_int"],don["Y_int"],don["Profil"],decL=decx,decT=decy)
        don[ncx[e]] = X.round(CONFIG.prec_coos)
        don[ncy[e]] = Y.round(CONFIG.prec_coos)
    return don.copy()


def appr_border(x1,x2,y1,y2,i_max,j_max,i_excl,j_excl):
    """
    Find one point of each dataframe that are close to each other.\n
    They may not be included in the exclusion lists ``i_excl`` and ``j_excl`` to avoid duplicates.
    
    Notes
    -----
    This function does not provide the global minimum of distance, it converges to a fair enough pair.\n
    It runs through both lists and remove far points one by one (linear complexity).\n
    Starting points are taken randomly.
    
    Parameters
    ----------
    x1 : list of float
        X coordinates of first dataframe.
    x2 : list of float
        X coordinates of second dataframe.
    y1 : list of float
        Y coordinates of first dataframe.
    y2 : list of float
        Y coordinates of second dataframe.
    i_max : int
        Number of points of first dataframe.
    j_max : int
        Number of points of second dataframe.
    i_excl : list of int
        Exclusion list (indexes) of points of first dataframe.
    j_excl : list of int
        Exclusion list (indexes) of points of second dataframe.
    
    Returns
    -------
    i_min : int
        Selected point (index) of first dataframe.
    j_min : int
        Selected point (index) of second dataframe.
    d_min : float
        Distance between ``i_min`` and ``j_min``.
    
    See Also
    --------
    ``calc_frontier``
    """
    i_dec = np.random.randint(0, i_max)
    j_dec = np.random.randint(0, j_max)
    while i_dec in i_excl:
        i_dec = np.random.randint(0, i_max)
    while j_dec in j_excl:
        j_dec = np.random.randint(0, j_max)
    i_min = i_dec
    j_min = j_dec
    i = 0
    j = 0
    i_ = i_dec - i_max
    j_ = j_dec - j_max
    #print("i_ = ",i_," | j_ = ",j_)
    d_min = (x1[i_min]-x2[j_min])**2 + (y1[i_min]-y2[j_min])**2
    turn = True
    
    # On parcourt les ensembles dans l'ordre de l'index à partir de points de départ aléatoires p1 et p2.
    # On parcourt le premier jeu jusqu'à trouver un point plus proche de p2 que p1.
    # Dans ce cas, il devient p1, et on itère sur l'autre jeu.
    # La démarche est identique avec p2.
    # On continue jusqu'à avoir exploré une fois tous les points des ensembles.
    # Complexité : O(n)
    while i < i_max or j < j_max:
        if turn:
            d = (x1[i_]-x2[j_min])**2 + (y1[i_]-y2[j_min])**2
            if d < d_min and i_%(i_max+1) not in i_excl:
                if j != j_max:
                    turn = False
                #print("|i = {}, i_min = {}, j = {}, j_min = {}, d = {}".format(i,i_min,j,j_min,d))
                d_min = d
                i_min = i_%(i_max+1)
            elif i == i_max:
                turn = False
            else:
                i+=1
                i_+=1
        else:
            d = (x1[i_min]-x2[j_])**2 + (y1[i_min]-y2[j_])**2
            if d < d_min and j_%(j_max+1) not in j_excl:
                if i != i_max:
                    turn = True
                #print("_i = {}, i_min = {}, j = {}, j_min = {}, d = {}".format(i,i_min,j,j_min,d))
                d_min = d
                j_min = j_%(j_max+1)
            elif j == j_max:
                turn = True
            else:
                j+=1
                j_+=1
    # print("i_min = {}, j_min = {}, d_min = {}".format(i_min,j_min,d_min))
    # print("pt 1 = [{},{}]".format(x1[i_min],y1[i_min]))
    # print("pt 2 = [{},{}]".format(x2[j_min],y2[j_min]))
    # print("--------------------------------------------------------------")
    return i_min,j_min,d_min


def appr_distmoygrp(x1,y1):
    """
    Compute the mean distance of points from same dataframe.
    """
    l = len(x1)-1
    d = 0
    for i in range(l):
        d = d + (x1[i]-x1[i+1])**2 + (y1[i]-y1[i+1])**2
    return d / l


def appr_taille_grp(x1,y1):
    """
    Compute the diagonal distance of one dataframe.
    """
    x_min = min(x1)
    x_max = max(x1)
    y_min = min(y1)
    y_max = max(y1)
    d = (x_max-x_min)**2 + (y_max-y_min)**2
    return d


def max_frontier(x1,y1,excl):
    """
    Compute the maximal distance of points from same dataframe in selected list.
    """
    d_max = 0
    for i in excl[:-1]:
        for j in excl[1:]:
            d = (x1[i]-x1[j])**2 + (y1[i]-y1[j])**2
            d_max = max(d_max,d)
    return d_max


def compute_coeff(col1,col2,excl1,excl2):
    """
    Compute a and b in the adjustment relation of type a + bx between both dataframes.
    """
    sig1 = np.std([col1[i] for i in excl1])
    sig2 = np.std([col2[j] for j in excl2])
    
    t = len(excl1)
    diff = 0
    ec = sig1/sig2
    for j in range(t):
        diff = diff + col1[excl1[j]] - (col2[excl2[j]])*ec
    
    return diff/t, ec


def dat_to_grid(don,ncx,ncy,nb_channels,nb_res,radius=0,prec=100,step=None,w_exp=0.0,
                only_nan=True,heatmap=False,verif=False):
    """
    Put raw data on a grid, then determine which tile should be removed (with ``NaN`` value).\n
    Removal procedure is detailled in the ``calc_coeff`` description.
    If ``heatmap = True``, launch a trial process giving the effective selected
    grid area for a given ``w_exp`` value.
    
    Notes
    -----
    Complexity :
        .. math:: O(d + p^2r^2) 
        where d is the number of points, p is ``prec`` and r is ``radius``.
        
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    ncx : list of str
        Names of every X columns.
    ncy : list of str
        Names of every Y columns.
    nb_channels : int
        Number of X and Y columns. The number of coils.
    nb_res : int
        The number of data per coil.
    ``[opt]`` radius : int, default : ``0``
        Detection radius around each tile for ``NaN`` completion.
    ``[opt]`` prec : int, default : ``100``
        Grid size of the biggest axis. The other one is deducted by proportionality.
    ``[opt]`` step : ``None`` or float, default : ``None``
        Step between each tile, according to the unit used by the position columns.
        If not ``None``, ignore ``prec`` value.
    ``[opt]`` w_exp : float, default : ``0.0``
        Exponent of the function used to compute the detection window coefficients.
        If negative, will be set to 0 but widen the acceptance.
    ``[opt]`` only_nan : bool, default : ``True``
        If ``True``, tiles that contain at least one point are always kept.
        If ``False``, will remove those that are too eccentric.
    ``[opt]`` heatmap : bool, default : ``False``
        If we compute the heatmap instead of the regular grid.
    ``[opt]`` verif : bool, default : ``False``
        Print some useful informations for testing.
    
    
    Returns
    -------
    grid_final : np.ndarray (dim 3) of float
        For each data column, contains the grid values (``0`` if tile is taken, ``NaN`` if not)
    ext : [float, float, float, float]
        Extend of the grid. Contains ``[min_X, max_X, min_Y, max_Y]``.
    pxy : [float, float]
        Size of the grid for each axis. Contains ``[prec_X, prec_Y]``.
    
    Raises
    ------
    * Some columns does not exist.
    * Some columns are not numeric.
    
    See also
    --------
    ``interp_grid, heatmap_grid_calc, heatmap_plot``
    """
    print("=== Grid construction phase ===")
    
    # Calcul des dimensions de la grille
    try:
        X = np.array(don[ncx])
        Y = np.array(don[ncy])
    except KeyError:
        raise KeyError('Columns "{}" and "{}" do not exist.'.format(ncx,ncy))

    
    try:
        max_X = max(X.flatten())
        min_X = min(X.flatten())
        max_Y = max(Y.flatten())
        min_Y = min(Y.flatten())
    except TypeError:
        raise TypeError('Columns "{}" and "{}" are not numeric.'.format(ncx,ncy))
    if verif:
        print("max_X = ",max_X)
        print("min_X = ",min_X)
        print("max_Y = ",max_Y)
        print("min_Y = ",min_Y)
    diff_X = max_X-min_X
    diff_Y = max_Y-min_Y
    if step == None:
        if diff_X > diff_Y:
            prec_X = prec
            prec_Y = int(prec*(diff_Y/diff_X))
        else:
            prec_Y = prec
            prec_X = int(prec*(diff_X/diff_Y))
        pas_X = diff_X/prec_X
        pas_Y = diff_Y/prec_Y
    else:
        pas_X = step
        pas_Y = step
        prec_X = int(diff_X/pas_X)+1
        prec_Y = int(diff_Y/pas_Y)+1
    
    # Coordonnées de la grille
    gridx = [min_X + pas_X*i for i in range(prec_X)]
    gridy = [min_Y + pas_Y*j for j in range(prec_Y)]
    
    # Calcul de la fenêtre pour la grille d'interpolation
    grid_conv, w_exp, w_exp_, quot = calc_coeff(w_exp,radius)
    
    # Grille d'interpolation vide
    grid = np.array([[[0 for i in range(prec_X)] for j in range(prec_Y)] for e in range(nb_channels)])
    
    # On associe à chaque case le nombre de points s'y trouvant
    # (avec le critère actuel, seul la présence d'au moins un point compte)
    for ind, row in don.iterrows():
        for e in range(nb_channels):
            curr_x = row[ncx[e]]
            curr_y = row[ncy[e]]
            i_x = 1
            i_y = 1
            while i_x < prec_X and (gridx[i_x]+0.5*pas_X) < curr_x:
                i_x += 1
            while i_y < prec_Y and (gridy[i_y]+0.5*pas_Y) < curr_y:
                i_y += 1
            grid[e,i_y-1,i_x-1] += 1
    
    # HEATMAP : On n'effectue l'opération que sur la première grille (les autres étant très proches)
    if heatmap:
        #print(grid)
        # Calcul du score de densité pour la grille d'interpolation (>1 pour accepter)
        grid_final = heatmap_grid_calc(grid[0],grid_conv,prec_X,prec_Y,quot)
        
        #print(grid_final)
        # Première heatmap avec le w_exp d'entrée
        heatmap_plot(don,grid_final,grid[0],ncx[0],ncy[0],[min_X,max_X,min_Y,max_Y],\
                     [prec_X,prec_Y],w_exp_)
        correct = False
        while correct == False:
            genut.input_mess(["Grid w_exp testing : ","",\
                              "Input value to test (float)","n : End procedure"])
            inp = input()
            if inp == "n":
                correct = True
            else:
                try:
                    w_exp = float(inp)
                    # Calcul de la fenêtre pour la grille d'interpolation
                    grid_conv, w_exp, w_exp_, quot = calc_coeff(w_exp,radius)
                    
                    # Calcul du score de densité pour la grille d'interpolation (>1 pour accepter)
                    grid_final = heatmap_grid_calc(grid[0],grid_conv,prec_X,prec_Y,quot)
                    
                    # Heatmap successive avec la valeur de w_exp rentré dynamiquement
                    heatmap_plot(don,grid_final,grid[0],ncx[0],ncy[0],\
                                 [min_X,max_X,min_Y,max_Y],[prec_X,prec_Y],w_exp_)
                except:
                    warnings.warn("Invalid answer.")
        
    else:
        # Grille d'interpolation vide
        grid_final = np.array([[[np.nan for i in range(prec_X)] for j in range(prec_Y)]\
                               for e in range(nb_channels)])
        if radius > 0:
            for e in range(nb_channels):
                for j in range(prec_Y):
                    for i in range(prec_X):
                        coeff = 0
                        # Si toute case comprenant au moins un point doit être laissée pleine
                        # peut importe son score de densité
                        if grid[e,j,i] != 0 and only_nan:
                            grid_final[e,j,i] = 0
                        else:
                            # Calcul de densité
                            for gc in grid_conv:
                                n_j = j+gc[0]
                                n_i = i+gc[1]
                                if n_j >= 0 and n_j < prec_Y and n_i >= 0 and n_i < prec_X:
                                    # Si une case adjacente est non vide, elle compte dans le calcul
                                    if grid[e,n_j,n_i] != 0:
                                        coeff += gc[2]
                            # Acceptation de la case (0 = oui, NaN = non)
                            if coeff > quot:
                                grid_final[e,j,i] = 0
    
    return grid_final, [min_X,max_X,min_Y,max_Y], [prec_X,prec_Y]


def calc_coeff(w_exp,radius):
    """
    Create the window for the grid.\n
    Removal procedure uses a circular window of radius ``radius``.
    Each tile is given a proximity coefficient from the center from 0 to 1.
    For each empty tile (containing no points), sum the coefficients of all
    non empty tiles included in the window.\n
    If the output sum surpasses ``quot``, we accept the tile.
    Otherwise, we will set its value to ``NaN``.
    
    Notes
    -----
    The ``mult`` variable is use if ``w_exp < 0`` to continue the acceptance trend,
    while not creating a curve of negative exponent.
    
    Parameters
    ----------
    radius : int
        Detection radius around each tile for ``NaN`` completion.
    w_exp : float
        Exponent of the function used to compute the detection window coefficients.
        If negative, will be set to 0 but widen the acceptance.
    
    Returns
    -------
    grid_conv : list of ``[x,y,coeff]``
        Contains every tile of the window with its x, y and associated coefficient.
    w_exp : float
        Updated exponent. Is equal to ``max(w_exp,0)``.
    w_exp_ : float
        Original value of exponent.
    quot : float
        Quotient used to set the acceptance value of a tile.
    
    See also
    --------
    ``dat_to_grid``
    """
    grid_conv = []
    rc = radius**2
    w_exp_ = w_exp
    # Cas d'un 'w_exp' négatif : on le met à 0 pour ne pas utiliser une puissance négative
    # En revanche on ajoute un multiplicateur pour garder l'évolution 
    if w_exp < 0:
        mult = 1-w_exp
        w_exp = 0
    else:
        mult = 1
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            d = (i**2)+(j**2)
            # On crée une grille circulaire avec des coefficients variant de 0 à 1
            # en fonction de leur distance au centre
            if d <= rc:
                coeff = ((1+radius-np.sqrt(i**2+j**2))/(1+radius))**w_exp * mult
                grid_conv.append([i,j,coeff])
    quot = ((radius)**2)/3
    
    return grid_conv, w_exp, w_exp_, quot


def heatmap_grid_calc(grid,grid_conv,prec_X,prec_Y,quot):
    """
    Associate a (positive) density coefficient to each tile.
    
    Notes
    -----
    Normally, a tile is included in the interpolation if its density coefficient
    is greater than ``quot``.\n
    For clarity, its density value is divided by ``quot``,
    so we can only check tiles acceptance by comparing their value to ``1``.\n
    Thus, il will be easier to judge by looking at the heatmap.
    
    Parameters
    ----------
    grid : np.ndarray (dim 2) of float
        Contains the number of points included in each tile.
        Only the emptiness (``0`` or ``> 0``) is important.
    grid_conv : list of ``[x,y,coeff]``
        Contains every tile of the window with its x, y and associated coefficient.
    prec_X : float
        Size of the grid for x axis.
    prec_Y : float
        Size of the grid for y axis.
    quot : float
        Quotient used to set the acceptance value of a tile.
    
    Returns
    -------
    grid_final : np.ndarray (dim 2) of float
        Contains the grid density values.
    
    See also
    --------
    ``dat_to_grid``
    """
    grid_final = np.array([[0.0 for i in range(prec_X)] for j in range(prec_Y)])
    for j in range(prec_Y):
        for i in range(prec_X):
            coeff = 0
            for gc in grid_conv:
                n_j = j+gc[0]
                n_i = i+gc[1]
                # Si une case adjacente est non vide, elle compte dans le calcul
                if n_j >= 0 and n_j < prec_Y and n_i >= 0 and n_i < prec_X:
                    if grid[n_j,n_i] != 0:
                        coeff += gc[2]
            grid_final[j,i] = coeff/quot
    
    return grid_final


def dat_to_grid_2(don,ncx,ncy,nb_channels,nb_res,radius=0,prec=100,step=None,only_nan=True,verif=False):
    """
    Put raw data on a grid, then determine which tile should be removed (with ``NaN`` value).\n
    This algorithm uses two criteras, the mean distance vector and the biggest empty cone of points.\n
    If the mean vector length is more than the half of radius, or there is 
    no point in a cone of more than 180 degrees, the tile is empty.\n
    Is better to crop borders but can create unwanted results if ``radius`` is smaller than 
    some holes' width.
    
    Notes
    -----
    Complexity :
        .. math:: O(d + p^2r^2) 
        where d is the number of points, p is ``prec`` and r is ``radius``.
        
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    ncx : list of str
        Names of every X columns.
    ncy : list of str
        Names of every Y columns.
    nb_channels : int
        Number of X and Y columns. The number of coils.
    nb_res : int
        The number of data per coil.
    ``[opt]`` radius : int, default : ``0``
        Detection radius around each tile for ``NaN`` completion.
    ``[opt]`` prec : int, default : ``100``
        Grid size of the biggest axis. The other one is deducted by proportionality.
    ``[opt]`` step : ``None`` or float, default : ``None``
        Step between each tile, according to the unit used by the position columns.
        If not ``None``, ignore ``prec`` value.
    ``[opt]`` only_nan : bool, default : ``True``
        If ``True``, tiles that contain at least one point are always kept.
        If ``False``, will remove those that are too eccentric.
    ``[opt]`` verif : bool, default : ``False``
        Print some useful informations for testing.
    
    
    Returns
    -------
    grid_final : np.ndarray (dim 3) of float
        For each data column, contains the grid values (``0`` if tile is taken, ``NaN`` if not)
    ext : [float, float, float, float]
        Extend of the grid. Contains ``[min_X, max_X, min_Y, max_Y]``.
    pxy : [float, float]
        Size of the grid for each axis. Contains ``[prec_X, prec_Y]``.
    
    Raises
    ------
    * Some columns does not exist.
    * Some columns are not numeric.
    
    See also
    --------
    ``CMD_grid``
    """
    print("=== Grid construction phase ===")
    
    # Calcul des dimensions de la grille
    try:
        X = np.array(don[ncx])
        Y = np.array(don[ncy])
    except KeyError:
        raise KeyError('Columns "{}" and "{}" do not exist.'.format(ncx,ncy))

    
    try:
        max_X = max(X.flatten())
        min_X = min(X.flatten())
        max_Y = max(Y.flatten())
        min_Y = min(Y.flatten())
    except TypeError:
        raise TypeError('Columns "{}" and "{}" are not numeric.'.format(ncx,ncy))
    if verif:
        print("max_X = ",max_X)
        print("min_X = ",min_X)
        print("max_Y = ",max_Y)
        print("min_Y = ",min_Y)
    diff_X = max_X-min_X
    diff_Y = max_Y-min_Y
    if step == None:
        if diff_X > diff_Y:
            prec_X = prec
            prec_Y = int(prec*(diff_Y/diff_X))
        else:
            prec_Y = prec
            prec_X = int(prec*(diff_X/diff_Y))
        pas_X = diff_X/prec_X
        pas_Y = diff_Y/prec_Y
    else:
        pas_X = step
        pas_Y = step
        prec_X = int(diff_X/pas_X)+1
        prec_Y = int(diff_Y/pas_Y)+1
    
    # Coordonnées de la grille
    gridx = [min_X + pas_X*i for i in range(prec_X)]
    gridy = [min_Y + pas_Y*j for j in range(prec_Y)]
    
    # Calcul de la fenêtre pour la grille d'interpolation
    grid_conv = calc_coeff(0,radius)[0]
    
    # Grille d'interpolation vide
    grid = [[[[] for i in range(prec_X)] for j in range(prec_Y)] for e in range(nb_channels)]
    
    # On associe à chaque case la liste des points s'y trouvant
    for ind, row in don.iterrows():
        for e in range(nb_channels):
            curr_x = row[ncx[e]]
            curr_y = row[ncy[e]]
            i_x = 1
            i_y = 1
            while i_x < prec_X and (gridx[i_x]+0.5*pas_X) < curr_x:
                i_x += 1
            while i_y < prec_Y and (gridy[i_y]+0.5*pas_Y) < curr_y:
                i_y += 1
            grid[e][i_y-1][i_x-1].append([curr_x,curr_y])
            
    # Grille d'interpolation vide
    grid_p = [[[[] for i in range(prec_X)] for j in range(prec_Y)] for e in range(nb_channels)]
    
    # On associe à chaque case la liste des points dans son voisinage (selon la fenêtre)
    for e in range(nb_channels):
        for j in range(prec_Y):
            for i in range(prec_X):
                for gc in grid_conv:
                    n_j = j+gc[0]
                    n_i = i+gc[1]
                    if n_j >= 0 and n_j < prec_Y and n_i >= 0 and n_i < prec_X:
                        grid_p[e][j][i] += grid[e][n_j][n_i]

    if verif:
        print("Step(2)")
    
    limit = (radius+1)*0.5 * ((pas_X+pas_Y)*0.5)
    # Grille d'interpolation vide
    grid_final = np.array([[[np.nan for i in range(prec_X)] for j in range(prec_Y)] for e in range(nb_channels)])
    for e in range(nb_channels):
        for j in range(prec_Y):
            for i in range(prec_X):
                g = grid_p[e][j][i]
                # Si toute case comprenant au moins un point doit être laissée pleine peut importe son score de densité
                if only_nan and len(grid[e][j][i]) > 0:
                    grid_final[e,j,i] = 0
                else:
                    t = len(g)
                    curr_x = gridx[i]
                    curr_y = gridy[j]
                    if t >= 2:
                        # Somme des vecteurs
                        vect = [0,0]
                        # Liste des angles orientés avec les points, en degrés (-180 à 180)
                        angle = []
                        for gg in g:
                            diff_x = gg[0] - curr_x
                            diff_y = gg[1] - curr_y
                            vect[0] += diff_x
                            vect[1] += diff_y
                            angle.append(np.angle(diff_x + 1j*diff_y, deg=True))
                        # On trie les valeurs pour comparer les angles proches
                        angle = sorted(angle,reverse=False)
                        # Calcul du plus gros "cône" sans points
                        max_angle = 0
                        for a in range(len(angle)-1):
                            max_angle = max(max_angle,angle[a+1]-angle[a])
                        max_angle = max(max_angle,angle[0]-angle[-1]+360)
                        # Acceptation de la case (0 = oui, NaN = non)
                        if np.sqrt(vect[0]**2 + vect[1]**2)/t < limit and max_angle < 180:
                            grid_final[e,j,i] = 0

    return grid_final, [min_X,max_X,min_Y,max_Y], [prec_X,prec_Y]


def scipy_interp(don,ncx,ncy,ext,pxy,nc_data,nb_channels,nb_res,i_method):
    """
    Interpolate data following one given method (``i_method``).\n
    If ``i_method`` starts with ``"RBF_"``, it is part of the radial basis function.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    ncx : list of str
        Names of every X columns.
    ncy : list of str
        Names of every Y columns.
    ext : [float, float, float, float]
        Extend of the grid. Contains ``[min_X, max_X, min_Y, max_Y]``.
    pxy : [float, float]
        Size of the grid for each axis. Contains ``[prec_X, prec_Y]``.
    nc_data : list of str
        Names of every Z columns (actual data).
    nb_channels : int
        Number of X and Y columns. The number of coils.
    nb_res : int
        The number of data per coil.
    i_method : str, {``'nearest'``, ``'linear'``, ``'cubic'``, ``'RBF_linear'``,
    ``'RBF_thin_plate_spline'``, ``'RBF_cubic'``, ``'RBF_quintic'``, ``'RBF_multiquadric'``,
    ``'RBF_inverse_multiquadric'``, ``'RBF_inverse_quadratic'``, ``'RBF_gaussian'``}
        Interpolation method from scipy.
    
    Returns
    -------
    grid_interp : np.ndarray (dim 3) of float
        For each data column, contains the grid interpolation values.
    
    Notes
    -----
    scipy has a automatic built-in method for convex cropping the grid,
    but the process will use the ``dat_to_grid`` crop.\n
    The last 4 methods of RBF are not to be used as of now.
    Otherwise, the ``epsilon`` parameter has to be modified.
    
    Raises
    ------
    * Columns contain NaN (probably not interpolated).
    
    See also
    --------
    ``grid, np.mgrid, scipy.interpolate.griddata, scipy.interpolate.RBFInterpolator``
    """
    print("=== Interpolation phase ===")
    
    # Si on utilise griddata ou RBFInterpolator
    gd = i_method in ['nearest','linear','cubic']
    
    grid_interp = []
    if gd:
        gridx, gridy = np.mgrid[ext[0]:ext[1]:pxy[0]*1j, ext[2]:ext[3]:pxy[1]*1j]
    else:
        gridxy = np.mgrid[ext[0]:ext[1]:pxy[0]*1j, ext[2]:ext[3]:pxy[1]*1j]
    
    for e in range(nb_channels):
        pos_data = don[[ncx[e],ncy[e]]].to_numpy()
        for r in range(nb_res):
            n = e*nb_res + r
            val_data = list(don[nc_data[n]])
            try:
                # griddata : rapide
                if gd:
                    grid_interp.append(griddata(pos_data, val_data, (gridx, gridy), method=i_method))
                # RBFInterpolator : lent (les derniers peuvent faire planter)
                else:
                    flat = gridxy.reshape(2, -1).T
                    grid_flat = RBFInterpolator(pos_data, val_data, kernel=i_method[4:], epsilon=1)(flat)
                    grid_res = np.array([[np.nan for j in range(pxy[1])] for i in range(pxy[0])])
                    for ic,gf in enumerate(grid_flat):
                        i = ic//pxy[1]
                        j = ic%pxy[1]
                        grid_res[i][j] = gf
                    grid_interp.append(grid_res)
            # Ces interpolateurs n'acceptent pas de NaN
            except ValueError:
                raise ValueError('NaNs detected in columns "{}", "{}" and/or "{}"'.format(ncx[e],ncy[e],nc_data[n]))
        
    return grid_interp


def variog(dat,all_models=False,l_d=None,l_e=None,l_t=None,l_c=None):
    """
    Main loop for variogram computation.\n
    **TO DO :** Fuse multiple variograms of different directions into one single model.
    
    Parameters
    ----------
    dat : gstlearn.Db
        Database object of active dataframe.
    ``[opt]`` all_models : bool, default : ``False``
        Add advanced models to selection. Some are expected to crash.
    ``[opt]`` l_d : ``None`` or bool, default : ``None``
        Decision for the number of direction (uni = ``False``, bi = ``True``) for kriging.
        If ``None``, enables a choice procedure. Bidirectional is not implemented.
    ``[opt]`` l_e : ``None`` or ``[[int,int], float, float, int], default : ``None``
        Decisions for the experimental variogram for kriging.
        In order : angle coordinates, angle tolerance, distance, step.
        If ``None``, enables a choice procedure.
    ``[opt]`` l_t : ``None`` or list of int, default : ``None``
        Decisions for the theoretical variogram for kriging.
        Lists the indexes of each chosen models (it is advised to do one loop
        with dynamic choice to get the correspondance).
        If ``None``, enables a choice procedure.
    ``[opt]`` l_c : ``None`` or list of list of [int, int, float], default : ``None``
        Decisions for the theoretical variogram constraints for kriging.
        Lists the indexes of each chosen constraint parameters 
        (it is advised to do one loop with dynamic choice to get the correspondance).
        The first dimension orders by model.
        If ``None``, enables a choice procedure.
    
    Returns
    -------
    fitmod : gstlearn.Model
        Effective variogram model.
    
    Notes
    -----
    Subfunction of ``kriging``.
    
    See also
    --------
    ``kriging, variog_dir_params, variog_fit, gstlearn.plot.varmod``
    """
    # Variogramme expérimental
    vario_list = variog_dir_params(dat,l_d,l_e)
    # print(vario_list)
    # print(vario_list[0])
    
    # Variogramme théorique
    fitmod = variog_fit(vario_list[0],all_models,l_t,l_c)
    
    # Marche pas du coup 
    #for variodir in vario_list[1:]:
        #fitmod_2 = variog_fit(variodir,all_models,l_t,l_c)
        #fitmod_3 = gl.model_rule_combine(fitmod,fitmod_2,gl.Rule(0.0))
        #print(fitmod_3)
    plt.close()
    
    # gstlearn fait des figures toutes petites, on corrige ça !
    for variodir in vario_list:
        gp.varmod(vario=variodir, flagLegend=True).figure.set_size_inches(CONFIG.fig_width, CONFIG.fig_height)
    # gp.varmod(model=fitmod, flagLegend=True).figure.set_size_inches(CONFIG.fig_width, CONFIG.fig_height)
    # gp.decoration(title="Modèle final !")
    # plt.show(block=False)
    # plt.pause(CONFIG.fig_render_time) # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser pour accélérer la vitesse de l'input
    
    return fitmod

# Fait le choix des paramètres du variogramme expérimental

def variog_dir_params(dat,l_d=None,l_e=None):
    """
    Main loop for experimental variogram.\n
    Asks for the number of desired directions, then verify for one if they are correct by user input.
    **TO DO** Currently, the second direction is idle (see ``variog``)
    Compute the experimental variogram from parameters set by user at execution time.\n
    
    Parameters
    ----------
    dat : gstlearn.Db
        Database object of active dataframe.
    ``[opt]`` l_d : ``None`` or bool, default : ``None``
        Decision for the number of direction (uni = ``False``, bi = ``True``) for kriging.
        If ``None``, enables a choice procedure.
        Bidirectional is not implemented.
    ``[opt]`` l_e : ``None`` or ``[[int,int], float, float, int], default : ``None``
        Decisions for the experimental variogram for kriging.
        In order : angle coordinates, angle tolerance, distance, step.
        If ``None``, enables a choice procedure.
    
    Returns
    -------
    vario_list : list of gstlearn.VarioParam
        List of all experimental variograms (for each direction).
    
    Notes
    -----
    Subfunction of ``variog``.
    
    Raises
    ------
    * ``l_d`` is not ``None`` nor a boolean.
    
    See also
    --------
    ``variog, variog_dir_params_choice, gstlearn.Vario.compute``
    """
    # Paramètres de variogrammes vierges
    varioParamMulti = gl.VarioParam()
    varioParamMulti2 = gl.VarioParam()
    choice = ""
    vario_list = []
    
    big_correct = False
    while big_correct == False:
        correct = False
        while correct == False:
            # Si on veut deux directions (marche pas) ou non
            if l_d != None:
                try:
                    inp = ["n","y"][l_d]
                except:
                    raise ValueError("'l_d' = {} : Must be boolean.".format(l_d))
            else:
                genut.input_mess(["Bidirectional variogram ?","",\
                                  "y : bidirectional (not implemented yet)","n : unidirectional"])
                inp = input()
            choice = inp
            if inp == "y":
                # Pour créer les paramètres des variogrammes expérimentaux
                varioParamMulti = variog_dir_params_choice(varioParamMulti,n=1,l_e=l_e)
                varioParamMulti2 = variog_dir_params_choice(varioParamMulti2,n=2,l_e=l_e)
                correct = True
            elif inp == "n":
                varioParamMulti = variog_dir_params_choice(varioParamMulti,n=1,l_e=l_e)
                correct = True
            else:
                warnings.warn("Invalid answer.")
        
        # Calcul du variogramme expérimental
        variodir = gl.Vario(varioParamMulti)
        variodir.compute(dat)
        plt.close()
        
        # Alors lui décide de faire chier des fois va savoir, mais sinon c'est juste du plot des courbes
        if l_d == None or l_e == None:
            try:
                gp.varmod(variodir, flagLegend=True).figure.set_size_inches(CONFIG.fig_width, CONFIG.fig_height)
            except ValueError: # Peut survenir de temps en temps, est réglé en mettant à jour...
                raise OSError("GSTLEARN ERROR: please retry. If this issue persist, try to update packages.")
        vario_list.append(variodir)
        if choice == "y":
            variodir2 = gl.Vario(varioParamMulti2)
            variodir2.compute(dat)
            gp.varmod(variodir2, flagLegend=True)
            vario_list.append(variodir2)
        
        # Si le modèle est correct
        if l_d == None or l_e == None:
            #fig.varmod(variodir, flagLegend=True)
            plt.show(block=False)
            # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser
            # pour accélérer la vitesse de l'input
            plt.pause(CONFIG.fig_render_time)
            print(variodir)
            correct = False
            while correct == False:
                genut.input_mess(["Is the variogram correct ?","","y : Yes","n : No"])
                inp = input()
                if inp == "y":
                    correct = True
                    big_correct = True
                elif inp == "n":
                    correct = True
                    vario_list = []
                else:
                    warnings.warn("Invalid answer.")
        # Si les choix sont faits via paramètre on ne demande pas
        else:
            correct = True
            big_correct = True
    
    return vario_list

# Fait le choix des paramètres du variogramme expérimental sur une direction
# Informations sur https://soft.minesparis.psl.eu/gstlearn/1.7.2/doxygen/classDirParam.html

def variog_dir_params_choice(varioParamMulti,n=1,l_e=None):
    """
    Compute the experimental variogram from parameters set by user at execution time.\n
    
    Parameters
    ----------
    varioParamMulti : list of gstlearn.VarioParam
        Empty variogram.
    ``[opt]`` n : int, default : ``1``
        Index of the direction in the full procedure. Only useful in prints.
    ``[opt]`` l_e : ``None`` or ``[[int,int], float, float, int], default : ``None``
        Decisions for the experimental variogram for kriging.
        In order : angle coordinates, angle tolerance, distance, step.
        If ``None``, enables a choice procedure.
    
    Returns
    -------
    varioParamMulti : list of gstlearn.VarioParam
        Experimental variogram with selected direction.
    
    Notes
    -----
    Subfunction of ``variog_dir_params``.
    
    See also
    --------
    ``variog_dir_params, gstlearn.DirParam``\n
    https://soft.minesparis.psl.eu/gstlearn/1.7.2/doxygen/classDirParam.html
    """
    angle = []
    angle_tol = 0
    dist = 0
    pas = 0
    
    # On demande à l'utilisateur de rentrer la valeur des quatres paramètres du dessus
    if l_e != None:
        try:
            angle = [int(l_e[0][0]),int(l_e[0][1])]
            if angle == [0,0]:
                raise ValueError("'l_e' = {} : Invalid angle.".format(l_e))
            angle_tol = float(l_e[1])
            dist = float(l_e[2])
            pas = int(l_e[3])
        except:
            raise ValueError("'l_e' = {} : Invalid values.".format(l_e))
    else:
        correct = False
        while correct == False:
            genut.input_mess(["Direction {} : ANGLE DIRECTION. Need two integer coordinates".format(n),
                            "Example : '1 1' for 45°, '1 0' for 0°, '-1 2' for 120°",
                            "The angle is oriented from X axis."])
            inp_a = input()
            try:
                res = re.split(r"[ ]+",inp_a)
                angle = [int(res[0]),int(res[1])]
                if angle == [0,0]:
                    warnings.warn("Invalid angle.")
                    continue
                correct = True
            except:
                warnings.warn("Invalid answer.")
        correct = False
        while correct == False:
            genut.input_mess(["Direction {} : ANGLE TOLERANCE. Answer in degrees (0-180)".format(n),
                              "Example : '45' for 45°, '3.14' for 3.14°"])
            inp_t = input()
            try:
                angle_tol = float(inp_t)
                correct = True
            except:
                warnings.warn("Invalid answer.")
        correct = False
        while correct == False:
            genut.input_mess(["Direction {} : COMPUTED RANGE. Distance considered around \
                              each point (float)".format(n),"Example : '100' for a radius of 100"])
            inp_d = input()
            try:
                dist = float(inp_d)
                correct = True
            except:
                warnings.warn("Invalid answer.")
        correct = False
        while correct == False:
            genut.input_mess(["Direction {} : STEP. Number of step".format(n),
                              "Example : '20' for 20 slices"])
            inp_p = input()
            try:
                pas = int(inp_p)
                correct = True
            except:
                warnings.warn("Invalid answer.")
    
    # C'est la seule manière que j'ai trouvé pour correctement définir le nombre d'intervalles de mesure
    breaks_list = [dist/pas*i for i in range(pas+1)]
    
    # Création de la direction
    mydir = gl.DirParam(pas,dist,0.5,angle_tol,0,0,np.nan,np.nan,0,breaks_list,angle)
    varioParamMulti.addDir(mydir)
    
    return varioParamMulti

# Calcule le modèle pour la direction choisie

def variog_fit(variodir,all_models=False,l_t=None,l_c=None):
    """
    Compute the experimental variogram from parameters set by user at execution time.\n
    Organize the selection of the variogram model types.
    
    Parameters
    ----------
    dat : gstlearn.Db
        Database object of active dataframe.
    ``[opt]`` all_models : bool, default : ``False``
        Add advanced models to selection. Some are expected to crash.
    ``[opt]`` l_t : ``None`` or list of int, default : ``None``
        Decisions for the theoretical variogram for kriging.
        Lists the indexes of each chosen models (it is advised to do one loop
        with dynamic choice to get the correspondance).
        If ``None``, enables a choice procedure.
    ``[opt]`` l_c : ``None`` or list of list of [int, int, float], default : ``None``
        Decisions for the theoretical variogram constraints for kriging.
        Lists the indexes of each chosen constraint parameters 
        (it is advised to do one loop with dynamic choice to get the correspondance).
        The first dimension orders by model.
        If ``None``, enables a choice procedure.
    
    Returns
    -------
    fitmod : gstlearn.Model
        Effective variogram model.
    
    Notes
    -----
    Subfunction of ``variog``.
    Constants starts with a ``_`` and their order correspond to the 
    built-in index of each gstlearn component. It should not be modified.
    
    See also
    --------
    ``variog, gstlearn.Constraints.addItemFromParamId, gstlearn.ECov.fromValue,
    gstlearn.EConsElem.fromValue, gstlearn.EConsType.fromValue``
    """
    # Modèles classiques (marchent bien)
    _Types_print = ["0 : NUGGET ", "1 : EXPONENTIAL ", "2 : SPHERICAL ", "3 : GAUSSIAN ",
                    "4 : CUBIC ", "5 : SINCARD (Sine Cardinal) ", "6 : BESSELJ ",
                    "7 : MATERN ", "8 : GAMMA ", "9 : CAUCHY ", "10 : STABLE ",
                    "11 : LINEAR ", "12 : POWER "]
    print_l = 30
    if all_models:
        # Modèles moins classiques (marchent pas tous)
        _Types_print += ["13 : ORDER1_GC (First Order Generalized covariance) ",
                         "14 : SPLINE_GC (Spline Generalized covariance) ", 
                         "15 : ORDER3_GC (Third Order Generalized covariance) ",
                         "16 : ORDER5_GC (Fifth Order Generalized covariance) ", 
                         "17 : COSINUS ", "18 : TRIANGLE ", "19 : COSEXP (Cosine Exponential) ",
                         "20 : REG1D (1-D Regular) ", "21 : PENTA (Pentamodel) ", 
                         "22 : SPLINE2_GC (Order-2 Spline) ", 
                         "23 : STORKEY (Storkey covariance in 1-D) ", 
                         "24 : WENDLAND0 (Wendland covariance (2,0)) ", 
                         "25 : WENDLAND1 (Wendland covariance (3,1)) ", 
                         "26 : WENDLAND2 (Wendland covariance (4,2)) ", 
                         "27 : MARKOV (Markovian covariances) ", "28 : GEOMETRIC (Sphere only) ", 
                         "29 : POISSON (Sphere only) ", "30 : LINEARSPH (Sphere only) ",]
        print_l = 54
    
    # Technique d'artiste ma che bellissima
    _Symbol_print = [" ",".","-","~"]
    _Color_print = [CONFIG.error_color, CONFIG.success_color]
    
    nb_models = len(_Types_print)
    type_choice=[False for i in range(nb_models)]
    types_list = []
    constr_list = []
    
    # Cas de sélection dynamique pour autoriser l'indexation)
    if l_c == None:
        l_c = [None]
    
    # Choix du modèle de variogramme
    cpt = -1 # Uniquement pour 'l_t'
    big_correct = False
    while big_correct == False:
        types = []
        # Type de modèle
        if l_t != None:
            cpt += 1
            try:
                inp = l_t[cpt]
            except IndexError:
                inp = "y"
        else:
            genut.input_mess(["Variogram choice : enter model index to add/remove it\
                              from selection",""]
                           +[p+(print_l-len(p))*_Symbol_print[ic%4]+\
                             _Color_print[int(type_choice[ic])]+\
                             str(type_choice[ic])+\
                             CONFIG.code_color for ic,p in enumerate(_Types_print)]
                           +["y : End"])
            inp = input()
        # Si on a fini d'ajouter des modèles
        if inp == "y":
            constraints = gl.Constraints()
            if not types_list:
                if l_t == None:
                    warnings.warn("Variogram can't be empty.")
                else:
                    raise ValueError("Variogram can't be empty.")
                continue
            curr_id = -1
            act_id = -1
            for c in constr_list:
                if c[0] != curr_id:
                    curr_id = c[0]
                    act_id += 1
                #print(act_id)
                constraints.addItemFromParamId(gl.EConsElem.fromValue(c[1]+1),icov=act_id,\
                                               type=gl.EConsType.fromValue(c[2]-1),value=c[3])
            #print(types_list)
            #print(constr_list)
            for t in types_list:
                types.append(gl.ECov.fromValue(t))
            plt.cla()
            fitmod = gl.Model()
            fitmod.fit(variodir,types=types, constraints=constraints)
            # Si le modèle est correct
            if l_t == None and l_c[0] == None:
                gp.varmod(variodir, fitmod, flagLegend=True)\
                    .figure.set_size_inches(CONFIG.fig_width, CONFIG.fig_height)
                gp.decoration(title="Model VS Experimental")
                plt.show(block=False)
                # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser
                # pour accélérer la vitesse de l'input
                plt.pause(CONFIG.fig_render_time)
                print(constraints)
                print(fitmod)
                correct = False
                while correct == False:
                    genut.input_mess(["Le modèle de variogramme semble-t-il correct ?",
                                      "","y : Oui","n : Non"])
                    inp = input()
                    # Fin de la boucle
                    if inp == "y":
                        #print(constraints)
                        correct = True
                        big_correct = True
                    # On continue le processus
                    elif inp == "n":
                        correct = True
                    else:
                        warnings.warn("Invalid answer.")
            # Si les choix sont faits via paramètre on ne demande pas
            else:
                correct = True
                big_correct = True
        else:
            try:
                inp_id = int(inp)
                type_choice[inp_id] = not type_choice[inp_id]
                
                # Ajout d'un modèle
                if type_choice[inp_id]:
                    types_list.append(inp_id)
                    # Sélection des contraintes
                    constr_list += variog_constraints(inp_id,l_c[cpt])
                # Retrait d'un modèle (et de ses contraintes)
                else:
                    types_list.remove(inp_id)
                    new_l = []
                    for e in constr_list:
                        print(e," ",inp_id)
                        if e[0] != inp_id:
                            new_l.append(e)
                    constr_list = new_l
                #print(constr_list)
            except:
                if l_t == None:
                    warnings.warn("Invalid answer.")
                else:
                    raise ValueError("'l_t' = {} : Invalid answer.".format(l_t[cpt]))
    
    if l_t == None or l_c[0] == None:
        print(fitmod)
    return fitmod
    
# Fait le choix des contraintes sur les modèles

def variog_constraints(var_id,l_c=None):
    """
    Organize the selection of the variogram model constraints.\n
    ``_ConsElem_exist`` contains whether the constraint exists for the current model,
    ordered as in the ``_ConsElem_print``.
    Each subset of the 2D list correspond to the respective model in the
    ``_Types_print`` constant of ``variog_fit``.
    
    Parameters
    ----------
    var_id : int
        Index of the current model.
    ``[opt]`` l_c : ``None`` or list of [int, int, float], default : ``None``
        Decisions for the theoretical variogram constraints for kriging.
        Lists the indexes of each chosen constraint parameters 
        (it is advised to do one loop with dynamic choice to get the correspondance).
        If ``None``, enables a choice procedure.
    
    Returns
    -------
    constr_list : [int, int, int , float]
        Contains the indexes of each constraint elements and the associated value.
    
    Notes
    -----
    Subfunction of ``variog_fit``\n
    Constants starts with a ``_`` and their order correspond to the built-in index
    of each gstlearn component. It should not be modified.\n
    Some 'advanced' models are apparently not fonctionnal and kill the kernel.
    In particular, if a model is marked as accepting all constraints,
    it means that it crashes the kernel (in all known cases).
    
    See also
    --------
    ``variog_fit``
    """
    # Pour chaque  modèle, sélectionne les contraintes existantes (1 = oui, 0 = non)
    # Si ya que des 1 c'est que le modèle plante
    _ConsElem_exist = [[0,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],
                       [1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],[1,0,1,1,0,0,0,0,0],[1,0,1,1,0,0,0,0,0],
                       [1,0,1,1,0,0,0,0,0],[1,0,1,1,0,0,0,0,0],[1,0,1,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],
                       [0,0,1,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],
                       [0,0,0,1,0,0,0,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,0,0,1,0,0,0,0,0],
                       [1,1,1,1,1,1,1,1,1],[1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],[1,1,1,1,1,1,1,1,1],
                       [1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],
                       [1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0,0]]
    # Contraintes possibles de gstlearn (la moitié n'existent jamais masi au cas où...)
    _ConsElem_print = ["1 : RANGE", "2 : ANGLE (Anisotropy rotation angle (degree))", 
                       "3 : PARAM (Auxiliary parameter)", "4 : SILL", "5 : SCALE", 
                       "6 : T_RANGE (Tapering range)", "7 : VELOCITY (advection)", 
                       "8 : SPHEROT (Rotation angle for Sphere)", "9 : TENSOR (Anisotropy Matrix term)"]
    ConsElem_curr = _ConsElem_exist[var_id]
    dispo_list = ["Constraint choice (variable)","","0 : None (end)"]
    for ic,c in enumerate(ConsElem_curr):
        if c:
            dispo_list.append(_ConsElem_print[ic])
    constr_list = []
    
    # Choix de la contrainte
    if l_c != None:
        cpt = 0 # Uniquement pour 'l_c'
        while cpt < len(l_c):
            try:
                # Paramètre de contrainte
                inp1 = int(l_c[cpt][0])
                # Type de contrainte
                inp2 = int(l_c[cpt][1])
                # Valeur de la contrainte
                inp3 = float(l_c[cpt][2])
                constr_list.append([var_id,inp1-1,inp2,inp3])
                cpt += 1
            except:
                raise ValueError("'l_c' = {} : Some values are invalid".format(l_c[cpt]))
    else:
        correct = False
        while correct == False:
            genut.input_mess(dispo_list)
            try:
                # Paramètre de contrainte
                inp1 = int(input())
                if inp1 == 0:
                    correct = True
                elif not ConsElem_curr[inp1-1]:
                    warnings.warn("Selected constraint does not exist.")
                else:
                    genut.input_mess(["Constraint choice (type)","","0 : LOWER (Lower Bound)", 
                                      "1 : DEFAULT (Default parameter)", "2 : UPPER (Upper Bound)", 
                                      "3 : EQUAL (Equality)"])
                    # Type de contrainte
                    inp2 = int(input())
                    if inp2 < 0 or inp2 > 3:
                        warnings.warn("Unknown type.")
                    else:
                        genut.input_mess(["Enter constraint value (float)"])
                        # Valeur de la contrainte
                        inp3 = float(input())
                        constr_list.append([var_id,inp1-1,inp2,inp3])
            except:
                warnings.warn("Invalid answer.")
    
    return constr_list


def coeffs_relation(X,Y,m_type="linear",choice=False,conv=True,nb_conv=50,plot=False):
    """
    Given two arrays ``X`` and ``Y``, compute the coefficients of the chosen regression.\n
    To be used in the context of finding a formula for a physical relation.
    
    Parameters
    ----------
    X : np.array of float
        X axis of relation.
    Y : np.array of float
        Y axis of relation.
    ``[opt]`` m_type : str, {``"linear"``, ``"poly_3"``, ``"inverse_3"``}, default : ``"linear"``
        Type of wanted relation.\n
        * ``"linear"`` is a simple linear regression.
            .. math::
                a + bx
        * ``"poly_3"`` is a polynomial regression of degree 3.
            .. math::
                a + bx + cx^{2} + dx^{3}
        * ``"inverse_3"`` is a symetrical relation of ``"poly_3"``.
            .. math::
                a + bx + cx^{\\frac{1}{2}} + dx^{\\frac{1}{3}}
    ``[opt]`` choice : bool, default : ``False``
        Allows the user to chose which regression estimator fits the best 
        (between numpy's `polyfit` as 'linear' \
        and sklearn's `TheilSenRegressor` and `HuberRegressor`). 
        If ``False``, choose `HuberRegressor`. Ignored if ``m_type = linear``.
    ``[opt]`` conv : bool, default : ``True``
        If ``m_type = inverse_3``, uses an iterative method to estimate the coefficients. 
        Otherwise, uses a simple linear system.\n
        The iterative method usually gives better results but is a bit slower.
    ``[opt]`` nb_conv : int, default : ``50``
        If ``conv = True``, represent the number of points used for the iterative method. 
        Taking more points is slower but more precise.
    ``[opt]`` plot : bool, default : ``False``
        Plots all steps.
    
    Returns
    -------
    model : list of float
        List of found coefficients for the chosen method (2 for linear, else 4).
    
    Notes
    -----
    [DEV] The ``conv`` parameter may be removed if one of the two procedure 
    is deemed better in all cases.
    
    Raises
    ------
    * Unknown regression type.
    
    See also
    --------
    ``poly_regr, convergence_inv_poly``
    """
    model_list = ["linear","poly_3","inverse_3"]
    if m_type not in model_list:
        raise ValueError("Unknown model {} ({})".format(m_type,model_list))
    
    # Plot du nuage de points initial
    if plot or (choice and m_type != "linear"):
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(CONFIG.fig_width,CONFIG.fig_height))
        ax.plot(X,Y,'+',label="Raw points")
        ax.set_xlabel(r"signal(ph)")
        ax.set_ylabel(r"$\sigma$")
    
    # Cas d'une relation linéaire (facile !)
    if m_type == "linear":
        l_r = linregress(X,Y)
        if plot:
            ax.plot(X,l_r.intercept+X*l_r.slope,'o',ms=1,label="Linear regression")
            ax.set_title("Linear model VS Data")
            plt.legend()
        #print([l_r.intercept, l_r.slope])
        return [l_r.intercept, l_r.slope]
    # Sinon ...
    else:
        # Si on veut choisir entre différents estimateurs
        if choice:
            l = ["linear","theilsen","huber"]
        # Sinon, Huber est souvent le meilleur
        else:
            l = ["huber"]
        # Cas de degré 3 classique
        if m_type == "poly_3":
            # Fonction de calcul
            p_r = poly_regr(X,Y,choice)
            p_r_list = []
            # Représentation pour le plot
            for i,c in enumerate(p_r):
                p_r_list.append(c[0]+c[1]*X+c[2]*X**2+c[3]*X**3)
                if plot or choice:
                    ax.plot(X,p_r_list[i],"o",ms=1,label=l[i])
        # Cas inverse (a+bx+cx^(-1/2)+dx^(-1/3))
        else:
            # Calcul de polynome 3 en inversant X et Y
            p_r = poly_regr(Y,X,choice)
            p_r_list = []
            # Représentation pour le plot
            for i,c in enumerate(p_r):
                p_r_list.append(c[0]+c[1]*Y+c[2]*Y**2+c[3]*Y**3)
                if plot or choice:
                    ax.plot(p_r_list[i],Y,"o",ms=1,label=l[i])
        if plot or choice:
            ax.set_title("Estimator VS Data")
            plt.legend()
            plt.show(block=False)
            # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
            # pour accélérer la vitesse de l'input
            plt.pause(CONFIG.fig_render_time)
        # Choix dynamique de l'interpolateur
        if choice:
            correct = False
            while correct == False:
                genut.input_mess(["Choose the estimator : ","","0 : linear",
                                  "1 : theilsen","2 : huber"])
                inp = input()
                try:
                    inp = int(inp)
                    if m_type == "poly_3":
                        model = p_r[inp]
                    else:
                        model = p_r_list[inp]
                    correct = True
                except ValueError:
                    warnings.warn("Invalid answer.")
                except IndexError:
                    warnings.warn("Invalid estimator ({}).".format(inp))
            plt.close(fig)
        # Sortie des coeffs pour le poly_3
        else:
            if m_type == "poly_3":
                model = p_r[0]
            else:
                model = p_r[0][0]+p_r[0][1]*Y+p_r[0][2]*Y**2+p_r[0][3]*Y**3
        
        if m_type == "poly_3":
            return list(model)
        
        # Estimation de l'équation inverse
        nb_pts = len(Y)
        # Par convergence (algo maison)
        if conv:
            mc = len(Y)/(nb_conv**2)
            npc_l = np.array([int(mc*i**2) for i in range(nb_conv)])
            X_c = model[npc_l]
            Y_c = Y[npc_l]
            fc = convergence_inv_poly(Y_c,X_c,nb_conv) # Inversion X et Y
        # Par résolution d'un système linéaire (facile mais moons efficace)
        else:
            npc_l = np.array([0,nb_pts//3,2*nb_pts//3,nb_pts-1])
            X_c = model[npc_l]
            X_c_l = np.array([[1,xi,xi**(1/2),xi**(1/3)] for xi in X_c])
            Y_c = Y[npc_l]
            fc = np.linalg.solve(X_c_l, Y_c)
        
        # Plot avec modèle polynomial trasposé VS modèle inverse final
        if plot:
            X_plot = np.linspace(min(model),max(model),100)
            fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(CONFIG.fig_width,CONFIG.fig_height))
            ax.plot(model,Y,"o",ms=7,label="Polynomial estimation")
            ax.plot(X_plot,fc[0]+fc[1]*X_plot+fc[2]*X_plot**(1/2)+fc[3]*X_plot**(1/3),
                    "-",label="Inverse model")
            ax.set_title("Final model VS Polynomial estimator")
            ax.set_xlabel(r"signal(ph)")
            ax.set_ylabel(r"$\sigma$")
            plt.legend()
            plt.show(block=False)
            # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
            # pour accélérer la vitesse de l'input
            plt.pause(CONFIG.fig_render_time)
        return fc

def poly_regr(X,Y,choice=False):
    """
    Given two arrays ``X`` and ``Y``, compute the coefficients of the polynomial regression.\n
    To be used in the context of finding a formula for a physical relation.
    
    Parameters
    ----------
    X : np.array of float
        X axis of relation.
    Y : np.array of float
        Y axis of relation.
    ``[opt]`` choice : bool, default : ``False``
        Allows the user to chose which regression estimator fits the best (between numpy's `polyfit`\
        and sklearn's `TheilSenRegressor` and `HuberRegressor`).
        If ``False``, choose `HuberRegressor`
    
    Returns
    -------
    coefs_list : list of float
        List of found coefficients for the chosen method (2 for linear, else 4).
    
    Notes
    -----
    Subfunction of ``coeffs_relation``.\n
    sklearn's estimators are glitchy and returns half the value of degree 0 coefficient.
    Hence it is manually doubled.
    
    See also
    --------
    ``coeffs_relation, sklearn.preprocessing.PolynomialFeatures, sklearn.pipeline.make_pipeline``
    """
    x = X.copy()
    y = Y.copy()

    coefs_list = []
    xd = x[:,np.newaxis]
    # Si on fait linear+ThielSen+Huber ou que Huber
    if choice:
        mymodel = np.poly1d(np.polyfit(x, y, 3))
        
        coefs = mymodel.c[::-1]
        coefs_list.append(coefs)
        
        r = np.random.randint(10000)
        estimator = [TheilSenRegressor(random_state=r),HuberRegressor(),]
    else:
        estimator = [HuberRegressor()]
    
    # Pour les estimateurs sklearn
    for i,e in enumerate(estimator):
        poly = PolynomialFeatures(3)
        model = make_pipeline(poly, e)
        model.fit(xd, y)
        coefs = e.coef_
        coefs[0] *= 2 # Corrige un bug sur le coefficient de degré 0 (je n'ai pas d'explication)
        coefs_list.append(coefs)
            
    return coefs_list

def mse_inv_poly(X,Y,c):
    """
    Compute the mean square error between the ``Y`` of the polynomial model (target) 
    and the ``new_Y`` of the current inverse model.
    Is the main convergence critera for the iterative method.
    
    Parameters
    ----------
    X : np.array of float
        X axis of relation.
    Y : np.array of float
        Y axis of relation.
    c : [float, float, float, float]
        List of coefficients of the current inverse model.
    
    Returns
    -------
    mse : float
        Mean square error between ``Y`` and ``new_Y``.
    
    Notes
    -----
    Subfunction of ``convergence_inv_poly`` and ``convergence_inv_step``.\n
    Must not be called if some values are negative (will create NaNs).
    """
    new_Y = c[0] + c[1]*X + c[2]*X**(1/2) + c[3]*X**(1/3)
    mse = sum((new_Y - Y)**2)
    return mse
    

def convergence_inv_poly(X,Y,nb_pts,nb_tours=1000,force_fin=25,verif=False):
    """
    Given two arrays ``X`` and ``Y``, converges to a transposed polynomial formula with 
    an inverse polynomial formula (see ``coeffs_relation``, ``"inverse_3"``).
    
    Parameters
    ----------
    X : np.array of float
        X axis of relation.
    Y : np.array of float
        Y axis of relation.
    nb_pts : int
        Number of points used for the iterative method (length of ``X`` and ``Y``). 
        Taking more points is slower but more precise.
    ``[opt]`` nb_tours : int, default : ``1000``
        Number of loops for each iterative cycle. At the end, check if mse is lower than
        the ``fin_mse`` w_exp. Otherwise, redo a cycle.
    ``[opt]`` force_fin : int, default : ``25``
        Number of maximum cycles. Upon reach, return the final result regardless of its relevance.
    ``[opt]`` verif : bool, default : ``False``
        Prints some relevant informations each cycle for tesing purposes.
    
    Returns
    -------
    best_cl : list of float
        List of best found coefficients.
    
    Notes
    -----
    Subfunction of ``coeffs_relation``.\n
    Description of the method :\n
    The algorithm is a probabilistic iterative method minimizing the mean square error between
    the target ``Y`` and the current model (``mse``).
    For each step, one of the four coefficients is chosen randomly. 
    It is set to a value which is in a local minimum for mse.
    If the current model is better, we keep it. 
    Else, its is kept with a certain probability (``convergence_inv_step``).
    The overall best model is saved as ``best_cl``.
    The maximum number of steps is ``nb_tours * force_fin``.
    
    See also
    --------
    ``coeffs_relation, convergence_inv_step, mse_inv_poly``
    """
    # Valeurs de départ des coeffs pour gagner du temps
    coef_list = [float(min(X)), float((max(Y)-min(Y))/(max(X)-min(X))), 0, 0]
    # Premier coeff à faire converger, peut être n'importe lequel
    current_coef = 3
    # Tentative de faire un cas d'arrêt dynamique
    diff_y = max(Y)-min(Y)
    fin = False
    fin_mse = diff_y/5
    if verif:
        print("fin : ",fin_mse)
    # Mis à infini pour toujours être moins bien que le premier calculé
    best_mse = np.inf
    # Stocke la meilleure configuration trouvée
    best_cl = coef_list.copy()
    # Nombre de tours
    cpt = 0
    # Tant que c'est pas fini c'est pas fini wallah
    while not fin:
        for i in range(nb_tours):
            mse = mse_inv_poly(X,Y,coef_list)
            r = np.random.randint(4)
            # Étape de convergence du coefficient
            coef_list, mse = convergence_inv_step(X,Y,coef_list,mse,current_coef)
            current_coef = r
            # Si la configuration est la meilleur, on la garde
            if best_mse > mse:
                best_mse = mse
                best_cl = coef_list.copy()
            cpt += 1
        if verif:
            print(best_mse)
        # Condition de fin
        if best_mse < fin_mse or cpt >= nb_tours*force_fin:
            fin = True
    if verif:
        print(cpt," ",fin_mse," ",best_mse)
    return best_cl
    
def convergence_inv_step(X,Y,coef_list,mse,cc):
    """
    Perform one step of the iterative method.
    Converges to the best mean square error by incrementing the chosen parameter 
    (of index ``cc``) by a fixed value until mse increases.\n
    Then, go the other way with a step twice as small, until the step is small enough.
    
    Parameters
    ----------
    X : np.array of float
        X axis of relation.
    Y : np.array of float
        Y axis of relation.
    coefs_list : list of float
        List of current coefficients.
    mse : float
        Mean square error with the initial configuration.
    cc : int, {``0``, ``1``, ``2``, ``3``}
        Index of the coefficient to iterate on.
    
    Returns
    -------
    best_cl : list of float
        List of best found coefficients.
    
    Notes
    -----
    Subfunction of ``convergence_inv_poly``.\n

    See also
    --------
    ``convergence_inv_poly, mse``
    """
    cl_cpy = coef_list.copy()
    # Sens de parcourt (croissant ou décroissant)
    sign = True
    # La fin ;)
    fin = False
    # Le score de l'itération précédente
    prev_mse = mse
    # Pas pour chaque tour
    step = 1
    while not fin:
        mse = mse_inv_poly(X,Y,cl_cpy)
        # Si on s'éloigne...
        if mse > prev_mse:
            # ... on change de sens et on divise le pas par deux
            sign = not sign
            step /= 2
            # Algo probabiliste pour ne pas stagner dans un minimum local (pas sûr de son efficacité)
            r = np.random.random()
            if r > np.exp(-(mse/prev_mse)*3):
                cl_cpy[cc] = coef_list[cc]
            else:
                prev_mse = mse
                coef_list[cc] = cl_cpy[cc]
        else:
            prev_mse = mse
            coef_list[cc] = cl_cpy[cc]
        # Nouvelle valeur après le pas
        cl_cpy[cc] += (int(not sign) - int(sign))*step
        # Au bout de suffisament d'aller-retour
        if step < 0.01:
            fin = True
    return coef_list, prev_mse