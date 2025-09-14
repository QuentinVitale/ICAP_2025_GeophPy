# -*- coding: utf-8 -*-
"""
   geophpy.core.utils
   ------------------

   Provides low-level, pure utility functions used by various modules
   throughout the package.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

from __future__ import unicode_literals
from .datastructures import PointData
from scipy import interpolate

import os
import glob
import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.stats import sem
from scipy.signal import correlate

from scipy.ndimage._ni_support import _normalize_sequence
import re
import datetime

import geophpy.__config__ as CONFIG

# --- Trajectory funcitons ---
def distance_along_track(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the cumulative distance along a series of 2D points.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the track points.
    y : np.ndarray
        The y-coordinates of the track points.

    Returns
    -------
    np.ndarray
        A 1D array of the same length as x and y, where each element is the
        cumulative distance from the first point to the current point.
    """

    if x.size != y.size:
        raise ValueError("Input x and y arrays must have the same size.")
    if x.size == 0:
        return np.array([])
        
    # Calculate the distance between each consecutive point
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
    # Calculate the cumulative sum and prepend a zero for the starting point
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    
    return cumulative_distances

def resample_profile(distance_array: np.ndarray, value_array: np.ndarray, step: float) -> np.ndarray:
    """
    Resamples a profile onto a regular distance interval.

    This function uses linear interpolation to create a new profile with
    values sampled at regular steps (e.g., every 0.5 meters).

    Parameters
    ----------
    distance_array : np.ndarray
        The 1D array of cumulative distances along the track.
    value_array : np.ndarray
        The 1D array of measurement values for the track.
    step : float
        The desired resampling distance step.

    Returns
    -------
    np.ndarray
        The new, resampled array of measurement values.
    """
    if distance_array.size == 0:
        return np.array([])

    # Create the new, regular distance axis for resampling
    max_distance = distance_array[-1]
    new_distance_axis = np.arange(0, max_distance, step)
    
    # Use NumPy's linear interpolation for efficient resampling
    resampled_values = np.interp(new_distance_axis, distance_array, value_array)
    
    return resampled_values

def shift_points_along_trajectory(x_coords: np.ndarray, y_coords: np.ndarray, shift_distance: float) -> (np.ndarray, np.ndarray):
    """
    Shifts a set of 2D points along their own trajectory.

    This function calculates the new (x, y) coordinates for a set of
    points after "sliding" them along the path they define by a specified
    distance. It uses linear interpolation to find the new positions.

    Parameters
    ----------
    x_coords : np.ndarray
        The original x-coordinates of the track points.
    y_coords : np.ndarray
        The original y-coordinates of the track points.
    shift_distance : float
        The physical distance to shift the points. Can be positive
        (shifts forward along the track) or negative (shifts backward).

    Returns
    -------
    x_new : np.ndarray
        The new, shifted x-coordinates.
    y_new : np.ndarray
        The new, shifted y-coordinates.
    """

    if x_coords.size < 2:
        # Cannot determine a trajectory with fewer than two points
        return x_coords, y_coords
        
    # 1. Calculate the cumulative distance along the original path
    original_distances = distance_along_track(x_coords, y_coords)

    # 2. Define the new target distances for each point
    new_distances = original_distances + shift_distance
    
    # 3. Use 1D linear interpolation to find the new coordinates
    #    interpolate.interp1D will handle linear extrapolation 
    #    for points outside the original ends.
    fx = interpolate.interp1d(original_distances, x_coords, kind='linear', fill_value="extrapolate")
    fy = interpolate.interp1d(original_distances, y_coords, kind='linear', fill_value="extrapolate")

    # 4. Use the new functions to find the new coordinates
    x_new = fx(new_distances)
    y_new = fy(new_distances)
    
    return x_new, y_new

# --- Correlation functions ---
def calculate_correlmap(image: np.ndarray, percentile_range: tuple = None) -> np.ndarray:
    """
    Calculates the profile-to-profile cross-correlation map for a grid.

    This low-level utility function computes the Pearson correlation
    coefficient between each adjacent profile (column) in the input image.

    Parameters
    ----------
    image : np.ndarray
        The 2D gridded data (z_image) to be analyzed.
    percentile_range : tuple of (int, int), optional
        If provided, the correlation is calculated on a robust subset of the
        data, excluding outliers based on this percentile range.
        If None, all data is used. Defaults to None.

    Returns
    -------
    correl_map : np.ndarray
        A 2D array where each column represents the correlation function
        between that profile and the one to its left.
    """

    if image.ndim != 2:
        raise ValueError("Input must be a 2D grid.")

    ny, nx = image.shape
    max_corr_len = ny * 2 - 1  # The maximum possible length for the correlation result
    # correl_map = np.zeros((ny * 2 - 1, nx))
    correl_map = np.full((max_corr_len, nx), np.nan) # Initialize with NaN


    data_for_corr = image
    if percentile_range:
        vmin, vmax = np.nanpercentile(image, percentile_range)
        data_for_corr = np.where((image >= vmin) & (image <= vmax), image, np.nan)

    # Correlate each profile with the one to its left
    for i in range(1, nx):
        p1 = data_for_corr[:, i - 1]
        p2 = data_for_corr[:, i]

        valid_mask = ~np.isnan(p1) & ~np.isnan(p2)
        if np.sum(valid_mask) < 2:
            continue

        # Using numpy's correlate for the cross-correlation function
        correlation = np.correlate(p1[valid_mask], p2[valid_mask], mode='full')
        # Normalize to be a true correlation coefficient
        correlation /= (np.std(p1[valid_mask]) * np.std(p2[valid_mask]) * len(p1[valid_mask]))

        # Center the correlation result in an array of the maximum size
        start_pad = (max_corr_len - len(correlation)) // 2
        end_pad = max_corr_len - len(correlation) - start_pad

        padded_correlation = np.pad(correlation, (start_pad, end_pad), 'constant', constant_values=np.nan)

        # Store the result in the map
        correl_map[:, i] = padded_correlation
        
    return correl_map

def calculate_correlation_sum(image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculates the sum of cross-correlations between all adjacent profiles.

    Parameters
    ----------
    image : np.ndarray
        The 2D gridded data (z_image) to be analyzed.

    Returns
    -------
    lags : np.ndarray
        The array of possible shifts (lags).
    correlation_sum : np.ndarray
        The summed correlation values for each lag.
    """

    if image.ndim != 2:
        raise ValueError("Input must be a 2D grid.")

    ny, nx = image.shape
    
    # 1. Compute the full correlation map.
    correl_map = calculate_correlmap(image)
    
    # 2. Sum the correlations across all profiles (columns).
    #    np.nansum treats the NaN values from empty profiles as zero.
    correlation_sum = np.nansum(correl_map, axis=1)

    # 3. Create the corresponding lags array.
    lags = np.arange(-ny + 1, ny)
    
    return lags, correlation_sum

def profile_get_shift(p1: np.ndarray, p2: np.ndarray) -> (int, float):
    """
    Calculates the vertical shift between two profiles using cross-correlation.

    Parameters
    ----------
    p1 : np.ndarray
        The first 1D profile (reference).
    p2 : np.ndarray
        The second 1D profile to be shifted.

    Returns
    -------
    shift : int
        The optimal integer shift (lag) to align p2 with p1.
    correlation_value : float
        The value of the Pearson correlation coefficient at the optimal lag.
    """

    valid_mask = ~np.isnan(p1) & ~np.isnan(p2)
    if not np.any(valid_mask):
        return 0, 0

    p1_valid = p1[valid_mask]
    p2_valid = p2[valid_mask]

    if p1_valid.size < 3: # Not enough points for meaningful correlation
        return 0, 0

    # # Cross-correlation to find the best alignment
    # correlation = correlate(p1_valid - np.mean(p1_valid), p2_valid - np.mean(p2_valid), mode='full')
    # lag_index = np.argmax(correlation)
    # lag = lag_index - (len(p1_valid) - 1)
    
    # # Calculate the Pearson correlation coefficient at the found lag
    # shifted_p2 = np.roll(p2_valid, -lag)
    # corr_coeff, _ = np.corrcoef(p1_valid, shifted_p2)
    
    # return int(lag), corr_coeff[0, 1]
    
    # Cross-correlation to find the best alignment
    correlation = correlate(p1_valid - np.mean(p1_valid), p2_valid - np.mean(p2_valid), mode='full')
    lag = np.argmax(correlation) - (len(p1_valid) - 1)
    
    # Calculate the Pearson correlation coefficient and check for flatness
    shifted_p2 = np.roll(p2_valid, -lag)
    # Check if either profile has zero variance
    if np.std(p1_valid) == 0 or np.std(shifted_p2) == 0:
        # If so, the correlation is undefined, return 0
        corr_value = 0.0
    else:
        corr_matrix = np.corrcoef(p1_valid, shifted_p2)
        # Ensure the matrix is 2D before indexing
        if corr_matrix.ndim == 2:
            corr_value = corr_matrix[0, 1]
        else:
            corr_value = 0.0 # Another safeguard

    return int(lag), corr_value

# --- Filtering functions ---
def hampel_filter_1d(data_array: np.ndarray, window_half_width: int, threshold: float = 3.0) -> np.ndarray:
    """
    Applies a 1D Hampel filter to an array to identify and replace outliers.

    The Hampel filter is a robust outlier detection algorithm that slides a
    window along the data. For each window, it calculates the median and the
    Median Absolute Deviation (MAD). If the central point of the window
    deviates from the median by more than a given threshold of MADs, it is
    replaced by the median.

    Parameters
    ----------
    data_array : np.ndarray
        Input 1D array to be filtered.
    window_half_width : int
        The half-width of the moving window (K). The full window size is 2*K+1.
    threshold : float, optional
        The standard deviation threshold (t) in terms of MADs. A common
        value is 3.0. Defaults to 3.0.

    Returns
    -------
    np.ndarray
        The filtered array with outliers replaced by their local window median.
    """
    arr = np.asarray(data_array)
    filtered_arr = arr.copy()
    n = arr.size
    
    for i in range(n):
        # Define the window boundaries, naturally handling edges
        start = max(0, i - window_half_width)
        end = min(n, i + window_half_width + 1)
        window = arr[start:end]
        
        median = np.nanmedian(window)
        # Calculate the Median Absolute Deviation (MAD) scale estimate
        mad = 1.4826 * np.nanmedian(np.abs(window - median))
        
        # If the point is an outlier, replace it with the median
        if mad > 0 and np.abs(arr[i] - median) > threshold * mad:
            filtered_arr[i] = median
            
    return filtered_arr

def median_decision_filter_1d(data_array: np.ndarray, window_half_width: int, threshold: float, mode: str = 'relative') -> np.ndarray:
    """
    Applies a 1D decision-theoretic median filter.

    This filter is a variation of the standard median filter. It replaces
    the central point of a sliding window with the window's median only if
    the point's deviation from the median exceeds a specified threshold.

    Parameters
    ----------
    data_array : np.ndarray
        Input 1D array to be filtered.
    window_half_width : int
        The half-width of the moving window (K).
    threshold : float
        The deviation threshold (t).
    mode : {'relative', 'absolute'}, optional
        Defines how the threshold is interpreted.
        - 'relative': The threshold is a fraction of the local median
          (e.g., 0.1 for 10%).
        - 'absolute': The threshold is an absolute value.
        Defaults to 'relative'.

    Returns
    -------
    np.ndarray
        The filtered array.
    """
    arr = np.asarray(data_array)
    filtered_arr = arr.copy()
    n = arr.size
    
    for i in range(n):
        # Define the window boundaries
        start = max(0, i - window_half_width)
        end = min(n, i + window_half_width + 1)
        window = arr[start:end]
        
        center_val = arr[i]
        median = np.nanmedian(window)
        deviation = np.abs(center_val - median)

        # Determine the condition based on the mode
        if mode == 'relative':
            condition = np.abs(threshold * median)
        else: # absolute
            condition = threshold

        # Replace the value only if it exceeds the condition
        if deviation > condition:
            filtered_arr[i] = median

    return filtered_arr


# ...TBD... following function seems not to be used !!!
def array1D_getdeltamedian(array):
    '''
    To get the median of the deltas from a 1D-array

    Parameters:

    :array: 1D-array to treat

    Returns:

    :deltamedian:
    '''
    #deltamedian = None
    #deltalist = []
    #for i in range(0,len(array)-1):
    #   if (array[i] != np.nan):
    #      prev = array[i]
    #      break
    #for val in array[i+1:]:
    #   if ((val != np.nan) and (val != prev)):
    #      deltalist.append(val - prev)
    #      prev = val
    #deltamedian = np.median(np.array(deltalist))
    #return deltamedian
    return np.median(np.diff(array))


# ...TBD... following function seems not to be used !!!
def array1D_extractdistelementslist(array):
    '''
    To extract distinct elements of a 1D-array

    Parameters:

    :array: 1D-array to treat

    Returns:

    :distlist: list of distinct values from the array
    '''
    #distlist = []
    #for val in array:
    #   if (val != np.nan):
    #      found = False
    #      for dist in distlist:
    #         if (val == dist):
    #            found = True
    #            break
    #      if (found == False):
    #         distlist.append(val)
    #return np.array(distlist)
    return np.unique(array)


def profile_completewithnan(x, y_array, nan_array, ydeltamin, factor=10, ymin=None, ymax=None):
    ''' Completes profile x with 'nan' values
    Parameters :

    :x: x value
    :y_array: 1D array to test gaps, [1,2,4,2,6,7,3,8,11,9,15,12,...]
    :nan_array: 2D array to complete with 'nan' values, [[x1, y1, 'nan'], [x2, y2, 'nan'], ...]
    :ydeltamin: delta min to test before two consecutives points, to complete with 'nan' values
    :factor: factor to take in account to test gap
    :ymin: min y position in the profile.
    :ymax: max y position in the profile.

    Returns:
        nan_array completed
    '''

    if (ymin == None):
        yprev = y_array[0]
        indexfirst = 1
    else:
        yprev = ymin
        indexfirst = 0

    for y in y_array[indexfirst:]:
        ydelta = y - yprev
        if (ydelta > (factor*2*ydeltamin)):
            # complete with 'nan' values
            for i in range(1,int(np.around(ydelta/ydeltamin))):
                nan_array.append([x, yprev+i*ydeltamin/2, np.nan])
                nan_array.append([x, y-i*ydeltamin/2, np.nan])
        yprev = y

    if (ymax != None):
        # treats the last potential gap
        ydelta = ymax - yprev
        if (ydelta > (factor*2*ydeltamin)):
            # complete with 'nan' values
            for i in range(1,int(np.around(ydelta/ydeltamin))):
                nan_array.append([x, yprev+i*ydeltamin/2, np.nan])
                nan_array.append([x, ymax-i*ydeltamin/2, np.nan])

    return nan_array


# ...TBD... following function seems not to be used !!!
def array2D_extractyprofile(x, x_array, y_array):
    '''
    To extract the y profile at x coordinate

    Parameters:

    :x: x value at which to extract the y profile

    :x_array: 1D-array containing x values associated to each y value

    :y_array: 1D-array containing y profiles

    Note: x_array and y_array must have the same dimension, but do not need to be sorted

    Returns:

    :profile: unique y values encountered at x coordinate, in ascending order
    '''
    #profile = []
    #for i in range(0, len(x_array)-1):
    #   if (x_array[i] == x):
    #      profile.append(y_array[i])
    #return np.array(profile)
    return np.unique(y_array[np.where(x_array == x)])


def make_sequence(value):
    ''' If input is scalar, make it iterable to be used in for loops.

    Inspired by :meth:`_normalize_sequence`` from :mod:`scipy.ndimage._ni_support`

    '''

    if not(hasattr(value, "__iter__") and not isinstance(value, str)):
        return [value]
    else:
        return value


def make_normalize_sequence(value, rank):
    ''' wrapper for :meth:`scipy.ndimage._ni_support._normalize_sequence`.

    If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.

    '''

    return _normalize_sequence(value, rank)


##def arraygetprecison(array):
##    '''
##    To get the (maximum) number of decimals from an array.
##    '''
##    
##    precision = []
##    for value in np.ravel(array):
##        precision.append(getdecimalsnb(value))
##
##    return max(precision)

def get_median_step(coords: np.ndarray, prec: int = 2) -> float:
    """
    Calculates the median step between distinct values in a coordinate array.

    Parameters
    ----------
    coords : np.ndarray
        A 1D array of coordinates (e.g., x or y values).
    prec : int, optional
        The decimal precision for the returned value. Defaults to 2.

    Returns
    -------
    float
        The calculated median step size.
    """
    if coords.size < 2:
        return 0.0
    unique_coords = np.unique(coords)
    if unique_coords.size < 2:
        return 0.0
    median_step = np.median(np.diff(unique_coords))
    return round(median_step, prec)

def get_decimals_nb(value: float) -> int:
    """
    Counts the number of decimal places in a float.

    Parameters
    ----------
    value : float
        The number to inspect.

    Returns
    -------
    int
        The number of decimal places.
    """
    return len(str(value).split('.')[-1])

# def getdecimalsnb(value):
#     ''' Return the number of decimals from a float value/array.  '''

#     # Decimal number for a single value
#     def get_value_decimalnb(value):
#         decimalsnb = 0
#         test = abs(value)
#         while ((test - int(test)) > 0.):
#             decimalsnb += 1
#             test *= 10
#         return decimalsnb

#     # Applying for each input element
#     value = make_sequence(value)
#     decimalsnb = []
#     for element in value:
#         decimalsnb.append(get_value_decimalnb(element))

#     # single value input
#     if len(decimalsnb)==1:  
#         decimalsnb = decimalsnb[0]

#     return decimalsnb

##def getdecimalsnb(value):
##    '''
##    To get the number of decimals from a float value
##
##    Parameters:
##
##    :value: float number to treat
##
##    Returns:
##
##    :decimalsnb: decimal precision of the value
##    '''
##    decimalsnb = 0
##### use abs(value) in order to :
#####- avoid referencing value
#####- get the correct answer when value is negative
##    test = abs(value)
##    while ((test - int(test)) > 0.):
##        decimalsnb += 1
##        test *= 10
##    return decimalsnb

def unique_multiplets(mylist):
    ''' Find the unique multiplets of a list.

    Returns the sorted unique multiplets of a list of multiplets.

    from https://stackoverflow.com/questions/48300501/how-to-remove-duplicate-tuples-from-a-list-in-python/48300601#48300601
    '''
    seen = set()
    unique = []
    
    for lst in mylist:

        # convert to hashable type
        current = tuple(lst)

        # If element not in seen, add it to both
        if current not in seen:
            unique.append(lst)
            seen.add(current)

    return unique

def find_multiplets(mylist):
    ''' Find the multiplets in the list of tuple. 
    
    

    Returns the sorted unique multiplets of a list of multiplets.
    
    '''
    
    # Find an element in list of tuples.
    Output = [item for item in mylist
          if item[0] == 3 or item[1] == 3] 
    return

def arrayshift(arr, shift, val=None):
    '''
    Roll array element.
    
    Elements that roll beyond the last position are replaced by val
    or re-introduced as the first element (if val=None).
    '''

    # No shift
    if shift==0:
        arrshift = arr
        return arrshift

    # Allocating empy (shifted) array
    arrshift = np.empty_like(arr)

    # Circular shift
    if val is None:
        arrshift = np.roll(arr, shift, axis=None)

    # Shifting & Padding with val
    if shift >= 0:
        arrshift[:shift] = val
        arrshift[shift:] = arr[:-shift]
    else:
        arrshift[shift:] = val
        arrshift[:shift] = arr[-shift:]

    return arrshift


def array_to_level(array, nblvl=256, valmin=None, valmax=None):
    '''
    Convert an array of values to brigthess level (grayscale).
    
    Parameters:

    :array: Values to be converted to brigthess level.

    :nblvl: Number of level.

    :valmin: Minimum value to consider for level definition, if None the values minimum is used.
    
    :valmax: Maximum value to consider for level definition, if None the values mmaximum is used.

    Returns:
    
    :lvl: brigthess level image (from 0to nblvl-1).
    '''
    # No min or max values provided
    if valmin==None:
        valmin = np.nanmin(array)
    if valmax==None:
        valmax = np.nanmax(array)

   # Scaling values from 0 to nblvl-1 brightness level
    step = (valmax-valmin) / (nblvl)    # conv. factor from lvl to val
    lvl = np.around((array-valmin)/step)  # brightness level
    
    # Insuring no overlimit brightness values 
    lvl[np.where(lvl<0)] = 0
    lvl[np.where(lvl>nblvl-1)] = nblvl-1

    return lvl


def level_to_array(lvl, valmin, valmax, nblvl=256):
    '''
    Convert brigthess level (grayscale) back to values.

    Parameters:

    :lvl: brigthess level image (from 0 to nblvl-1)

    :valmin: minimum value to consider for value recovering
    
    :valmax: maximum value to consider for value recovering

    :nblvl: number of level to consider for value recovering

    Returns:
    
    :array: level converted back values.
    '''
    
    # Scaling values from valmin to valmax
    step = (valmax-valmin) / (nblvl)  # conv. factor from lvl to val
    array = lvl*step + valmin  # from brightness level to values

    # Insuring no overlimit values
    array[np.where(array<valmin)] = valmin
    array[np.where(array>valmax)] = valmax

    return array


def arraygetstats(array):
    '''
    Computes the basic statistics of an array.

    Parameters:

    :array:

    Returns:

    :mean: array arithmetic mean.

    :std: array standard deviation

    :median: array median (Q2, 2nd quartile, 50th percentile).

    :Q1, Q3: array 1st and 3rd quartiles (25th and 75th percentiles).
    
    :IQR: array interquartile range.
    '''

    mean = np.nanmean(array)
    std = np.nanstd(array)
    median = np.nanmedian(array)
    Q1, Q3 = np.nanpercentile(array,[25,75])
    IQR =  Q3 - Q1

    return mean, std, median, Q1, Q3, IQR


def arraygetmidXpercentinterval(array, percent=0.80):
    '''
    get the mi X perncent interval
    '''
    
    lb = (1-percent)/2  # Lower bound in percentage
    ub = percent + lb  # Upper bound in percentage

    return np.nanpercentile(array,[lb,ub])


def arraysetmedian(array, val=0, method='additive'):
    '''
    Set an array median to a given value.

    Parameters:

    :array: array of values.

    :val: value to set the array median to.
    
    :method: method used to set the median ('additive' or 'multiplicative').
    '''

    arraymedian = np.nanmedian(array)

    # Using additive offset
    if method.lower() == 'additive':
        offset = val - arraymedian
        return array + offset

    # Using multiplicative offset (scaling)
    elif method.lower() == 'multiplicative':
        offset = val / arraymedian
        return array*offset


def arraysetmean(array, val=0, method='additive'):
    '''
    Set an array mean to a given value.

    Parameters:

    :array: array of values.

    :val: value to set the array mean to.
    
    :method: method used to set the mean ('additive' or 'multiplicative').
    '''
    arraymean = np.nanmean(array)

    # Using additive offset
    if method.lower() == 'additive':
        offset = val - arraymean
        return array + offset

    # Using multiplicative offset (scaling)
    elif method.lower() == 'multiplicative':
        offset = val / arraymean
        return array*offset

def array1D_getoverlap(arr1, arr2, tol=0.1):
    '''
    Return the overlapping elements of two arrays.

    a list of x and y coordinates and an additional z value.

    x, y and z must be on separate lineswise
    array([x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...]])

    Parameters:

    :array1, array2: 2D-arrays containing x and y coordinates with
        an additional z value:
        array([x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...]])

    :tol: tolerance (same unit as x,y) at which two points are considered
        at the same location.

    Returns:

    :arr: Overlapping coordinates and corresponding distance:
            [[x,y,val]_1, idx_1,[x,y,val]_2, idx_2, dist_1-2]
            where idx_ is the index of the overlapping value
            in the original array.
    '''

    # Pairwise distance matrix (cdist takes column array)
    arr1 = arr1.T
    arr2 = arr2.T
    xy1 = arr1[:,0:2]
    xy2 = arr2[:,0:2]
    dist = cdist(xy1,xy2)  # dist[ij] = dist(arr1[i], arr2[j])
        
    # Overlapping array
    idx = np.where(dist <= tol)
    arr = np.column_stack((arr1[idx[0]],idx[0],
                           arr2[idx[1]], idx[1],
                           np.reshape(dist[idx],(-1,1))))

    return arr

def arraymismatch(arr1, arr2, weighted=True, discardspurious=True):
    '''
    Return the mean (weighted) mismatch between arrays of the same dimensions.
    '''

    # Initial mismatch #########################################################
    # Mismatch going from arr1 to arr2
    dk = arr1-arr2
    dy = np.nanmean(dk)  
    M = dk.size
    wy = 1  # default weighting factor

    # Array 1 an 2 are equals
    if all(val==0 for val in dk):
        return 0

    # Weighting factor (Haigh 1992)
    if weighted:
        ## In Haigh (1992) : "the weighting factor is effectively
        ## the inverse square of the standard error". It is defined as
        ## wy = M**2 / np.sum((dy - dk)**2) which truly is M * (1/stde)**2.
        ## [is it a mistake in the article ?].
        ## The standard error is used here.
        #wy = M**2 / np.sum((dy - dk)**2)  # weighting factor (Haigh 1992)
        std = np.nanstd(dk)
        wy = (1/std)**2  # M / np.sum((dy - dk)**2) ?

    # Discarding spurious data #################################################
    ## data out of the range [dy - 2.5*std, dy + 2.5*std]
    ## are discarded (Haigh 1992)
    if discardspurious:
        # Non spurious data
        valmin = dy - 2.5*std
        valmax = dy + 2.5*std
        idx = (dk >= valmin) & (dk <= valmax)
        dk = dk[idx]

        # New mismatch value
        dy = np.nanmean(dk)  
        M = dk.size
        if weighted:
            std = np.nanstd(dk)
            wy = (1/std)**2

    return wy*dy


#------------------------------------------------------------------------------#
# Spatial transformations                                                      #
#------------------------------------------------------------------------------#
def array1D_centroid(array):
    '''
    Returns the centroid of an array containing
    x, y and z coordinates.

    x, y and z must be on separate lineswise
    array([x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...]])

    Examples:
    --------
    >>> x = np.random.rand(1,10)*100
    >>> y = np.random.rand(1,10)*100
    >>> z = np.random.rand(1,10)*100
    >>> xyz = np.array(x, y, z)
    >>> center = arraycentroid(xyz)
    '''

    return np.nanmean(array, axis=1)


def array1D_translate(array, vect):
    '''
    Translation of an array containing x, y and eventually z coordinates. 

    Array must be ?line-wise? [[x1, x2, x3, ..., xn],
                                [y1, y2, y3, ..., xn],
                                [z1, z2, z3, ..., zn]]

    Parameters:

    :array:

    :vect: vector containing the shift for each dimension.

    Returns translated array.     
    '''

    xyz_flag = array.shape[0] == 3 # True if array contains x, y and z

    # Checking z-dimension shift
    shiftx = vect[0]
    shifty = vect[1]
    if vect.size==3:
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
    M[:,-1] = np.stack((shiftx, shifty, shiftz, 1))

    # Translating data.values
    xyz = M.dot(xyz)

    # Getting rid of homogeneous coordinates
    xyz = np.delete(xyz, -1, 0)
    if not xyz_flag:
        xyz = np.delete(xyz, -1, 0)  # 'false' z=0

    return xyz


def array1D_rotate(array, angle=90, center=None):
    '''
    Clockwise rotation about the z-axis of an array containing
    x, y and eventually z coordinates.

    Array must be ?line-wise? [[x1, x2, x3, ..., xn],
                                [y1, y2, y3, ..., xn],
                                [z1, z2, z3, ..., zn]]

    Parameters:

    :array:

    :angle:

    :center:

    Returns a rotated array trough an angle 'angle' about the point 'center'.
    '''
    angle = np.mod(angle,360)  # positive angle (-90->270)
    xyz_flag = array.shape[0] == 3 # True if array contains x, y and z

    # Center of rotation #######################################################
    # array centroid 
    if center is None:
        if xyz_flag:
            center = array1D_centroid(array[:-1,:])
        else:
            center = array1D_centroid(array[:,:])
        
    # Bottom Left as center of rotation 
    elif center.upper() in ['BL']:
        center = np.append(np.nanmin(array[0,:]), np.nanmin(array[1,:]))
        
    # Bottom Right as center of rotation 
    elif center.upper() in ['BR']:
        center = np.append(np.nanmax(array[0,:]), np.nanmin(array[1,:]))
        
    # Top Left as center of rotation 
    elif center.upper() in ['TL']:
        center = np.append(np.nanmin(array[0,:]), np.nanmax(array[1,:]))
        
    # Top Right as center of rotation 
    elif center.upper() in ['TR']:
        center = np.append(np.nanmmax(array[0,:]), np.nanmax(array[1,:]))

    # Given center vector of coordinates
    else:
        pass

    # Homogeneous coordinates matrix ###########################################
    # Rotation center homogeneous coordinates
    if center.size == 2:
        center = np.append(center,0) # adding false 0 z value
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
    
    # Array rotation ###########################################################
    # Moving the center of rotation at the array centroid
    Mc = np.eye(4)
    Mc[:,-1] = np.append(-center, 1)
    xyz = Mc.dot(xyz)

    # Rotating array
    xyz = M.dot(xyz)

    # Translation to origin back to data centroid
    Mc[:,-1] = np.append(center, 1)
    xyz = Mc.dot(xyz)

    # Getting rid of homogeneous coordinates
    xyz = np.delete(xyz, -1, 0)
    if not xyz_flag:
        xyz = np.delete(xyz, -1, 0)  # 'false' z=0

    return xyz


####
# Not used yet
####
def sliding_window_1D(array, n):
    '''
    Creates a sliding windows of size n for 1D-array.
    
    This function uses stride_tricks to creates a sliding windows
    without using explicit loops. Each of the windows is return in an
    extra dimension of the array.

    n = 3

    0 |    0      0     0     0     0     0     0
    1 |    1 |    1     1     1     1     1     1
    2 |    2 |    2|    2     2     2     2     2
    3      3 |    3|    3|    3     3     3     3
    4      4      4|    4|    4|    4     4     4
    5      5      5     5|    5|    5|    5     5
    6      6      6     6     6|    6|    6|    6
    7      7      7     7     7     7|    7|    7|
    8      8      8     8     8     8     8|    8|
    9      9      9     9     9     9     9     9|
    -->
    
        - - - - - - - - > arr.size - n + 1
      | 0 1 2 3 4 5 6 7
    n | 1 2 3 4 5 6 7 8
      | 2 3 4 5 6 7 8 9
      v   
    '''

    shape = array.shape[:-1] + (array.shape[-1] - n + 1, n)  #
    strides = array.strides + (array.strides[-1],)           #

    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
####
####


def isidentical(iterator):
    '''
    Check if all element of an iterable are identical
    '''
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True

    return all(first == rest for rest in iterator)




#------------------------------------------------------------------------------#
# Basic curve fitting                                                          #
#------------------------------------------------------------------------------#
def gauss_func(x, a, x0, sigma):
    r''' Gaussian function (or bell curve).

    .. math::

       f(x) = a e^{ -\frac{1}{2}\left(\frac{x-x_0}{\sigma}\right)^2 }

    Parameters
    ----------
    x : array_like
        Abscisse at which compute the fucntion.

    a : scalar
        Peak amplitude of the curve.

    x0 : scalar
        Peak position (mean of a normal ditribution).

    sigma : scalar
        Controls the curve with (standard deviation of a normal ditribution). 

    Returns
    -------

    gaussian curve

    '''

    return a*np.exp(-(x-x0)**2/(2*sigma**2)) # a*np.exp(-0.5*( (x-x0)/sigma )**2)


def gauss_fit(xdata, ydata):
    ''' Non-linear least squares fit of a gausian function to the data.

    This is a convenience wrapper of scipy.optimize.curve_fit for a gaussian function.

    Parameters
    ----------
    xdata : array_like
        The independent variable where the data is measured.
        Must be an M-length sequence or an (k,M)-shaped array for functions with k predictors.

    ydata : array_like
        The dependent data, a length M array - nominally f(xdata, ...).

    Returns
    -------
    popt : array
        Optimal values for the parameters of the gaussian fuction (a, x0 and sigma) so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.

    pcov2d : array
        The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).

    '''

    popt, pcov = curve_fit(gauss_func, xdata, ydata)
    #yfit = gauss(xdata, popt[0], popt[1], popt[2])

    return popt, pcov




#------------------------------------------------------------------------------#
# User input display                                                           #
#------------------------------------------------------------------------------#
def input_mess(mess_list):
    """
    Print message in a specific 'user input' format.
    Is meant to be used before the ``input()`` function and if the ``GUI`` global variable is set to ``False``.

    Parameters
    ----------
    mess_list : list of str
        Message to display, row per row.
    """ 
    try:
        nc = os.get_terminal_size().columns -1
    except OSError:
        nc = 49
    print(CONFIG.code_color+CONFIG.blink_color)
    print("+---"*(nc//4)+"+")
    print(CONFIG.code_color)
    for mess in mess_list:
        print(mess)
    print(CONFIG.code_color+CONFIG.blink_color)
    print("+---"*(nc//4)+"+")
    print(CONFIG.base_color)




#------------------------------------------------------------------------------#
# String management                                                            #
#------------------------------------------------------------------------------#
def str_clean(dirty_str,l=False,path=False):
    """ 
    Remove unwanted characters from string.
    Is used on function call arguments.
    
    Notes
    -----
    Filters ``'[', ']', ' ', '"'`` and ``'''``.
    
    Parameters
    ----------
    dirty_str : str
        String to filter
    ``[opt]`` l : bool, default : ``False``
        If we keep brackets (mainly for column names)
    ``[opt]`` path : bool, default : ``False``
        If we keep spaces (mainly for file paths)
    
    Returns
    -------
    dirty_str : str
        Output str after cleaning process.
    
    See Also
    --------
    ``optargs_list, split_str_list``
    """ 
    # Retrait des crochets (à eviter pour certains labels)
    if l == False:
        dirty_str = dirty_str.replace('[','')
        dirty_str = dirty_str.replace(']','')
    # Retrait de l'espace (à éviter pour les chemins)
    if path == False:
        dirty_str = dirty_str.replace(' ','')
    dirty_str = dirty_str.replace('"','')
    dirty_str = dirty_str.replace("'",'')
    return dirty_str


def split_str_list(list_string, list_type, path=False, noclean=False):
    """ 
    Split a list string to list of elements from a specified type.
    
    Parameters
    ----------
    list_string : str
        String to split.
    list_type : data-type
        Type of output list.
    ``[opt]`` path : bool, default : ``False``
        If we keep spaces.
    ``[opt]`` noclean : bool, default : ``False``
        If we do not filter the splitted strings.
    
    Returns
    -------
    l : list of type ``list_type``
        Output list.
    
    See Also
    --------
    ``optargs_list, str_clean, str_to_bool``
    """ 
    l = []
    #print(list_string)
    if list_type in [int, float]:
        occurs = re.compile(r"[0-9.]+").findall(list_string)
        for oc in occurs:
            l.append(list_type(oc))
    if list_type in [bool]:
        occurs = list_string.split(',')
        for oc in occurs:
            l.append(str_to_bool(oc))
    if list_type in [str]:
        if noclean:
            occurs = list_string.split(',')
        else:
            occurs = str_clean(list_string,path=path).split(',')
        l = occurs
    #print(occurs)
    # La sortie doit toujours être une liste
    if isinstance(l,list):
        return l
    return [l]


def optargs_list(list_args, list_args_name, list_args_type):
    """ 
    Associate each optional argument with the correct value (if specified).
    
    Parameters
    ----------
    list_args : list of str
        List of all optional arguments specified by user.
    list_args_name : list of str
        List of all existing optional arguments names of the selected function.
    list_args_type : list of data-type
        List of all existing optional arguments types of the selected function.
    
    Returns
    -------
    l : dict of [``arg_name : arg_value``]
        Dictionary of every specified argument and their value.
    
    Notes
    -----
    All arguments should be written as such : ``[arg_name]=[arg_value]``.\n
    Variable names in '``occurs[0] in [values]``' line are hardcoded path variable names.
    They are processed differently from the others in order to keep their path structure.
    If a new path variable name is to be added, it should be indicated in theses statements.\n
    ``GraphicUI``, ``GraphicUIn't`` and ``GraphicUI_ignore`` are special keywords related to ``GraphicInterface.py``.
    
    Raises
    ------
    * Parameter does not exist.
    * Parameter does not have any value.
    * Parameter is of a wrong type.
    
    See Also
    --------
    ``split_str_list, str_clean, str_to_bool``
    """
    dict_args = {}
    for c_arg in list_args:
        c_arg = str_clean(c_arg,l=True,path=True)
        occurs = re.split(r"[ ]*=[ ]*",c_arg)
        #print(occurs)
        try:
            # Vérification que le paramètre optionel existe
            try:
                ic = list_args_name.index(occurs[0])
            except ValueError:
                raise LookupError("Optional parameter '{}' does not exist ({}).".format(occurs[0],list_args_name))
            # Si le type est une liste
            if isinstance(list_args_type[ic],list):
                # Noms spécifiques de chemins de fichiers : on ne supprime pas les espaces
                if occurs[0] in ["file_list","file_list_rev","cfg_file_list"]:
                    path = True
                # Listes classiques
                else:
                    path = False
                dict_args[list_args_name[ic]] = split_str_list(occurs[1], list_args_type[ic][0], path=path)
            # Conversion de la chaîne en booléen
            elif list_args_type[ic] == bool:
                dict_args[list_args_name[ic]] = str_to_bool(occurs[1])
            # Noms spécifiques de chemins de fichiers : on ne supprime pas les espaces
            elif occurs[0] in ["file","output_file","output_file_base"]:
                dict_args[list_args_name[ic]] = str_clean(occurs[1],path=True)
            else:
                dict_args[list_args_name[ic]] = list_args_type[ic](occurs[1])
        except ValueError or TypeError:
            raise TypeError("Optional parameter '{}' if not of type {} ({}).".format(occurs[0],occurs[1],list_args_type[ic]))
        except IndexError:
            raise ValueError("Optional parameter '{}' is not associated with a value.".format(occurs[0]))
    #print(dict_args)
    return dict_args


def str_to_bool(bool_str):
    """ 
    Convert 'bool' str to bool.
    
    Parameters
    ----------
    bool_str : str
        String to convert.
    
    Returns
    -------
    bool
        ``True`` if ``bool_str = "True", "true", "T", "t", "1"``.\n
        ``False`` if ``bool_str = "False", "false", "F", "f", "0"``.
    
    Notes
    -----
    Any other value returns 'False' but raises a warning.
    
    See Also
    --------
    ``str_clean``
    """ 
    if str_clean(bool_str).lower() in ["true","t","1"]:
        return True
    elif str_clean(bool_str).lower() not in ["false","f","0"]:
        warnings.warn('Value "{}" not recognised, will be considered as False.'.format(bool_str))
    return False


def true_file_list(file_list=None):
    """ 
    Return the file list path of 'file_list'.
    If no path is specified (``None``), return the list of every .dat files in the current working directory (``CONFIG.script_path``).
    
    Parameters
    ----------
    ``[opt]`` file_list : ``None`` or list of str, default : ``None``
        List of path, or None.
    
    Returns
    -------
    ls_nomfich : list of str
        List of path of active files.
    """ 
    ls_nomfich = []
    if file_list == None:
        ls_nomfich = glob.glob("*.dat")
    else:
        for f in file_list:
            # On en profile pour retirer les guillemets au cas où...
            ls_nomfich.append(f.replace('"',''))
            
    return ls_nomfich




#------------------------------------------------------------------------------#
# pandas dataset management                                                    #
#------------------------------------------------------------------------------#
def check_time_date(f,sep):
    """
    Load dataframe from file.\n
    Detect and remove ``"Time"`` or ``"Date"`` column labels that are empty.\n
    Convert dataframe to numeric (if possible).
    
    Parameters
    ----------
    f : str
        File name or path to load.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    
    Returns
    -------
    data : dataframe
        Output dataframe.
    
    Notes
    -----
    If such column is detected, raises a warning and delete it.
    
    Raises
    ------
    * File not found.
    * Wrong separator.
    
    See Also
    --------
    ``pop_and_dec``
    """ 
    try:
        # Chargement des données
        data = pd.read_csv(f,sep=sep)
        cols_to_drop = []
        if len(data.columns) == 1:
            raise OSError("File '{}' does not have '{}' as its separator.".format(f,repr(sep)))
        try:
            # Champ "Time" de format incorrect
            if ":" not in str(data.at[0,"Time"]):
                warnings.warn("File '{}' seems to have a unused \"Time\" column label. Will be ignored.".format(f))
                # On retire le label du dataframe (on touche pas au fichier brut)
                data = pop_and_dec([f],"Time",sep,False,"")[0]
            else:
                cols_to_drop.append("Time")
        except KeyError:
            pass
        try:
            # Champ "Date" de format incorrect
            if "/" not in str(data.at[0,"Date"]):
                warnings.warn("File '{}' seems to have a unused \"Date\" column label. Will be ignored.".format(f))
                # On retire le label du dataframe (on touche pas au fichier brut)
                data = pop_and_dec([f],"Date",sep,False,"")[0]
            else:
                cols_to_drop.append("Date")
        except KeyError:
            pass
        num_cols = data.columns.drop(cols_to_drop)
        # Conversion des champs en numérique, si possible
        data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
        return data
    except FileNotFoundError:
        raise FileNotFoundError("File '{}' not found.".format(f))


def manage_cols(don,col_x,col_y,col_z):
    """
    Obtain meaningful informations from active columns of the ``don`` dataframe.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    col_x : list of int
        Index of every X coordinates columns.
    col_y : list of int
        Index of every Y coordinates columns.
    col_z : list of int
        Index of every Z coordinates columns (actual data).
    
    Returns
    -------
    ncx : list of str
        Names of every X columns.
    ncy : list of str
        Names of every Y columns.
    col_T : list of str
        Names of every Z columns (actual data).
    nb_data : int
        Number of Z columns. The number of data.
    nb_channels : int
        Number of X and Y columns. The number of coils.
    nb_res : int
        The number of data per coil.
    
    Raises
    ------
    * The numbers of X and Y columns are not the same.
    * The numbers of Z columns are not multiple of the number of X/Y columns.
    """ 
    if len(col_x) != len(col_y):
        raise ValueError("Lengths of 'col_x' and 'col_y' are not equal ({} and {}).".format(len(col_x),len(col_y)))
    if len(col_z)%len(col_x) != 0:
        raise ValueError("Length of 'col_z' is not multiple of length of 'col_x' ({} and {}). Please put the same number of data per channel.".format(len(col_x),len(col_y)))
    ncx = don.columns[col_x]
    ncy = don.columns[col_y]
    col_T = don.columns[col_z]
    nb_data = len(col_z)
    nb_channels = len(col_x)
    nb_res = max(1, nb_data//nb_channels)
    return ncx, ncy, col_T, nb_data, nb_channels, nb_res


def change_date(file_list,date_str,sep='\t',replace=False,output_file_list=None,in_file=False):
    """
    Change the date of a dataframe.
    
    Notes
    -----
    Date format is *mm/dd/yyyy*.
    
    Parameters
    ----------
    file_list : (list of) str or (list of) dataframe
        List of files or loaded dataframes to process, or a single one.
    date_str : str
        New date in the correct date format.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or (list of) str, default : ``None``
        List of output files names, ordered as ``file_list``, otherwise add the suffix ``"_corr"``. Is ignored if ``replace = True``.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        file_list : dataframe
            Output dataframe list.
    
    Raises
    ------
    * File not found.
    * Wrong separator or ``"Date"`` column not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    * ``output_file_list = None`` and ``in_file = True`` with loaded data.
    * Invalid date.
    """
    # Conversion en liste si 'file_list' ou 'output_file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    if output_file_list != None and not isinstance(output_file_list,list):
        output_file_list = [output_file_list]
    if output_file_list != None and len(file_list) != len(output_file_list) and not replace and in_file:
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Pour chaque fichier/dataframe
    res_list = []
    for ic, file in enumerate(file_list):
        if isinstance(file,str):
            try:
                # Chargement des données
                df = pd.read_csv(file, sep=sep, dtype=object)
            except FileNotFoundError:
                raise FileNotFoundError("File '{}' not found.".format(file))
        else:
            df = file
            if output_file_list == None and in_file:
                raise ValueError('With loaded dataframes, please add output file names.')
        
        # Séparation de la date en mm/jj/yyyy
        oc = re.split(r"/",date_str)
        
        if len(oc) != 3:
            raise ValueError("Invalid date format (mm/dd/yyyy).")
        
        # Création de la date avec le type date, pour vérifier la validité
        datetime.datetime(int(oc[2]), int(oc[0]), int(oc[1]))
        
        # Si le dataframe a une colonne "Date"
        try:
            df["Date"]
        except KeyError:
            raise KeyError("File '{}' have no \"Date\" column. Is the separator '{}' correct ?".format(file,repr(sep)))
            
        # Changement de la date
        for i, row in df.iterrows():
            df.at[i,'Date'] = date_str
        
        # Sortie du dataframe (option)
        if not in_file:
            res_list.append(df)
        # Résultat enregistré en .dat (option)
        else:
            if replace:
                df.to_csv(file, index=False, sep=sep)
            elif output_file_list == None:
                df.to_csv(file[:-4]+"_corr.dat", index=False, sep=sep)
            else:
                df.to_csv(output_file_list[ic], index=False, sep=sep)
    if not in_file:
        return res_list
 

def pop_and_dec(file_list,colsup,sep='\t',replace=False,output_file_list=None,in_file=False):
    """
    Remove specified column name from dataframe.\n
    Does not interfere with the data.
    To be used if some column labels are not associated with any data to avoid shifts.
    
    Notes
    -----
    Most functions in CMD processes are loading data with ``check_time_date``, which should handle this issue for ``"Date"`` and ``"Time"`` columns.
    
    Parameters
    ----------
    file_list : (list of) str or (list of) dataframe
        List of files or loaded dataframes to process, or a single one.
    colsup : str
        Column label to remove.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or (list of) str, default : ``None``
        List of output files names, ordered as ``file_list``, otherwise add the suffix ``"_corr"``. Is ignored if ``replace = True``.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        file_list : dataframe
            Output dataframe list.
    
    Raises
    ------
    * File not found.
    * Wrong separator or column not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    * ``output_file_list = None`` and ``in_file = True`` with loaded data.
    """
    # Conversion en liste si 'file_list' ou 'output_file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    if output_file_list != None and not isinstance(output_file_list,list):
        output_file_list = [output_file_list]
    if output_file_list != None and len(file_list) != len(output_file_list) and not replace and in_file:
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Pour chaque fichier/dataframe
    res_list = []
    for ic, file in enumerate(file_list):
        if isinstance(file,str):
            try:
                # Chargement des données
                df = pd.read_csv(file, sep=sep)
            except FileNotFoundError:
                raise FileNotFoundError("File '{}' not found.".format(file))
        else:
            df = file
            if output_file_list == None and in_file:
                raise ValueError('With loaded dataframes, please add output file names.')
        
        # On prend la colonne voulue
        try:
            dec_ind = df.columns.get_loc(colsup)
        except KeyError:
            raise KeyError("File '{}' have no \"{}\" column. Is the separator '{}' correct ?".format(file,colsup,repr(sep)))
        # Retire le label sans toucher aux valeurs (je me suis compliqué la vie probablement)
        shift_df = pd.concat([df.iloc[:,:dec_ind],df.iloc[:,dec_ind:].shift(periods=1, axis="columns").iloc[:,1:]], axis = 1)
        
        # Sortie du dataframe (option)
        if not in_file:
            res_list.append(shift_df)
        # Résultat enregistré en .dat (option)
        else:
            if replace:
                shift_df.to_csv(file, index=False, sep=sep)
            elif output_file_list == None:
                shift_df.to_csv(file[:-4]+"_corr.dat", index=False, sep=sep)
            else:
                shift_df.to_csv(output_file_list[ic], index=False, sep=sep)
    if not in_file:
        return res_list


def switch_cols(file_list,col_a,col_b,sep='\t',replace=False,output_file_list=None,in_file=False):
    """
    Switches specified column names from dataframe.\n
    Does not interfere with the data.\n
    To be used if some columns labels are mismatched, like X and Y being misplaced.
    
    Parameters
    ----------
    file_list : (list of) str or (list of) dataframe
        List of files or loaded dataframes to process, or a single one.
    col_a : str
        First column label to switch.
    col_b : str
        Second column label to switch.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or (list of) str, default : ``None``
        List of output files names, ordered as ``file_list``, otherwise add the suffix ``"_corr"``. Is ignored if ``replace = True``.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        file_list : dataframe
            Output dataframe list.
    
    Raises
    ------
    * File not found.
    * Wrong separator or column not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    * ``output_file_list = None`` and ``in_file = True`` with loaded data.
    """
    # Conversion en liste si 'file_list' ou 'output_file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    if output_file_list != None and not isinstance(output_file_list,list):
        output_file_list = [output_file_list]
    if output_file_list != None and len(file_list) != len(output_file_list) and not replace and in_file:
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Pour chaque fichier/dataframe
    res_list = []
    for ic, file in enumerate(file_list):
        if isinstance(file,str):
            try:
                # Chargement des données
                df = pd.read_csv(file, sep=sep)
            except FileNotFoundError:
                raise FileNotFoundError("File '{}' not found.".format(file))
        else:
            df = file
            if output_file_list == None and in_file:
                raise ValueError('With loaded dataframes, please add output file names.')
        
        # Vérifier que les colonnes existent
        try:
            df[col_a]
        except KeyError:
            raise KeyError("File '{}' have no \"{}\" column. Is the separator '{}' correct ?".format(file,col_a,repr(sep)))
        try:
            df[col_b]
        except KeyError:
            raise KeyError("File '{}' have no \"{}\" column. Is the separator '{}' correct ?".format(file,col_b,repr(sep)))

        # Oui c'est juste ça en vrai mais bon, c'est pour les fichiers pas chargés on va dire
        df.rename(columns={col_a: col_b, col_b: col_a}, inplace=True)
        
        # Sortie du dataframe (option)
        if not in_file:
            res_list.append(df)
        # Résultat enregistré en .dat (option)
        else:
            if replace:
                df.to_csv(file, index=False, sep=sep)
            elif output_file_list == None:
                df.to_csv(file[:-4]+"_corr.dat", index=False, sep=sep)
            else:
                df.to_csv(output_file_list[ic], index=False, sep=sep)
    if not in_file:
        return res_list


def remove_cols(file_list,colsup_list,keep=False,sep='\t',replace=False,output_file_list=None,in_file=False):
    """
    Remove specified columns from dataframe.\n
    To be used if some columns are not significant to lighten data or improve readability.
    
    Notes
    -----
    For a more automatic procedure, see ``light_format``.\n
    Do not accept loaded dataframes because it is equivalent to ``pd.drop``.
    
    Parameters
    ----------
    file_list : (list of) str
        List of files to process, or a single one.
    colsup_list : list of str
        Column names.
    ``[opt]`` keep : bool, default : ``False``
        If the specified columns are to be kept, and removing the others instead.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or (list of) str, default : ``None``
        List of output files names, ordered as ``file_list``, otherwise add the suffix ``"_corr"``. Is ignored if ``replace = True``.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        file_list : dataframe
            Output dataframe list.
    
    Raises
    ------
    * File not found.
    * Wrong separator or columns not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    
    See also
    --------
    ``light_format, pd.drop``
    """
    # Conversion en liste si 'file_list' ou 'output_file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    if output_file_list != None and not isinstance(output_file_list,list):
        output_file_list = [output_file_list]
    if output_file_list != None and len(file_list) != len(output_file_list) and not replace and in_file:
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Pour chaque fichier/dataframe
    res_list = []
    for ic, file in enumerate(file_list):
        try:
            # Chargement des données
            df = pd.read_csv(file, sep=sep)
        except FileNotFoundError:
            raise FileNotFoundError("File '{}' not found.".format(file))
        
        # Sélection inclusive
        if keep:
            try:
                small_df = df.filter(colsup_list, axis=1)
            except KeyError:
                raise KeyError("File '{}' misses some of {} columns. Is the separator '{}' correct ?".format(file,colsup_list,repr(sep)))
        # Sélection exclusive
        else:
            try:
                small_df = df.drop(colsup_list, axis=1)
            except KeyError:
                raise KeyError("File '{}' misses some of {} columns. Is the separator '{}' correct ?".format(file,colsup_list,repr(sep)))
        
        # Sortie du dataframe (optio
        if not in_file:
            res_list.append(small_df)
        # Résultat enregistré en .dat (option)
        else:
            if replace:
                small_df.to_csv(file, index=False, sep=sep)
            elif output_file_list == None:
                small_df.to_csv(file[:-4]+"_corr.dat", index=False, sep=sep)
            else:
                small_df.to_csv(output_file_list[ic], index=False, sep=sep)
    if not in_file:
        return res_list


def remove_data(file_list,colsup_list,i_min,i_max,sep='\t',replace=False,output_file_list=None,in_file=False):
    """
    Remove data between two lines in specified columns from dataframe.\n
    In this context, deleting means setting values to ``NaN``.\n
    To be used if some column contains incorrect data.
    
    Notes
    -----
    Both first and last lines are included in the deletion.\n
    The first line of ``df`` is indexed at ``0``, since its how it is labelled in pandas.
    Consequently, line indexes may not match if opened with a regular text editor.\n
    Please reset the dataframe index before using.\n
    To ease the detection of problematic lines, the ``data_stats`` function can be used.
    
    Parameters
    ----------
    file_list : (list of) str or (list of) dataframe
        List of files or loaded dataframes to process, or a single one.
    colsup_list : list of str
        Column names.
    i_min : bool
        First line of the block.
    i_max : bool
        Last line of the block.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or (list of) str, default  ``None``
        List of output files names, ordered as ``file_list``, otherwise add the suffix ``"_corr"``. Is ignored if ``replace = True``.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        file_list : dataframe
            Output dataframe list.
    
    Raises
    ------
    * File not found.
    * Wrong separator or columns not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    * ``output_file_list = None`` and ``in_file = True`` with loaded data.
    * Indexes are not ordered correctly.
    * Index is negative.
    * Index goes beyond dataframe.
    
    See also
    --------
    ``data_stats``
    """
    # Conversion en liste si 'file_list' ou 'output_file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    if output_file_list != None and not isinstance(output_file_list,list):
        output_file_list = [output_file_list]
    if output_file_list != None and len(file_list) != len(output_file_list) and not replace and in_file:
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Début et fin invalides
    if i_min > i_max:
        raise ValueError("Ending line ({}) is before starting line ({}).".format(i_max,i_min))
    if i_min < 0:
        raise ValueError("Lines indexes must be positive ({}).".format(i_min))
    
    # Pour chaque fichier/dataframe
    res_list = []
    for ic, file in enumerate(file_list):
        if isinstance(file,str):
            try:
                # Chargement des données
                df = pd.read_csv(file, sep=sep)
            except FileNotFoundError:
                raise FileNotFoundError("File '{}' not found.".format(file))
        else:
            df = file
            if output_file_list == None and in_file:
                raise ValueError('With loaded dataframes, please add output file names.')
        
        # Sélection des colonnes
        try:
            col_list = df[colsup_list]
        except KeyError:
            raise KeyError("File '{}' misses some of {} columns. Is the separator '{}' correct ?".format(file,colsup_list,repr(sep)))
        
        # Vérifier qu'on ne dépasse pas du dataframe
        try:
            for index, row in df[i_min:i_max+1].iterrows():
                for col in col_list:
                    df.loc[index, col] = np.nan
        except:
            raise ValueError("Index {} is beyond dataframe.".format(i_max))
        
        # Sortie du dataframe (option)
        if not in_file:
            res_list.append(df)
        # Résultat enregistré en .dat (option)
        else:
            if replace:
                df.to_csv(file, index=False, sep=sep)
            elif output_file_list == None:
                df.to_csv(file[:-4]+"_corr.dat", index=False, sep=sep)
            else:
                df.to_csv(output_file_list[ic], index=False, sep=sep)
    if not in_file:
         return res_list


def data_stats(file_list,col_list,sep='\t',bins=25,n=10,**kwargs):
    """
    Prints the top and bottom ``n`` values of the requested columns.
    To be used to find extreme values that are due to glitches.
    
    Parameters
    ----------
    file_list : (list of) str or (list of) dataframe
        List of files or loaded dataframes to process, or a single one.
    col_list : list of str
        Column names.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` bins : int, default : ``25``
        Number of bars in histogram.
    ``[opt]`` n : int, default : ``10``
        Number of top values to print.
    **kwargs
        Optional arguments for ``pandas.Dataframe.hist`` that are not ``column`` (overwritten).
    
    Raises
    ------
    * File not found.
    * Wrong separator or columns not found.
    
    See also
    --------
    ``pandas.Dataframe.hist``
    """
    # Conversion en liste si 'file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    
    # Pour chaque fichier/dataframe
    for ic, file in enumerate(file_list):
        if isinstance(file,str):
            try:
                # Chargement des données
                df = pd.read_csv(file, sep=sep)
            except FileNotFoundError:
                raise FileNotFoundError("File '{}' not found.".format(file))
        else:
            df = file
        
        # Sélection des colonnes
        try:
            cl = df[col_list]
        except KeyError:
            raise KeyError("File '{}' misses some of {} columns. Is the separator \
                           '{}' correct ?".format(file,col_list,repr(sep)))
        
        # Ajout de certains paramètres aux kwargs (pour pas de conflits)
        kwargs["column"] = col_list
        kwargs["bins"] = bins
        # Forcing pour mettre une meilleur taille initiale
        if "figsize" not in kwargs:
            kwargs["figsize"] = (CONFIG.fig_width,CONFIG.fig_height)
        # Histogramme
        df.hist(**kwargs)
        
        # Print des valeurs extrêmes
        if isinstance(file,str):
            print(CONFIG.warning_color+"<<< {} >>>".format(file))
        else:
            print(CONFIG.warning_color+"<<< {} >>>".format(ic))
        for c in cl:
            print(CONFIG.bold_color+"- {} -".format(c))
            print(CONFIG.und_color+"[MIN]"+CONFIG.base_color)
            print(df.nsmallest(n, c)[c])
            print(CONFIG.und_color+"[MAX]"+CONFIG.base_color)
            print(df.nlargest(n, c)[c])


def light_format(file_list,sep='\t',replace=False,output_file_list=None,nb_channels=3,
                 restr=None,meta=True,split=False,in_file=False):
    """
    Sort columns to match the following structure :\n
    ``X_int_1|Y_int_1|data1_1|data1_2|...|X_int_2|...|File_id|B+P|Base|Profil``\n
    Any other column is deleted.
    To be used if some columns are not significant to lighten data or improve readability.
    
    Notes
    -----
    Data columns are detected as long as they have the coil index in their name.
    If some of them are still not to be included, use the exclusion parameter ``restr``.\n
    For a less strict approach, see ``remove_cols``.\n
    ``restr = ['']`` (which is supposed to be an empty list) is equivalent to ``None``.
    
    Parameters
    ----------
    file_list : (list of) str or (list of) dataframe
        List of files or loaded dataframes to process, or a single one.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or list of str, default : ``None``
        List of output files names, ordered as ``file_list``, 
        otherwise add the suffix ``"_clean"``. Is ignored if ``replace = True``.
    ``[opt]`` nb_channels : int, default : ``3``
        Number of X and Y columns. The number of coils.
    ``[opt]`` restr : ``None`` or list of str, default : ``None``
        Exclusion strings: any data including one of the specified strings will be ignored. 
        If ``None``, is an empty list.
    ``[opt]`` meta : bool, default : ``True``
        If ``True``, keep metadata columns (recommended).
    ``[opt]`` split : bool, default : ``False``
        If ``True``, splits each file by channel.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        file_list : dataframe
            Output dataframe list.
    
    Raises
    ------
    * File not found.
    * Wrong separator or columns not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    * ``output_file_list = None`` and ``in_file = True`` with loaded data
    
    See also
    --------
    ``remove_cols``
    """
    # Conversion en liste si 'file_list' ou 'output_file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    if output_file_list != None and not isinstance(output_file_list,list):
        output_file_list = [output_file_list]
    if output_file_list != None and len(file_list) != len(output_file_list) and not replace and in_file:
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Bon format
    if restr in [[''],None]:
        restr = []
    
    # Pour chaque fichier/dataframe
    res_list = []
    for ic, file in enumerate(file_list):
        if isinstance(file,str):
            try:
                # Chargement des données
                df = pd.read_csv(file, sep=sep)
            except FileNotFoundError:
                raise FileNotFoundError("File '{}' not found.".format(file))
        else:
            df = file
            if output_file_list == None and in_file:
                raise ValueError('With loaded dataframes, please add output file names.')
        
        # On construit un dataframe de zéro
        clean_df = pd.DataFrame()
        # On prend les colonnes position + données interpolées
        for e in range(nb_channels):
            try:
                ncx = "X_int_"+str(e+1)
                ncy = "Y_int_"+str(e+1)
                clean_df[ncx] = df[ncx]
                clean_df[ncy] = df[ncy]
                for c in df.columns:
                    if str(e+1) in c and c not in [ncx,ncy] and not any([r in c for r in restr]):
                        clean_df[c] = df[c]
            except KeyError:
                raise KeyError("File '{}' misses some of interpolated position columns. Is the separator '{}' correct ?".format(file,repr(sep)))
            
        # On prend 4 colonnes enplus
        if meta:
            end_cols = ["File_id","B+P","Base","Profil"]
            for c in end_cols:
                try:
                    clean_df[c] = df[c]
                except KeyError:
                    raise KeyError("File '{}' have no \"{}\" column.".format(file,c))
        
        # Séparation par voie
        if split:
            clean_df_list = []
            start_list = [0]
            for e in range(1,nb_channels):
                start_list.append(list(clean_df.columns).index("X_int_"+str(e+1)))
            if meta:
                start_list.append(list(clean_df.columns).index(end_cols[0]))
            else:
                start_list.append(len(clean_df.columns))
                
            for e in range(nb_channels):
                df_temp = clean_df.filter(clean_df.columns[start_list[e]:start_list[e+1]], axis=1)
                if meta:
                    for c in end_cols:
                        df_temp[c] = clean_df[c]
                clean_df_list.append(df_temp)
            
            # Sortie du dataframe (option)
            if not in_file:
                for cdf in clean_df_list:
                    res_list.append(cdf)
            # Résultat enregistré en .dat (option)
            else:
                for e,cdf in enumerate(clean_df_list):
                    if replace:
                        cdf.to_csv(file[:-4]+"_"+str(e+1)+".dat", index=False, sep=sep)
                    elif output_file_list == None:
                        cdf.to_csv(file[:-4]+"_clean_"+str(e+1)+".dat", index=False, sep=sep)
                    else:
                        cdf.to_csv(output_file_list[ic][:-4]+"_"+str(e+1)+".dat", index=False, sep=sep)
        # Pas de séparation
        else:
            # Sortie du dataframe (option)
            if not in_file:
                res_list.append(clean_df)
            # Résultat enregistré en .dat (option)
            else:
                if replace:
                    clean_df.to_csv(file, index=False, sep=sep)
                elif output_file_list == None:
                    clean_df.to_csv(file[:-4]+"_clean.dat", index=False, sep=sep)
                else:
                    clean_df.to_csv(output_file_list[ic], index=False, sep=sep)
    if not in_file:
        return res_list


def change_sep(file_list,sep,new_sep,replace=False,output_file_list=None):
    """
    Change dataframe sepator in file.\n
    To be used if files with different separators are to be used in a single operation.
    
    Parameters
    ----------
    file_list : (list of) str
        List of files to process, or a single one.
    sep : str
        Dataframe old separator.
    new_sep : str
        Dataframe new separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or (list of) str, default : ``None``
        List of output files names, ordered as ``file_list``, otherwise add the suffix ``"_corr"``. Is ignored if ``replace = True``.
    
    Returns
    -------
    none, but save output dataframe in a .dat
    
    Warns
    -----
    * Only one column found (wrong separator).
    
    Raises
    ------
    * File not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    """
    # Conversion en liste si 'file_list' ou 'output_file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    if output_file_list != None and not isinstance(output_file_list,list):
        output_file_list = [output_file_list]
    if output_file_list != None and len(file_list) != len(output_file_list) and not replace:
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Pour chaque fichier/dataframe
    for ic, file in enumerate(file_list):
        try:
            # Chargement des données
            df = pd.read_csv(file, sep=sep, dtype=object)
        except FileNotFoundError:
            raise FileNotFoundError("File '{}' not found.".format(file))
        
        if len(df.columns) == 1:
            warnings.warn("File '{}' does not have '{}' as its separator : case ignored.".format(file,repr(sep)))
            continue
        
        # Résultat enregistré en .dat
        if replace:
            df.to_csv(file, index=False, sep=new_sep)
        elif output_file_list == None:
            df.to_csv(file[:-4]+"_corr.dat", index=False, sep=new_sep)
        else:
            df.to_csv(output_file_list[ic], index=False, sep=new_sep)


def no_gps_pos(file_list,sep='\t',replace=False,output_file_list=None,in_file=False):
    """
    Fuses a list of prospection files in one .dat\n
    This procedure works as soon as all columns are matching 
    
    Notes
    -----
    Files are put in the same order as in ``file_list``.\n
    All files must have the same columns, but the order is not important (will match the order of the first one).
    
    Parameters
    ----------
    file_list : (list of) str or (list of) dataframe
        List of files or loaded dataframes to process, or a single one.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or (list of) str, default : ``None``
        List of output files names, ordered as ``file_list``, otherwise add the suffix ``"_pos"``. Is ignored if ``replace = True``.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        file_list : dataframe
            Output dataframe list.
    
    Raises
    ------
    * File not found.
    * Wrong separator or columns not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    * ``output_file_list = None`` and ``in_file = True`` with loaded data.
    * ``y[m]`` column not found.
    """
    # Conversion en liste si 'file_list' ou 'output_file_list' ne l'est pas
    if not isinstance(file_list,list):
        file_list = [file_list]
    if output_file_list != None and not isinstance(output_file_list,list):
        output_file_list = [output_file_list]
    if output_file_list != None and len(file_list) != len(output_file_list) and not replace and in_file:
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Pour chaque fichier/dataframe
    res_list = []
    for ic, file in enumerate(file_list):
        if isinstance(file,str):
            try:
                # Chargement des données
                df = pd.read_csv(file, sep=sep)
                if len(df.columns) < 2:
                    raise OSError("File '{}' does not have '{}' as its separator.".format(file,repr(sep)))
            except FileNotFoundError:
                raise FileNotFoundError("File '{}' not found.".format(file))
        else:
            df = file
            if output_file_list == None and in_file:
                raise ValueError('With loaded dataframes, please add output file names.')
        
        # Vérification sur la colonne
        try:
            df["y[m]"]
        except KeyError:
            raise KeyError("File '{}' have no \"y[m]\" column, wrong format.".format(file))        
        # On prend les points sans données (plus un 0 au début) pour délimiter les profils
        pos_pts=df[df[df.columns[2]].isna()].index.insert(0,-1)
        
        # Pour chaque profil, on fait une régression sur la position
        for ic,index_fin in enumerate(pos_pts[1:]):
            index_deb = pos_pts[ic]+1
            deb = df.loc[index_deb,"y[m]"]
            fin = df.loc[index_fin,"y[m]"]
            y_pos = np.round(np.linspace(deb,fin,index_fin-index_deb,endpoint=True),2)
            df.loc[index_deb:index_fin-1,"y[m]"] = y_pos
        
        # Retrait des points sans données
        df.dropna(subset = df.columns[2],inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        # Sortie du dataframe (option)
        if not in_file:
            res_list.append(df)
        # Résultat enregistré en .dat (option)
        else:
            if replace:
                df.to_csv(file, index=False, sep=sep)
            elif output_file_list == None:
                df.to_csv(file[:-4]+"_pos.dat", index=False, sep=sep)
            else:
                df.to_csv(output_file_list[ic], index=False, sep=sep)
    if not in_file:
        return res_list


def fuse_data(file_list,sep='\t',output_file="fused.dat",in_file=True):
    """
    Fuses a list of prospection files in one .dat\n
    This procedure works as soon as all columns are matching 
    
    Notes
    -----
    Files are put in the same order as in ``file_list``.\n
    All files must have the same columns, but the order is not important (will match the order of the first one).\n
    Do not accept loaded dataframes because it is equivalent to ``pd.concat``.
    
    Parameters
    ----------
    file_list : list of str
        List of files to process.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` output_file : str, default : "fused.dat"
        Output file name.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        big_df : dataframe
            Output base dataframe.
    
    Raises
    ------
    * Not enough files (at least 2).
    * File not found.
    * Wrong separator or columns not found.
    * Columns are not matching.
    * Error during ``pd.concat``.
    
    See also
    --------
    ``pd.concat``
    """
    if len(file_list) < 2:
        raise ValueError("Need at least 2 files.")
    df_list = []
    
    # Pour chaque fichier/dataframe
    for ic, file in enumerate(file_list):
        try:
            df = pd.read_csv(file, sep=sep)
            if len(df.columns) < 2:
                raise OSError("File '{}' does not have '{}' as its separator.".format(file,repr(sep)))
            df_list.append(df)
            if set(df.columns) != set(df_list[0].columns):
                raise KeyError("Columns of {} and {} are not matching.".format(file_list[0],file_list[ic]))
        except FileNotFoundError:
            raise FileNotFoundError("File '{}' not found.".format(file))
    
    # L'entièreté de l'action finalement
    try:
        big_df = pd.concat(df_list)
    except Exception as e:
        raise FileNotFoundError("Fuse error : {}".format(e))
    
    # Sortie du dataframe (option)
    if not in_file:
        return big_df
    
    # Résultat enregistré en .dat (option)
    big_df.to_csv(output_file, index=False, sep=sep)