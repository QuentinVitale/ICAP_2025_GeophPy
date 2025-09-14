# -*- coding: utf-8 -*-
'''
   geophpy.georeferencing.crs
   ------------------------------

   DataSet Object general geopositioning routines.

   :copyright: Copyright 2014-2019 Lionel Darras, Philippe Marty, Quentin Vitale and contributors, see AUTHORS.
   :license: GNU GPL v3.

'''

import numpy as np
import geophpy.georeferencing.utils as pset
from scipy import interpolate as interp

def setgeoref(dataset, refsystem, points_list, utm_zoneletter=None, utm_zonenumber=None, valfilt=False):
    ''' Dataset Georeferencing using 2-D linear interpolation.

    cf. :meth:`~geophpy.dataset.DataSet.setgeoref`

    '''

    # /!\ GCPs must be grid nodes INCLUDING the dataset extend (no extrapolation possible so far)

    # Not enough points
    ### Raise Error ? ###
    if not len(points_list) >= 4:
        return -1

    points_list = np.asarray(points_list)

    # Conversion to UTM for WGS84 system
    if refsystem == 'WGS84':
        lat = points_list.T[2]
        long = points_list.T[1]
        east_gcp, north_gcp = pset.wgs84_to_utm(lat, long)

    # Other systems
    else:
        east_gcp = points_list.T[1]
        north_gcp = points_list.T[2]

    # Selecting ungridded or gridded values
    is_ungridded = valfilt or dataset.data.z_image is None
    if is_ungridded:
        X, Y = dataset.data.values.T[:2]

    else:
        X, Y = dataset.get_xygrid()

    # Linear interpolation georeferenced coordinates at data positions
    ###
    ##
    # ...TODO... manage extrapolation possibility, maybe with nearest neighbour.
    ##
    ###
    x_gcp = points_list.T[3]
    y_gcp = points_list.T[4]

    east = interp.griddata((x_gcp, y_gcp), east_gcp, (X, Y), method='linear')
    north = interp.griddata((x_gcp, y_gcp), north_gcp, (X, Y), method='linear')

    # Storing results in dataset
    is_error = (np.isnan(east).any()
                or np.isnan(north).any())
    if is_error:
        error = -2     # dataset zone greater than points list area for instance

    else:
        error = 0
        if is_ungridded:
            dataset.data.east = east
            dataset.data.north = north
        else:
            dataset.data.easting_image = east
            dataset.data.northing_image = north
        dataset.georef.active = True
        dataset.georef.refsystem = refsystem
        dataset.georef.points_list = points_list
        dataset.georef.utm_zoneletter = utm_zoneletter
        dataset.georef.utm_zonenumber = utm_zonenumber

    return error
