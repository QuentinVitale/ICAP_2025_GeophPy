# -*- coding: utf-8 -*-
'''
    geophpy.georeferencing.raster
    -----------------------------

    SIG raster format management.

    :copyright: Copyright 2014-2019 Lionel Darras, Philippe Marty and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import os
from PIL import Image
#import numpy as np

class WorldFileParams:
    ''' Class to hold world file parameters. '''

    A = 1.0        # xprim = Ax + By + C
    D = 0.0        # yprim = Dx + Ey + F
    B = 0.0
    E = 1.0
    C = 0.0        # utm easting position of the center of the left-upper corner point
    F = 0.0        # utm northing position of the center of the left-upper corner point

# list of picture format files availables for raster
RASTERFORMAT_LIST = ['.jpg', '.png', '.tiff']


def getrasterformatlist():
    '''Get the the list of raster formats availables
    Returns :
        - RASTERFORMAT_LIST : list of raster formats.
    '''
    return RASTERFORMAT_LIST


def israsterformat(filename):
    '''
    Detects if the raster picture format is available
    '''
    splitedfilename = os.path.splitext(filename)
    extension = (splitedfilename[-1]).lower()

    is_raster_frmt = False
    for ext in RASTERFORMAT_LIST:
        if extension == ext:
            is_raster_frmt = True
            break

    return is_raster_frmt


def picture_to_worldfile(picturefilename, quadcoords):
    '''
    Builds a tfw file associated with an image to display.

    Parameters :

    :picturefilename: Picture filename, .jpg, .png, .tiff, ...

    :quadcoords: Array of the 4 corner utm positions (easting, northing), the first is the left bottom corner, and the others in clock wise direction

    Returns :

    :success: True if worldfile has been created, False if a problem occured.

    '''

    if len(quadcoords) != 4:   # if not 4 points of 2 coordinates, unsuccess
        success = False
    else:
        splitedfilename = os.path.splitext(picturefilename)
        picture_extension = (splitedfilename[-1]).lower()
        success = True
        if picture_extension == ".tiff":
            worldfile_extension = ".tfw"
        elif picture_extension == ".jpg":
            worldfile_extension = ".jgw"
        elif picture_extension == ".png":
            worldfile_extension = ".pgw"
        else:
            success = False

    if success:
        worldfilename = splitedfilename[0] + worldfile_extension

        im = Image.open(picturefilename)

        x_size, y_size = im.size

        worldfileparams = WorldFileParams()
        worldfileparams.A = (quadcoords[1][0] - quadcoords[0][0])/x_size
        worldfileparams.B = (quadcoords[0][0] - quadcoords[3][0])/y_size
        worldfileparams.C = quadcoords[3][0] - (0.5*worldfileparams.A)
        worldfileparams.D = (quadcoords[1][1] - quadcoords[0][1])/x_size
        worldfileparams.E = (quadcoords[0][1] - quadcoords[3][1])/y_size
        worldfileparams.F = quadcoords[3][1] - (0.5*worldfileparams.E)

        worldfile = open(worldfilename, "w")
        worldfile.write(str(worldfileparams.A) + "\n")
        worldfile.write(str(worldfileparams.D) + "\n")
        worldfile.write(str(worldfileparams.B) + "\n")
        worldfile.write(str(worldfileparams.E) + "\n")
        worldfile.write(str(worldfileparams.C) + "\n")
        worldfile.write(str(worldfileparams.F) + "\n")

    return success
