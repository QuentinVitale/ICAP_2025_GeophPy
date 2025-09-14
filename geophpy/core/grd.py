# -*- coding: utf-8 -*-.
"""
   geophpy.core.grd
   -------------------------

   Module to manage Golden Software Surfer grid files input and output.

   :copyright: Copyright 2018-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.
"""

from .registry import register_reader, register_writer  # to register reader and writer functions

import os
import struct
import numpy as np

#from geophpy.core.registry import register_reader, register_writer  # to register reader and writer functions

# --- Constants ---
VALID_GRID_FORMAT = ['surfer7bin', 'surfer6bin', 'surfer6ascii']
VALID_GRID_LIST = ['Surfer 7 binary Grid', 'Surfer 6 binary Grid', 'Surfer 6 ascii Grid']

# --- Low-Level Functions ---
def read_format_from_file(filename):
    ''' Return the  the grid file format.

    Parameters
    ----------
    filename : str
        Name of the grid file.

    Returns
    -------

    frmt : str {'surfer7bin', 'surfer6bin', 'surfer6ascii', None}
        Format of the grid file to read.

    '''

    if not os.path.exists(filename):
        raise OSError(2, 'No such file or directory', filename)

    file = open(filename, 'rb')
    tag_id = struct.unpack('4s', file.read(4))[0]
    file.close()

    frmt = None
    # Surfer 7 binary grid tag
    if tag_id == b'DSRB':
        #reader = self.read_surfer7bin
        frmt = 'surfer7bin'

    # Surfer 6 binary grid tag
    if tag_id == b'DSBB':
        #reader = self.read_surfer6bin
        frmt = 'surfer6bin'

    # Surfer 6 ascii grid tag
    if tag_id == b'DSAA':
        #reader = self.read_surfer6ascii
        frmt = 'surfer6ascii'

    # Unknown tag
    if frmt is None:
        raise TypeError(('Invalid file identifier for Surfer .grd file. '
                         'Must be DSRB, DSBB, or DSAA'))

    return frmt


def read_surfer7bin(filename):
    ''' Read Golden Software Surfer 7 binary grid format.

    Based on  Seequent's steno3d_surfer/parser (MIT License).

    Returns
    -------
    data : dict
        Dictionary containing the data.
        Dictionary keys are:
            * 'nrow' : number of rows in the grid.
            * 'ncol' : number of columns in the grid.
            * 'xll' : coordinate of the lower left corner of the grid.
            * 'yll' : coordinate of the lower left corner of the grid.
            * 'xmin' : minimum X value of the grid.
            * 'xmax' : maximum X value of the grid.
            * 'ymin' : minimum Y value of the grid.
            * 'ymax' : maximum Y value of the grid.
            * 'xsize' : spacing between adjacent nodes in the X direction (between columns).
            * 'ysize' : spacing between adjacent nodes in the Y direction (between rows).
            * 'zmin' : minimum Z value of the grid.
            * 'zmax' : maximum Z value of the grid.
            * 'values' : Z value of the grid.
            * 'blankvalue' : blank value (nodes are blanked if greater or equal to this value).

    '''

    if not os.path.exists(filename):
        raise OSError(2, 'No such file or directory', filename)

    with open(filename, "rb") as file:

        # Headers section
        # Valid surfer 7 binary grid file
        tag_id = struct.unpack('4s', file.read(4))[0]
        if  tag_id != b'DSRB':
            raise TypeError(('Invalid surfer 7 binary grid file'
                             'First 4 characters must be DSRB'))

        # Passing headers
        _ = struct.unpack('<i', file.read(4))[0]  # Size of Header section
        _ = struct.unpack('<i', file.read(4))[0]  # Header Section Version
                                                       # 1: value >= blankvalue blanked.
                                                       # 2: value == blankvalue blanked.

        # Grid section
        tag_id = struct.unpack('4s', file.read(4))[0]
        if tag_id != b'GRID':
            raise TypeError(('Invalid Surfer 7 binary grid file '
                             'GRID section expected directly after '
                             'the HEADERS section but %s encountered') %(tag_id))

        # Length in bytes of the grid section
        nbytes = struct.unpack('<i', file.read(4))[0]
        if nbytes != 72:
            raise TypeError(('Invalid Surfer 7 binary grid file '
                             'Expected length in bytes of the grid section '
                             'is 72 but %d encountered.') % (nbytes))

        # Grid info
        nrow = struct.unpack('<i', file.read(4))[0]  # int
        ncol = struct.unpack('<i', file.read(4))[0]  # int
        xll = struct.unpack('<d', file.read(8))[0]  # double
        yll = struct.unpack('<d', file.read(8))[0]  # double
        xsize = struct.unpack('<d', file.read(8))[0]  # double
        ysize = struct.unpack('<d', file.read(8))[0]  # double
        zmin = struct.unpack('<d', file.read(8))[0]  # double
        zmax = struct.unpack('<d', file.read(8))[0]  # double
        _ = struct.unpack('<d', file.read(8))[0]  # double rotation
        blankvalue = struct.unpack('<d', file.read(8))[0]  # double

        xmin = xll
        ymin = yll
        xmax = xmin + xsize*(ncol-1)
        ymax = ymin + ysize*(nrow-1)

        # Data section
        tag_id = struct.unpack('4s', file.read(4))[0]  # str
        if tag_id != b'DATA':
            raise TypeError(('Invalid Surfer 7 binary grid file '
                             'DATA section expected directly after '
                             'the GRID section but %s encountered.') %(tag_id))

        datalen = struct.unpack('<i', file.read(4))[0] #  Length in bytes of the data section

        if datalen != ncol*nrow*8:
            raise TypeError(('Invalid Surfer 7 binary grid file '
                             'Inconsistency between expected DATA '
                             'Length and nrow, ncol. '
                             'Expected length is (%d) but %s encountered.') %(ncol*nrow*8, datalen))

        # Data
        values = np.empty((nrow, ncol))

        for row in range(nrow):  # Y
            for col in range(ncol): # X
                values[row][col] = struct.unpack('<d', file.read(8))[0] # float

        values[values >= blankvalue] = np.nan

        data = {'nrow' : nrow,
                'ncol' : ncol,
                'xll' : xll,
                'yll' : yll,
                'xmin' : xmin,
                'xmax' : xmax,
                'ymin' : ymin,
                'ymax' : ymax,
                'xsize' : xsize,
                'ysize' : ysize,
                'zmin' : zmin,
                'zmax' : zmax,
                'values' : values,
                'blankvalue' : blankvalue
               }

        return data


def read_surfer6bin(filename):
    ''' Read Golden Software Surfer 6 binary grid formats.

    Based on  Seequent's steno3d_surfer/parser (MIT License).

    Returns
    -------
    data : dict
        Dictionary containing the data.
        Dictionary keys are:
            * 'nrow' : number of rows in the grid.
            * 'ncol' : number of columns in the grid.
            * 'xll' : coordinate of the lower left corner of the grid.
            * 'yll' : coordinate of the lower left corner of the grid.
            * 'xmin' : minimum X value of the grid.
            * 'xmax' : maximum X value of the grid.
            * 'ymin' : minimum Y value of the grid.
            * 'ymax' : maximum Y value of the grid.
            * 'xsize' : spacing between adjacent nodes in the X direction (between columns).
            * 'ysize' : spacing between adjacent nodes in the Y direction (between rows).
            * 'zmin' : minimum Z value of the grid.
            * 'zmax' : maximum Z value of the grid.
            * 'values' : Z value of the grid.
            * 'blankvalue' : blank value (nodes are blanked if greater or equal to this value).

    '''

    if not os.path.exists(filename):
        raise OSError(2, 'No such file or directory', filename)

    with open(filename, "rb") as file:

        # Valid surfer 6 binary grid file
        tag_id = struct.unpack('4s', file.read(4))[0]
        if  tag_id != b'DSBB':
            raise TypeError(('Invalid surfer 6 binary grid file'
                             'First 4 characters must be DSBB'))

        # Grid info
        ncol = struct.unpack('<h', file.read(2))[0]  # short
        nrow = struct.unpack('<h', file.read(2))[0]  # short
        xmin = struct.unpack('<d', file.read(8))[0]  # double
        xmax = struct.unpack('<d', file.read(8))[0]  # double
        ymin = struct.unpack('<d', file.read(8))[0]  # double
        ymax = struct.unpack('<d', file.read(8))[0]  # double
        zmin = struct.unpack('<d', file.read(8))[0]  # double
        zmax = struct.unpack('<d', file.read(8))[0]  # double

        xsize = (xmax-xmin)/(ncol-1)
        ysize = (ymax-ymin)/(nrow-1)
        xll = xmin
        yll = ymin

        blankvalue = 1.701410009187828e+38

        # Data
        values = np.empty((nrow, ncol))

        for row in range(nrow):  # Y
            for col in range(ncol): # X
                values[row][col] = struct.unpack('<f', file.read(4))[0] # float

        values[values >= blankvalue] = np.nan

        data = {'nrow' : nrow,
                'ncol' : ncol,
                'xll' : xll,
                'yll' : yll,
                'xmin' : xmin,
                'xmax' : xmax,
                'ymin' : ymin,
                'ymax' : ymax,
                'xsize' : xsize,
                'ysize' : ysize,
                'zmin' : zmin,
                'zmax' : zmax,
                'values' : values,
                'blankvalue' : blankvalue
                }

    return data


def read_surfer6ascii(filename):
    ''' Read Golden Software Surfer 6 ASCII grid formats.

    Based on  Seequent's steno3d_surfer/parser (MIT License).

    Returns
    -------
    data : dict
        Dictionary containing the data.
        Dictionary keys are:
            * 'nrow' : number of rows in the grid.
            * 'ncol' : number of columns in the grid.
            * 'xll' : coordinate of the lower left corner of the grid.
            * 'yll' : coordinate of the lower left corner of the grid.
            * 'xmin' : minimum X value of the grid.
            * 'xmax' : maximum X value of the grid.
            * 'ymin' : minimum Y value of the grid.
            * 'ymax' : maximum Y value of the grid.
            * 'xsize' : spacing between adjacent nodes in the X direction (between columns).
            * 'ysize' : spacing between adjacent nodes in the Y direction (between rows).
            * 'zmin' : minimum Z value of the grid.
            * 'zmax' : maximum Z value of the grid.
            * 'values' : Z value of the grid.
            * 'blankvalue' : blank value (nodes are blanked if greater or equal to this value).

    '''

    if not os.path.exists(filename):
        raise OSError(2, 'No such file or directory', filename)

    with open(filename, "r") as file:

        # Valid surfer 6 binary grid file
        tag_id = file.readline().strip()
        if  tag_id != 'DSAA':
            raise TypeError('Invalid surfer 6 ASCII grid file '
                            'First 4 characters must be DSAA')
        # Grid info
        ncol, nrow = [int(n) for n in file.readline().split()]
        xmin, xmax = [float(n) for n in file.readline().split()]
        ymin, ymax = [float(n) for n in file.readline().split()]
        zmin, zmax = [float(n) for n in file.readline().split()]

        xsize = (xmax-xmin)/(ncol-1)
        ysize = (ymax-ymin)/(nrow-1)
        xll = xmin
        yll = ymin

        blankvalue = 1.70141e38

        # Data
        values = np.empty((nrow, ncol))

        for i in range(nrow):
            values[i, :] = [float(n) for n in file.readline().split()]

        values[np.where(values >= blankvalue)] = np.nan

        data = {'nrow' : nrow,
                'ncol' : ncol,
                'xll' : xll,
                'yll' : yll,
                'xmin' : xmin,
                'xmax' : xmax,
                'ymin' : ymin,
                'ymax' : ymax,
                'xsize' : xsize,
                'ysize' : ysize,
                'zmin' : zmin,
                'zmax' : zmax,
                'values' : values,
                'blankvalue' : blankvalue
                }

    return data

GRID_READER_MAP = {
    'surfer7bin': read_surfer7bin,
    'surfer6bin': read_surfer6bin,
    'surfer6ascii': read_surfer6ascii
}

# GRID_READER = {'surfer7bin' : read_surfer7bin,
#                'surfer6bin' : read_surfer6bin,
#                'surfer6ascii': read_surfer6ascii
#                }

# def read_grd(filename, frmt=None):
#     '''
#     Read data from Golden Software Surfer grid format.

#     Parameters
#     ----------
#     filename : str
#         Name of the grid file to read.

#     frmt : str {None, 'surfer7bin', 'surfer6bin', 'surfer6ascii'}, optional
#         Format of the grid file to read.
#         If None (by default), it will be determined from the file itself.

#     Returns
#     -------
#     data : dict
#         Dictionary containing the data.
#         Dictionary keys are:
#             * 'nrow' : number of rows in the grid.
#             * 'ncol' : number of columns in the grid.
#             * 'xll' : coordinate of the lower left corner of the grid.
#             * 'yll' : coordinate of the lower left corner of the grid.
#             * 'xmin' : minimum X value of the grid.
#             * 'xmax' : maximum X value of the grid.
#             * 'ymin' : minimum Y value of the grid.
#             * 'ymax' : maximum Y value of the grid.
#             * 'xsize' : spacing between adjacent nodes in the X direction (between columns).
#             * 'ysize' : spacing between adjacent nodes in the Y direction (between rows).
#             * 'zmin' : minimum Z value of the grid.
#             * 'zmax' : maximum Z value of the grid.
#             * 'values' : Z value of the grid.
#             * 'blankvalue' : blank value (nodes are blanked if greater or equal to this value).

#     '''

#     if not os.path.exists(filename):
#         raise OSError(2, 'No such file or directory', filename)

#     # Format from file
#     data = None
#     if frmt is None:
#         frmt = read_format_from_file(filename)

#     # Using given format
#     if frmt in VALID_GRID_FORMAT:
#         readfun = GRID_READER[frmt]
#         data = readfun(filename)

#     else:
#         raise TypeError(('Invalid file identifier for Surfer .grd file. '
#                          'Must be DSRB, DSBB, or DSAA'))

#     return data


def write_surfer7bin(filename, data, version=2):
    ''' Write data to a Golden Software Surfer 7 binary grid file.

    Parameters
    ----------
    filename : str
        Output file name.

    data : dict
        Dictionary containing the data.
        Dictionary keys are:
            * 'nrow' : number of rows in the grid.
            * 'ncol' : number of columns in the grid.
            * 'xll' [optional] : coordinate of the lower left corner of the grid.
                Mandatory if 'xmin' is not provided.
            * 'yll' [optional] : coordinate of the lower left corner of the grid.
                Mandatory if 'ymin' is not provided.
            * 'xmin' [optional] : minimum X value of the grid.
                Mandatory if 'xll' is not provided.
            * 'ymin' [optional] : minimum Y value of the grid.
                Mandatory if 'yll' is not provided.
            * 'xsize' : spacing between adjacent nodes in the X direction (between columns).
            * 'ysize' : spacing between adjacent nodes in the Y direction (between rows).
            * 'zmin' : minimum Z value of the grid.
            * 'zmax' : maximum Z value of the grid.
            * 'values' : Z value of the grid.

    version : {1, 2}
        Version number of the file format. Can be set to 1 or 2.
        If the version field is 1, then any value >= blankvalue will be blanked
        using Surfer’s blanking value, 1.70141e+038.
        If the version field is 2, then any value == blankvalue will be blanked
        using Surfer’s blanking value, 1.70141e+038.

    Returns
    -------
    error : int
        0 if no error.

    '''

    error = 1

    # Recovering grid info
    ncol = data.get('ncol', None)
    nrow = data.get('nrow', None)
    xll = data.get('xll', None)
    yll = data.get('yll', None)
    xmin = data.get('xmin', None)
    ymin = data.get('ymin', None)
    xsize = data['xsize']
    ysize = data['ysize']
    zmin = data['zmin']
    zmax = data['zmax']
    values = data['values']

    if nrow is None:
        nrow = values.shape[0]

    if ncol is None:
        ncol = values.shape[1]

    if xmin is None and ymin is None and xll is None and yll is None:
        raise TypeError('Missing grid information. '
                        '"xll" and "yll" or "xmin", "ymin" should be provided.')

    xll = [item for item in [xmin, xll] if item is not None]
    yll = [item for item in [ymin, yll] if item is not None]
    if not xll:
        raise TypeError('Missing grid information. '
                        '"xll" or "xmin" should be provided.')
    if not yll:
        raise TypeError('Missing grid information. '
                        '"yll" or "ymin" should be provided.')
    xll = xll[0]
    yll = yll[0]

    # Converting nans to blanks
    rotation = 0
    blankvalue = 1.70141e38
    values[np.isnan(values)] = blankvalue

    with open(filename, "wb") as file:

        # Headers
        headersize = 4
        headerversion = version
        file.write(struct.pack('4s', b'DSRB'))  # DSRB: Surfer 7 binary grid tag

        file.write(struct.pack('<l', headersize))  # Size of Header section
        file.write(struct.pack('<l', headerversion))  # Header Section Version
                                                       # 1: value >= blankvalue blanked.
                                                       # 2: value == blankvalue blanked.

        # Grid section
        file.write(struct.pack('4s', b'GRID'))  # Tag: ID indicating a grid section
        file.write(struct.pack('<l', 72))  # Tag: Length in bytes of the grid section

        file.write(struct.pack('<l', nrow))  # long
        file.write(struct.pack('<l', ncol))  # long
        file.write(struct.pack('<d', xll))  # double
        file.write(struct.pack('<d', yll))  # double
        file.write(struct.pack('<d', xsize))  # double
        file.write(struct.pack('<d', ysize))  # double
        file.write(struct.pack('<d', zmin))  # double
        file.write(struct.pack('<d', zmax))  # double
        file.write(struct.pack('<d', rotation))  # double
        file.write(struct.pack('<d', blankvalue))  # double

        # Data section
        datalen = ncol*nrow*8
        if datalen != ncol*nrow*8:
            raise TypeError(('Invalid Surfer 7 binary grid file '
                             'Inconsistency between expected DATA '
                             'Length and nrow, ncol. '
                             'Expected length is (%d) but %s encountered.') %(ncol*nrow*8, datalen))

        file.write(struct.pack('4s', b'DATA'))  # Tag: ID indicating a data section
        file.write(struct.pack('<l', datalen))  # Tag: Length in bytes of the data section
                                                #      (nrows x ncol x 8 bytes per double)

        for row in range(nrow):  # Y
            for col in range(ncol): # X
                file.write(struct.pack('<d', float(values[row][col])))  # float

        error = 0
        return error


def write_surfer6bin(filename, data):
    ''' Write Golden Software Surfer 6 binary grid formats.

    Parameters
    ----------
    filename : str
        Output file name.

    data : dict
    Dictionary containing the data.
    Dictionary keys are:
        * 'nrow' : number of rows in the grid.
        * 'ncol' : number of columns in the grid.
        * 'xll' [optional] : coordinate of the lower left corner of the grid.
            Mandatory if 'xmin' and 'xmax' are not provided.
        * 'yll' [optional] : coordinate of the lower left corner of the grid.
            Mandatory if 'ymin' and 'ymax' are not provided.
        * 'xmin' [optional] : minimum X value of the grid.
            Mandatory if 'xll' is not provided.
        * 'xmax' [optional] : maximum X value of the grid.
            Mandatory if 'xll' is not provided.
        * 'ymin' [optional] : minimum Y value of the grid.
            Mandatory if 'yll' is not provided.
        * 'ymax' [optional] : maximum Y value of the grid.
            Mandatory if 'yll' is not provided.
        * 'xsize' [optional] : spacing between adjacent nodes in the X direction (between columns).
            Mandatory if 'xmax' is not provided.
        * 'ysize' [optional] : spacing between adjacent nodes in the Y direction (between rows).
            Mandatory if 'ymax' is not provided.
        * 'zmin' : minimum Z value of the grid.
        * 'zmax' : maximum Z value of the grid.
        * 'values' : Z value of the grid.
        * 'blankvalue' : blank value (nodes are blanked if greater or equal to this value).

    Returns
    -------
    error : int
        0 if no error.

    '''

    error = 1

    # Retrieving grid info
    ncol = data.get('ncol', None)
    nrow = data.get('nrow', None)
    xll = data.get('xll', None)
    yll = data.get('yll', None)
    xmin = data.get('xmin', None)
    xmax = data.get('xmax', None)
    ymin = data.get('ymin', None)
    ymax = data.get('ymax', None)
    xsize = data.get('xsize', None)
    ysize = data.get('ysize', None)
    zmin = data['zmin']
    zmax = data['zmax']
    values = data['values']

    if nrow is None:
        nrow = values.shape[0]

    if ncol is None:
        ncol = values.shape[1]

    if xmin is None and ymin is None and xll is None and yll is None:
        raise TypeError('Missing grid information. '
                        '"xll" and "yll" or "xmin", "ymin" should be provided.')

    xmin = [item for item in [xmin, xll] if item is not None]
    ymin = [item for item in [ymin, yll] if item is not None]
    if not xmin:
        raise TypeError('Missing grid information. '
                        '"xll" or "xmin" should be provided.')
    if not ymin:
        raise TypeError('Missing grid information. '
                        '"yll" or "ymin" should be provided.')
    xmin = xmin[0]
    ymin = ymin[0]
    if xmax is None:
        if xsize is None:
            raise TypeError('Missing grid information. '
                            '"xsize" or "xmax" should be provided.')
        xmax = xmin + xsize*(ncol-1)

    if ymax is None:
        if ysize is None:
            raise TypeError('Missing grid information. '
                            '"ysize" or "ymax" should be provided.')
        ymax = ymin + ysize*(ncol-1)

    # Converting nans to blanks
    blankvalue = 1.70141e38
    values[np.isnan(values)] = blankvalue

    with open(filename, "wb") as file:

        # Surfer 6 binary grid tag
        file.write(struct.pack('4s', b'DSBB'))  # DSBB

        # Grid info
        file.write(struct.pack('<h', ncol))  # short
        file.write(struct.pack('<h', nrow))  # short
        file.write(struct.pack('<d', xmin))  # double
        file.write(struct.pack('<d', xmax))  # double
        file.write(struct.pack('<d', ymin))  # double
        file.write(struct.pack('<d', ymax))  # double
        file.write(struct.pack('<d', zmin))  # double
        file.write(struct.pack('<d', zmax))  # double

        # Data
        for row in range(nrow):  # Y
            for col in range(ncol): # X
                file.write(struct.pack('<f', float(values[row][col]))) # float

    error = 0
    return error


def write_surfer6ascii(filename, data):
    ''' Write Golden Software Surfer 6 ASCII grid formats.

    Parameters
    ----------
    filename : str
        Output file name.

    data : dict
    Dictionary containing the data.
    Dictionary keys are:
        * 'nrow' : number of rows in the grid.
        * 'ncol' : number of columns in the grid.
        * 'xll' [optional] : coordinate of the lower left corner of the grid.
            Mandatory if 'xmin' and 'xmax' are not provided.
        * 'yll' [optional] : coordinate of the lower left corner of the grid.
            Mandatory if 'ymin' and 'ymax' are not provided.
        * 'xmin' [optional] : minimum X value of the grid.
            Mandatory if 'xll' is not provided.
        * 'xmax' [optional] : maximum X value of the grid.
            Mandatory if 'xll' is not provided.
        * 'ymin' [optional] : minimum Y value of the grid.
            Mandatory if 'yll' is not provided.
        * 'ymax' [optional] : maximum Y value of the grid.
            Mandatory if 'yll' is not provided.
        * 'xsize' [optional] : spacing between adjacent nodes in the X direction (between columns).
            Mandatory if 'xmax' is not provided.
        * 'ysize' [optional] : spacing between adjacent nodes in the Y direction (between rows).
            Mandatory if 'ymax' is not provided.
        * 'zmin' : minimum Z value of the grid.
        * 'zmax' : maximum Z value of the grid.
        * 'values' : Z value of the grid. 2-D array like.
        * 'blankvalue' : blank value (nodes are blanked if greater or equal to this value).

    Returns
    -------
    error : int
        0 if no error.

    '''

    error = 1

    # Retrieving grid info
    ncol = data.get('ncol', None)
    nrow = data.get('nrow', None)
    xll = data.get('xll', None)
    yll = data.get('yll', None)
    xmin = data.get('xmin', None)
    xmax = data.get('xmax', None)
    ymin = data.get('ymin', None)
    ymax = data.get('ymax', None)
    xsize = data.get('xsize', None)
    ysize = data.get('ysize', None)
    zmin = data['zmin']
    zmax = data['zmax']
    values = data['values']

    if nrow is None:
        nrow = values.shape[0]

    if ncol is None:
        ncol = values.shape[1]

    if xmin is None and ymin is None and xll is None and yll is None:
        raise TypeError('Missing grid information. '
                        '"xll" and "yll" or "xmin", "ymin" should be provided.')

    xmin = [item for item in [xmin, xll] if item is not None]
    ymin = [item for item in [ymin, yll] if item is not None]
    if not xmin:
        raise TypeError('Missing grid information. '
                        '"xll" or "xmin" should be provided.')
    if not ymin:
        raise TypeError('Missing grid information. '
                        '"yll" or "ymin" should be provided.')
    xmin = xmin[0]
    ymin = ymin[0]
    if xmax is None:
        if xsize is None:
            raise TypeError('Missing grid information. '
                            '"xsize" or "xmax" should be provided.')
        xmax = xmin + xsize*(ncol-1)

    if ymax is None:
        if ysize is None:
            raise TypeError('Missing grid information. '
                            '"ysize" or "ymax" should be provided.')
        ymax = ymin + ysize*(ncol-1)

    # Converting nans to blanks
    blankvalue = 1.70141e38
    values[np.isnan(values)] = blankvalue

    with open(filename, "w") as file:

        # Surfer 6 ascii grid tag
        file.write("DSAA\n")  # DSAA

        # Headers ([[nx, ny], [xlo, xhi], [ylo, yhi], [zlo, zhi]]
        headers = [[ncol, nrow], [xmin, xmax], [ymin, ymax], [zmin, zmax]]
        for row in headers:
            file.write('{:g} {:g}\n'.format(*row))  # "*row" unpacks the list : format()

        # Data
        for row in values:
            frmt = '{:g} '*(ncol-1) + '{:g}\n'  # format val1 val2 ... valncol\n
            file.write(frmt.format(*row))  # "*row" unpacks the list : format()

    error = 0
    return error


GRID_WRITER_MAP = {'surfer7bin' : write_surfer7bin,
               'surfer6bin' : write_surfer6bin,
               'surfer6ascii': write_surfer6ascii
               }

# def write_grd(filename, data, frmt='surfer7bin'):
#     '''
#     Write data to Golden Software Surfer grid format.

#     Parameters
#     ----------
#     filename : str
#         Name of the grid file to read.

#     data : dict
#     Dictionary containing the data.
#     Dictionary keys are:
#         * 'nrow' : number of rows in the grid.
#         * 'ncol' : number of columns in the grid.
#         * 'xll' [optional] : coordinate of the lower left corner of the grid.
#             Mandatory if 'xmin' and 'xmax' are not provided.
#         * 'yll' [optional] : coordinate of the lower left corner of the grid.
#             Mandatory if 'ymin' and 'ymax' are not provided.
#         * 'xmin' [optional] : minimum X value of the grid.
#             Mandatory if 'xll' is not provided.
#         * 'xmax' [optional] : maximum X value of the grid.
#             Mandatory if 'xll' is not provided.
#         * 'ymin' [optional] : minimum Y value of the grid.
#             Mandatory if 'yll' is not provided.
#         * 'ymax' [optional] : maximum Y value of the grid.
#             Mandatory if 'yll' is not provided.
#         * 'xsize' [optional] : spacing between adjacent nodes in the X direction (between columns).
#             Mandatory if 'xmax' is not provided.
#         * 'ysize' [optional] : spacing between adjacent nodes in the Y direction (between rows).
#             Mandatory if 'ymax' is not provided.
#         * 'zmin' : minimum Z value of the grid.
#         * 'zmax' : maximum Z value of the grid.
#         * 'values' : Z value of the grid.
#         * 'blankvalue' : blank value (nodes are blanked if greater or equal to this value).

#     frmt : str {'surfer7bin', 'surfer6bin', 'surfer6ascii'}, optional
#         Format of the grid file to write.

#     Returns
#     -------
#     error : int
#         0 if no error.

#     '''

#     if frmt not in VALID_GRID_FORMAT:
#         raise TypeError(('Invalid grid format for Surfer .grd file. '
#                          'Must be %s')
#                         %(', '.join(VALID_GRID_FORMAT[:-1]) + ' or ' + VALID_GRID_FORMAT[-1]))

#     if frmt in VALID_GRID_FORMAT:
#         writefun = GRID_WRITER_MAP[frmt]
#         error = writefun(filename, data)

#     return error


# --- Registered High-Level Functions ---
@register_reader('surfer', 'Golden Software Surfer Grid (*.grd)')
def read_grd(filename: str, fmt: str = None, **kwargs) -> dict:
    """
    Reads data from a Golden Software Surfer grid file.

    This function acts as a dispatcher, calling the appropriate low-level
    reader based on the specified format. It returns a pandas DataFrame,
    which is then converted by the IOMixin.

    Parameters
    ----------
    filename : str
        The path to the .grd file.
    fmt : {None, 'surfer7bin', 'surfer6bin', 'surfer6ascii'}, optional
            The specific Surfer grid format. Defaults to None.

    Returns
    -------
    dict or None
        A standardized dictionary containing the grid data on success,
        or ``None`` on failure.

    Raises
    ------
    TypeError
        If the detected or specified format is invalid.
    """

    # Auto-detect the format if not explicitly provided
    if fmt is None:
        fmt = read_format_from_file(filename)

    # Validate the format
    if fmt not in VALID_GRID_FORMAT:
        raise TypeError(f"Invalid Surfer format '{fmt}'. Must be one of {VALID_GRID_FORMAT}")

    # Get the appropriate low-level reader function from the map
    reader_function = GRID_READER_MAP.get(fmt)

    # Call the reader and return its result
    if reader_function:
        try:
            # The low-level function now returns the standardized dictionary
            return reader_function(filename)
        except Exception as e:
            print(f"An error occurred while reading '{filename}' as '{fmt}': {e}")
            return None
    else:
        print(f"Reading for the format '{fmt}' is not yet implemented.")
        return None

@register_writer('surfer', 'Golden Software Surfer Grid (*.grd)')
def write_grd(survey, filename: str, **kwargs):
    """Writes Survey grid data to a Golden Software Surfer grid file.

    This function acts as a dispatcher, calling the appropriate low-level
    writer function based on the specified format.

    Parameters
    ----------
    survey : Survey
        The Survey object containing the gridded data to export.
    filename : str
        The path for the output .grd file.
    **kwargs
        fmt : {'surfer7bin', 'surfer6bin', 'surfer6ascii'}, optional
            The specific Surfer grid format. Defaults to 'surfer7bin'.

    Returns
    -------
    int or None
        Returns 0 on success, None on failure.
    """

    if survey.grid is None:
        raise ValueError("No gridded data available to export.")

    fmt = kwargs.get('fmt', 'surfer7bin').lower()
    if fmt not in VALID_GRID_FORMAT:
        raise TypeError(f"Invalid Surfer format '{fmt}'. Must be one of {VALID_GRID_FORMAT}")

    # Get the data dictionary from the Survey object
    try:
        data_for_writer = survey.grid_to_dict()
    except Exception as e:
        print(f"Could not prepare data for writing: {e}")
        return None

    # Find the correct low-level writer function
    writer_function = GRID_WRITER_MAP.get(fmt)
    
    # Call the low-level writer and handle exceptions
    if writer_function:
        try:
            # Call the writer (e.g., write_surfer7bin) with the filename and data dict
            error_code = writer_function(filename, data_for_writer)
            return error_code
        except Exception as e:
            print(f"An error occurred while writing '{filename}' as '{fmt}': {e}")
            return None
    else:
        print(f"Writing for the format '{fmt}' is not yet implemented.")
        return None

