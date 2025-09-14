# -*- coding: utf-8 -*-
'''
    geophpy.visualization.utils
    -----------------

    Module for the management of Geographic Positioning Sets.

    This module provides a number of tools, including the
    :class:`~geophpy.geoposset.GeoPosSet` class,
    to dealing with Geographic Positioning Sets (or Ground Control Points).

    :copyright: Copyright 2014-2019 Lionel Darras, Philippe Marty, Quentin Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

    Conversion
    ----------
    - `utm_to_wgs84` -- Convert UTM to lat, long coordinates.
    - `wgs84_to_utm` -- Convert lat, long to UTM coordinates.

    Saving
    ------
    - `save` -- Save GCPs to an ascii file.
    - `to_kml` -- Save GCPs to a kml file.

'''

from __future__ import absolute_import

import os
# import glob  # to manage severals files thanks to "*." extension
import csv

import shapefile   # pyshp package for reading and writing shapefiles
import utm # utm and wgs84 conversion

import matplotlib.pyplot as plt

import numpy as np

import geophpy.georeferencing.kml as kml
import geophpy.core.io as iofiles
import geophpy.core.utils as utils





#-----------------------------------------------------------------------------#
# Module's parameters definition                                              #
#-----------------------------------------------------------------------------#

###
##
# ... TODO ...
# use pyproj to manage other coordinate system and re-projection ?
##
###

# Display parameters
POINT_PARAMS = 'bo'

# available reference systems
REFSYSTEM_LIST = ['UTM', 'WGS84']

# UTM system limits
UTM_MINLETTER = 'E'
UTM_MINNUMBER = 1
UTM_MAXLETTER = 'X'
UTM_MAXNUMBER = 60

# file types read
FILETYPE_LIST = ['ascii', 'shapefile']  # list of file types available

format_chooser = {
    ".dat" : "ascii",
    ".txt" : "ascii",
    ".csv" : "ascii",
    ".shp" : "shapefile"
    }

#-----------------------------------------------------------------------------#
# Geographic Positioning Set Object                                           #
#-----------------------------------------------------------------------------#
class GeoPosSet:
    ''' Class to manage geographic positioning set.

    Attributes
    ----------
    refsystem : :obj:`str` or :obj:`None`, opt
        Geographic reference system ('UTM', 'WGS84', ...).

    utm_zoneletter : str, opt
        Utm zone letter for 'UTM' `refsystem` (E -> X).

    utm_zonenumber : int, opt
        Utm zone number for 'UTM' `refsystem` (1 -> 60).

    points_list : :obj:`list` of :obj:`scalar`, opt
        List of Ground Control Points:
        >>> [[lat1, lon1, x1, y1], [lat2, lon2, x2, y2], ...]

    '''

    ###
    ##
    ### Change arguments order to more logic
    # points_list, refsystem, utm_letter and utm_number
    ##
    ###
    def __init__(self, refsystem=None, utm_letter=None, utm_number=None, points_list=None):

        self.refsystem = refsystem
        self.utm_letter = utm_letter
        self.utm_number = utm_number
        self.points_list = points_list

    def __str__(self):

        return '%s:\n Geographic system: %s\n GCPs: %s points' % (self.__class__.__name__,
                                                                  self.refsystem,
                                                                  len(self.points_list))


##    ###
##    ##
##    # To Be Implemented
##    @classmethod
##    def from_ascii_file(cls, filenameslist):
##        pass
##
##    @classmethod
##    def from_shape_file(cls, filenameslist):
##        pass
##    #
##    ##
##    ###

    @classmethod
    def from_ascii_file(cls, filenames, delimiter=None):
        ''' Build a :class:`geophpy.geoposdet.GeoPosSet` object from one or several ascii files.

        Parameters
        ----------
        filenames : :obj:`str` or :obj:`list` of :obj:`str`
            Names of files to be read.

        delimiter : :obj:`str` or ``None``
            The ASCII file delimiter. If ``None`` (by default),
            the delimiter will be sniffed from the file itself.

        Returns
        -------
        :class:`~geophpy.geoposset.GeoPosSet` object (possibly empty).

        succes : ``bool``
            ``True`` if build was successful, ``False`` otherwise.

        '''

        # Attributes initialization
        refsystem = None
        utm_letter = None
        utm_number = None
        points_list = []
        point_num = 1
        success = True

        # Type checking & extending file names list
        ## takes of cases with a star operator : ['GPS_ex2.dat', '*.csv']
        full_filelist = iofiles.extent_file_list(filenames)

        # Reading files
        for filename in full_filelist:

            # If file exists
            if os.path.isfile(filename) is True:

                # Sniffing delimiter
                ## passing 1st line (=refsystem)
                if delimiter is None:
                    delimiter = iofiles.sniff_delimiter(filename, skiplines=1)

                # Reading csv file
                with open(filename, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=delimiter)

                    # Dealing with the 1st line (refsystem) ####################

                    # refsytem + info (could be ['UTM', '32', 'N'])
                    fullrefsystem = reader.__next__()
                    # only refsytem
                    ref_system = fullrefsystem[0]

                    # UTM
                    if ref_system.lower() == 'UTM'.lower():
                        refsystem = ref_system

                        try:
                            # Valid UTM zone letter
                            utm_zoneletter = fullrefsystem[1]
                            if isvalid_utm_letter(utm_zoneletter):
                                utm_letter = utm_zoneletter.upper()

                            # Valid UTM zone number
                            utm_zonenumber = fullrefsystem[2]
                            if isvalid_utm_number(utm_zonenumber):
                                utm_number = utm_zonenumber

                        except:
                            pass

                    # WGS84
                    elif ref_system.lower() == 'WGS84'.lower():
                        refsystem = ref_system

                    # Unkwnown system or just points
                    else:

                        # Unkwnown refsystem
                        if not len(fullrefsystem) > 1:
                            refsystem = ref_system

                        # Just points
                        else:
                            row = fullrefsystem
                            try:
                                x = float(row[3])
                            except:
                                x = None
                            try:
                                y = float(row[4])
                            except:
                                y = None
                            try:
                                ###
                                ##
                                # Why ask for point num in file format if not used here ????
                                ##
                                ###
                                points_list.append([point_num, float(row[1]), float(row[2]), x, y])
                                point_num += 1
                            except IOError:
                                print("File Read Error (", filename, ": ", row, ")")
                                success = False

                    # Reading data points ######################################
                    # num, east, north, y, x
                    for row in reader:
                        try:
                            x = float(row[3])
                        except:
                            x = None
                        try:
                            y = float(row[4])
                        except:
                            y = None
                        try:
                            points_list.append([point_num, float(row[1]), float(row[2]), x, y])
                            point_num += 1
                        except IOError:
                            print("File Read Error (", filename, ": ", row, ")")
                            success = False

        # No points read
        if not points_list:  # empty sequence are False
            points_list = None
            success = False

        # Conversion to np.array
        points_list.append([-1, 0, 0, None, None])  # trick to convert np.array in good format
        points_list = np.array(points_list)
        points_list = np.delete(points_list, -1, 0)  # to delete last row added

        return success, cls(refsystem, utm_letter, utm_number, points_list)

    @classmethod
    def from_shape_file(cls, filenames):
        ''' Build a :class:`geophpy.geoposdet.GeoPosSet` object from one or several shape files.

        Parameters
        ----------
        filenames : :obj:`str` or :obj:`list` of :obj:`str`
            Names of files to be read.

        delimiter : :obj:`str` or ``None``
            The ASCII file delimiter. If ``None`` (by default),
            the delimiter will be sniffed from the file itself.

        Returns
        -------
        :class:`~geophpy.geoposset.GeoPosSet` object (possibly empty).

        succes : bool
            ``True`` if build was successful, ``False`` otherwise.

        '''

        # Attributes initialization
        refsystem = None
        utm_letter = None
        utm_number = None
        points_list = []
        point_num = 0
        success = True

        # Type checking & extending file names list
        ## takes of cases with a star operator : ['GPS_ex2.dat', '*.csv']
        full_filelist = iofiles.extent_file_list(filenames)

        # Reading shapefile
        for filename in full_filelist:
            try:
                sf = shapefile.Reader(filename)
                shapes = sf.shapes()

                # for each shape in the file
                for shape in shapes:
                    valid_point = (shape.shapeType == shapefile.POINT
                                   or shape.shapeType == shapefile.POINTZ
                                   or shape.shapeType == shapefile.POINTM)

                    if valid_point:
                        points_list.append([point_num, shape.points[0][0],
                                            shape.points[0][1],
                                            None,
                                            None
                                            ])
                        point_num = point_num + 1

            except IOError:
                print("File Read Error (", filename, ")")
                success = False
                break

        # Conversion to np.array
        points_list.append([-1, 0, 0, None, None])  # trick to convert np.array in good format
        points_list = np.array(points_list)
        points_list = np.delete(points_list, -1, 0)  # to delete last row added

        return success, cls(refsystem, utm_letter, utm_number, points_list)

    @classmethod
    def from_file(cls, filenames, filetype=None):
        '''Build a :class:`~geophpy.geoposdet.GeoPosSet` object from one or several files.

        Parameters
        ----------
        filenames : ``str`` or ``list`` of ``str``
            Names of files to be read.

        filetype : {'ascii', 'shapefile', ``None``}
            Type of the files to read. Id ``None`` (default), the file type
            will be determined from the file extension.

        Returns
        -------
        :class:`~geophpy.geoposset.GeoPosSet` object (possibly empty).

        succes : bool
            ``True`` if build was successful, ``False`` otherwise.

        '''

        read_func_chooser = {'ascii' :cls.from_ascii_file,
                             'shape' : cls.from_shape_file}

        # type checking
        if isinstance(filenames, str):
            filenames = [filenames]

        # Type from file extension
        if filetype is None:
            file_ext = os.path.splitext(filenames[0])[1]
            filetype = format_chooser.get(file_ext, 'ascii')

        return read_func_chooser[filetype](filenames)


    def to_ascii(self, filename, delimiter=';'):
        ''' Save :class:`~geophpy.geoposset.GeoPosSet` points list to an ascii file.

        Parameters
        ----------
        filename: ``str``
            Name of the ascii file to save in.

        delimiter: ``str``, opt
            Delimiter to use.

        Returns
        -------
        success: ``bool``
            ``True`` if a file was saved.

        '''

        # Writing reference system
        if self.refsystem is not None:

            # UTM
            if self.refsystem.upper() == 'UTM':
                # UTM;N;31
                try:
                    # Valid UTM zone letter
                    if isvalid_utm_letter(self.utm_zoneletter):
                        utm_letter = self.utm_zoneletter.upper()

                    # Valid UTM zone number
                    if isvalid_utm_number(self.utm_zonenumber):
                        utm_number = self.utm_zonenumber

                except:
                    utm_letter = ''
                    utm_number = ''

                refsystem = delimiter.join('UTM' + utm_letter + utm_number)

            # Other
            else:
                refsystem = self.refsystem.upper()

        # No refsystem
        else:
            refsystem = ''

        # Writting GCPs
        arr = self.points_list
        np.savetxt(filename, arr, fmt='%g', delimiter=delimiter, header=refsystem, comments='')

        success = True

##        try:
##            with open(filename, 'w', newline='') as csvfile:
##                writer = csv.writer(csvfile, delimiter=delimiter)
##
##                # Reference system (1st row)
##                if self.refsystem is not None:
##                    # UTM
##                    if 'UTM'.lower() == self.refsystem.lower():
##                        # UTM;N;31
##                        refsystem = [self.refsystem.upper(),
##                                     self.utm_letter.upper(),
##                                     self.utm_number]
##                    # Other
##                    else:
##                        refsystem = [self.refsystem.upper()]
##                        print(refsystem)
##
##                    writer.writerow(refsystem)
##
##                # No refsystem
##                else:
##                    pass
##                # Points list
##                writer.writerows(self.points_list)
##
##        except:
##            success = False

##        ResultFile = csv.writer(open(filename,"w", newline=''), delimiter=';')
##        # writes the first line of the file with reference system ('UTM', 'WGS84', ...)
##        fullrefsystem = [self.refsystem]
##        # if 'UTM' ref system, writes the second and third lines with UTM zone letter and number
##        if (self.refsystem == 'UTM'):
##            fullrefsystem = [self.refsystem, self.utm_letter, self.utm_number]
##        else :
##            fullrefsystem = [self.refsystem]
##
##        ResultFile.writerow(fullrefsystem)
##        # writes points list in the file.
##        ResultFile.writerows(self.points_list)

        return success

    def plot(self, filename=None, dpi=None, transparent=False,
             i_xmin=None, i_xmax=None, i_ymin=None, i_ymax=None, long_label=False):
        ''' Display GCPs.

        Plots the GCPs using the point number as label.
        To save the plot, use the ``picturefilename`` option.

        Parameters
        ----------
        filename : ``str`` or ``None``
            Name of the file to save the picture.
            If ``None``, no picture is saved.

        dpi : ``int``
            'dot per inch' definition for the picture file if filename is not None.

        transparent : ``bool``
            if True, picture display points not plotted as transparents

        i_xmin : x minimal value to display, None by default

        i_xmax : x maximal value to display, None by default

        i_ymin : y minimal value to display, None by default

        i_ymax : y maximal value to display, None by default

        long_label : bool
            Flag to display both point number and local coordinates.

        Returns
        -------
        success: True if no error

        fig: ``Figure`` object

        '''

        fig = plt.figure()                                  # creation of the empty figure

        for point in self.points_list:
            plt.plot(point[1], point[2], POINT_PARAMS)
            ###
            # Complete label plot
            if long_label:
                dx, dy = 3, 0
                label_box_opts = dict(fc='white', ec='none', alpha=0.5)
                if point.size == 5:
                    strg = '%s (%s, %s)' % (point[0], point[3], point[4])
                else:
                    strg = '%s (%s, %s)' % (point[0], 'x', 'y')
                plt.text(point[1] + dx, point[2] + dy, strg, bbox=label_box_opts)

            else:
                plt.text(point[1] + dx, point[2] + dy, point[0], bbox=label_box_opts)

        xmin = min(self.points_list.T[0])
        xmax = max(self.points_list.T[0])
        ymin = min(self.points_list.T[1])
        ymax = max(self.points_list.T[1])

        success = True
        if success:
            # for each x or y limit not configured in input, initialisation of the value with the limits of points displayed
            nolimits = True
            if i_xmin is None:
                i_xmin = xmin
            else:
                nolimits = False
            if i_xmax is None:
                i_xmax = xmax
            else:
                nolimits = False
            if i_ymin is None:
                i_ymin = ymin
            else:
                nolimits = False
            if i_ymax is None:
                i_ymax = ymax
            else:
                nolimits = False

            if not nolimits:   # sets the axis limits
                xmin, xmax, ymin, ymax = plt.axis([i_xmin, i_xmax, i_ymin, i_ymax])
            else:                                      # gets the axis limits
                xmin, xmax, ymin, ymax = plt.axis()

            # to have the same scale in X and Y axis
            dy = ymax - ymin
            dx = xmax - xmin
            if dy > dx:
                xmax = xmin + dy
            elif dx > dy:
                ymax = ymin + dx
            plt.axis([xmin, xmax, ymin, ymax])

            ax = plt.gca()
            ax.ticklabel_format(useOffset=False)

            # Forcing equal x and Y
            ax.set_aspect('equal')

            if filename is not None:
                plt.savefig(filename, dpi=dpi, transparent=transparent)

        return success, fig

    def to_kml(self, filename):
        """ Save :class:`~geophpy.geoposset.GeoPosSet` points list to a kml file.

        Parameters
        ----------
        filename: ``str``
            Name for the kml file to save in.

        Returns
        -------
        success: ``bool``
            ``True`` if a file was saved.

        """

        success = True # by default
        kml.geoposset_to_kmlfile(self.points_list, filename)

        return success
    
    def GCPs(self):
        ''' Returns the list of ground control Points. '''

        return  self.points_list

#-----------------------------------------------------------------------------#
# General Geographic Positioning Set Environments functions                   #
#-----------------------------------------------------------------------------#
def refsys_getlist():
    """ List of available geographic reference system. """

    return REFSYSTEM_LIST


def filetype_getlist():
    """ List read file types, 'ascii', 'shapefile', ... """

    return FILETYPE_LIST

###
##
# ... MOVE TO ... Should we move those general routines to geopositionning.general
##
###
def utm_to_wgs84(easting, northing, zonenumber, zoneletter):
    """ Conversion from UTM to WGS84 coordinates (lat, lon).

    Parameters
    ----------
    easting : ``scalar``
        Easting UTM coordinate.

    northing : ``scalar``
        Northing UTM coordinate.

    zonenumber : ``int``
        UTM zone number.

    zoneletter: ``str``
        UTM zone letter.

    Returns
    -------
    latitude : ``scalar``
        WGS84 latitude coordinate.

    longitude : ``scalar``
        WGS84 longitude coordinate.

    """

    ### Making it work on list too
    easting = utils.make_sequence(easting)
    northing = utils.make_sequence(northing)

    if len(easting) != len(northing):
        print('Easting and Norting should be same size but'
              'but (%s and %s) encountered' %(len(easting), len(northing))
              )
        return None

    rank = len(easting)
    zonenumber = utils.make_normalize_sequence(zonenumber, rank)
    zoneletter = utils.make_normalize_sequence(zoneletter, rank)

    latitude, longitude = [], []
    for i in range(rank):
        lat, long = utm.to_latlon(easting[i],
                                  northing[i],
                                  zonenumber[i],
                                  zoneletter[i])
        latitude.append(lat)
        longitude.append(long)

    # Single value
    if len(latitude) == 1:

        return latitude[0], longitude[0]

    return latitude, longitude
    ###
#    return utm.to_latlon(easting, northing, zonenumber, zoneletter)


def wgs84_to_utm(latitude, longitude):
    """ Conversion from WGS84 to UTM coordinates.

    works on list

    Parameters
    ----------
    latitude : ``scalar``
        WGS84 latitude coordinate.

    longitude : ``scalar``
        WGS84 longitude coordinate.

    Returns
    -------
    easting : ``scalar``
        Easting UTM coordinate.

    northing : ``scalar``
        Northing UTM coordinate.

    zonenumber : ``int``
        UTM zone number.

    zoneletter: ``str``
        UTM zone letter.

    """

    ### Making it work on list too
    latitude = utils.make_sequence(latitude)
    longitude = utils.make_sequence(longitude)

    if len(latitude) != len(longitude):
        print('Latitude and Longitude should be same size but'
              'but (%s and %s) encountered' %(len(latitude), len(longitude))
              )
        return None

    easting, northing, number, letter = [], [], [], []
    for i, lat in enumerate(latitude):
        #lat = latitude[i]
        long = longitude[i]
        east, north, num, let = utm.from_latlon(lat, long)
        easting.append(east)
        northing.append(north)
        number.append(num)
        letter.append(let)

    # Single value
    if len(easting) == 1:
        return easting[0], northing[0], number[0], letter[0]

    return easting, northing, number, letter
    ###

    #return utm.from_latlon(latitude, longitude)


def utm_getzonelimits():
    """ UTM coordinates system min and max numbers and letters.

    Returns
    -------

    min_number : ``int``
        Minimal number of the UTM zone (1).

    min_letter : ``str``
        Minimal letter of the UTM zone (E).

    max_number : ``int``
        Maximal number of the UTM zone (60).

    max letter : ``str``
        Maximal letter of the UTM zone (X).

    """

    return UTM_MINNUMBER, UTM_MINLETTER, UTM_MAXNUMBER, UTM_MAXLETTER


def isvalid_utm_letter(strg):
    ''' Check validity of an UTM zone letter. '''

    if not isinstance(strg, str):
        strg = str(strg)

    valid = (strg.isalpha()
             and strg >= UTM_MINLETTER
             and strg <= UTM_MAXLETTER
             )

    return valid


def isvalid_utm_number(strg):
    ''' Check validity of an UTM zone number. '''

    if not isinstance(strg, str):
        strg = str(strg)

    valid = (strg.isdigit()
             and float(strg) >= UTM_MINNUMBER
             and float(strg) <= UTM_MAXNUMBER
             )

    return valid
