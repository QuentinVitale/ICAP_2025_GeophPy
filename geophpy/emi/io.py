# -*- coding: utf-8 -*-.
'''
    geophpy.emi.io
    ---------------------------

    Gf instruments CMD files input and output management.

    :copyright: Copyright 2019 Quentin Vitale, Lionel Darras, Philippe Marty and contributors, see AUTHORS.
    :license: GNU GPL v3.
    :author: Quentin VITALE
    :license: GNU GPL v3.
    :version: 0.1
    :revision: 2019/05/09

'''

import copy
import numpy as np
import utm

from geophpy.core.reader import RowReader, RowWriter, Data
import geophpy.core.utils as gutils

###
##
# ?? Put all Readers in reader.py ??
##
###
class CMDFileReader(RowReader):
    ''' Class to read .dat files from Gf Instruments CMD Electromagnetometers. 

    Parameters
    ----------
    filename : str
        Name of the CMD .dat to read.

    delimiter : str, optional
        Delimiter to use when parsing the CMD file. 

    '''

    def __init__(self, filename=None, delimiter='\t'):

        super().__init__(filename=filename, delimiter=delimiter)
        self.data = None

        if filename is not None:
            self.read()

    def read(self):
        ''' reading and parsing .dat file. '''

        super().read()
        self.parse()

        return self

    def parse(self):
        ''' parsing .dat file. '''

        # File has been read
        if self.rows:

            # Header line
            ## Removing internal whitespace ('x [m]'->'x[m]') to prevent errors
            ## when comparing with 'x[m]', 'y[m]' etc.
            headers = self.rows[0]
            headers = [s.replace(' ','') for s in headers]      

            # Data line
            self.rows = self.rows[1:]

            # Grid survey
            if 'x[m]' in headers:

                # Checking for actual date & time in the data
                ## Sometimes it is present in the header
                ## but not in the actual data
                if 'Time' in headers and ':' not in self.rows[0]:
                    headers.remove('Time')

                if 'Date' in headers and '/' not in self.rows[0]:
                    headers.remove('Date')

                # Not interpolated Grid survey #############
                ncol = [len(row) for row in self.rows]
                if any(n == 2 for n in ncol):
                    self._parse_grid_survey(headers)

                # Interpolated Grid survey
                else:
                    self._parse_grid_interp_survey(headers)

            # GPS survey in geographic coordinates
            elif 'Latitude' in headers:
                self._parse_gps_lat_survey(headers)

            # GPS survey in projected coordinates
            elif 'Easting' in headers:
                self._parse_gps_east_survey(headers)

        return self

    def _parse_grid_survey(self, headers):
        ''' Parse CMD (not interpolated) continuous grid survey. '''

        # Retrieving data column index
        try:
            iddate, idtime = None, None
            idx = headers.index('x[m]')
            idy = headers.index('y[m]')
            if 'Time' in headers:
                idtime = headers.index('Time')
            if 'Date' in headers:
                iddate = headers.index('Date')
            idc1 = headers.index('Cond.1[mS/m]')
            idc2 = headers.index('Cond.2[mS/m]')
            idc3 = headers.index('Cond.3[mS/m]')
            idi1 = headers.index('Inph.1[ppt]')
            idi2 = headers.index('Inph.2[ppt]')
            idi3 = headers.index('Inph.3[ppt]')

        except:
            iddate, idtime = None, None
            if 'Date' not in headers and 'Time' not in headers:
                idx, idy = 0, 1
                idc1, idc2, idc3 = 2, 4, 6
                idi1, idi2, idi3 = 3, 5, 7

            if 'Date' in headers and 'Time' in headers:
                idx, idy, iddate, idtime = 0, 1, 2, 3
                idc1, idc2, idc3 = 4, 6, 8
                idi1, idi2, idi3 = 5, 7, 9

            if 'Date' in headers and 'Time' not in headers:
                idx, idy, iddate = 0, 1, 2
                idc1, idc2, idc3 = 3, 5, 7
                idi1, idi2, idi3 = 4, 6, 8
            if 'Date' not in headers and 'Time' in headers:
                idx, idy, idtime = 0, 1, 2
                idc1, idc2, idc3 = 3, 5, 7
                idi1, idi2, idi3 = 4, 6, 8

        # Retrieving x, y
        x = np.array([float(row[idx]) for row in self.rows])
        y = np.array([float(row[idy]) for row in self.rows])
        ncol = np.array([len(row) for row in self.rows])

        # Re-arranging start/stop data into x-directed segments
        npts, segments, segment_points = [], [], []
        x0, y0 = x[0], y[0]
        rows = self.rows
        for i, row in enumerate(rows):
            xi, yi = float(row[idx]), float(row[idy])  # current point

            # Data point
            if len(row)>2:

                # Mark point
                if yi!=y0 and segment_points:
                    # Mark point is used as the end of segment
                    segment_points.append(row)  # coordinates + extra values
                    segments.append(segment_points)

                    # And is repeated as the start of a new segment
                    segment_points = []
                    segment_points.append(row)
                    y0 = yi

                # Segment point
                else:
                    # Adding point to the current segment
                    segment_points.append(row)  # coordinates + extra values

            # End of profile point
            else:
                # Updating last point coordinates
                rows[i-1][idx] = xi
                rows[i-1][idy] = yi

                segments.append(segment_points)

                # Initializing a new segment
                segment_points = []
                if i< len(x)-1: # not the last point
                    x0 = x[i+1]
                    y0 = y[i+1]

        # Position interpolation
        seg_interp = self.interp_segment(segments)
        row_interp = []

        # 'Merging' all interpolated segment
        for segment in seg_interp:   
            for point in segment:
                row_interp.append(point)

        # Getting rid of repeated points
        row_interp = gutils.unique_multiplets(row_interp)

        # Variable initialization
        data = Data()
        x, y, z = [], [], []
        date, time = [], []
        cond1, cond2, cond3 = [], [], []
        inph1, inph2, inph3 = [], [], []

        # Retrieving data
        for row in row_interp:
            # coordinates
            x.append(float(row[idx]))  # x[m]
            y.append(float(row[idy]))  # y[m]
            z.append(0)                # z[m]

            # date & time
            if iddate is not None:
                date.append(row[iddate])  # Date [dd/mm/yyyy]
            if idtime is not None:
                time.append(row[idtime])  # Time [hh:mm:ss.dd]

            # data values
            cond1.append(float(row[idc1]))  # Cond.1[mS/m]
            inph1.append(float(row[idi1]))  # Inph.1[ppt]
            cond2.append(float(row[idc2]))  # Cond.2[mS/m]
            inph2.append(float(row[idi2]))  # Inph.2[ppt]
            cond3.append(float(row[idc3]))  # Cond.3[mS/m]
            inph3.append(float(row[idi3]))  # Inph.3[ppt]

        # Storing data
        data.type = 'GRID_Survey'
        data.x = np.array(x)
        data.y = np.array(y)
        data.z = np.array(z)
        data.date = np.array(date)
        data.time = np.array(time)
        data.cond1 = np.array(cond1)
        data.cond2 = np.array(cond2)
        data.cond3 = np.array(cond3)
        data.inph1 = np.array(inph1)
        data.inph2 = np.array(inph2)
        data.inph3 = np.array(inph3)

        data.Nval = data.x.size
        self.data = data

        return self
    ###
    ##
    # Put at a better place this staic methods
    # Maybe in a more general class that handle grid surveys.
    ##
    ###
    @staticmethod
    def interp_segment(segments, decimals=2):
        ''' Interpolate coordinate from a Start/Stop style list. '''

        segment_interp = copy.deepcopy(segments)  # to not alter original
        for segment in segment_interp:
            npts = len(segment) 

            x0 = float(segment[0][0])
            y0 = float(segment[0][1])

            xend = float(segment[-1][0])
            yend  = float(segment[-1][1])

            x = np.around(np.linspace(x0, xend, npts), decimals=decimals)
            y = np.around(np.linspace(y0, yend, npts), decimals=decimals)

            for i, point in enumerate(segment):
                point[0] = x[i]
                point[1] = y[i]

        return segment_interp

    def _parse_grid_interp_survey(self, headers):
        ''' Parse CMD interpolated continuous grid survey. '''
        
        # Variable initialization
        data = Data()
        x, y, z = [], [], []
        date, time = [], []
        cond1, cond2, cond3 = [], [], []
        inph1, inph2, inph3 = [], [], []

        # Retrieving data index
        try:
            iddate, idtime = None, None
            idx = headers.index('x[m]')
            idy = headers.index('y[m]')
            if 'Time' in headers:
                idtime = headers.index('Time')
            if 'Date' in headers:
                iddate = headers.index('Date')
            idc1 = headers.index('Cond.1[mS/m]')
            idc2 = headers.index('Cond.2[mS/m]')
            idc3 = headers.index('Cond.3[mS/m]')
            idi1 = headers.index('Inph.1[ppt]')
            idi2 = headers.index('Inph.2[ppt]')
            idi3 = headers.index('Inph.3[ppt]')

        except:
            iddate, idtime = None, None
            if 'Date' not in headers and 'Time' not in headers:
                idx, idy = 0, 1
                idc1, idc2, idc3 = 2, 4, 6
                idi1, idi2, idi3 = 3, 5, 7

            if 'Date' in headers and 'Time' in headers:
                idx, idy, iddate, idtime = 0, 1, 2, 3
                idc1, idc2, idc3 = 4, 6, 8
                idi1, idi2, idi3 = 5, 7, 9

            if 'Date' in headers and 'Time' not in headers:
                idx, idy, iddate = 0, 1, 2
                idc1, idc2, idc3 = 3, 5, 7
                idi1, idi2, idi3 = 4, 6, 8

            if 'Date' not in headers and 'Time' in headers:
                idx, idy, idtime = 0, 1, 2
                idc1, idc2, idc3 = 3, 5, 7
                idi1, idi2, idi3 = 4, 6, 8

        # Retrieving data
        for row in self.rows:
            # coordinates
            x.append(float(row[idx]))  # x[m]
            y.append(float(row[idy]))  # y[m]
            z.append(0)                # z[m]

            # date & time
            if iddate is not None:
                date.append(row[iddate])  # Date [dd/mm/yyyy]
            if idtime is not None:
                time.append(row[idtime])  # Time [hh:mm:ss.dd]

            # data values
            cond1.append(float(row[idc1]))  # Cond.1[mS/m]
            inph1.append(float(row[idi1]))  # Inph.1[ppt]
            cond2.append(float(row[idc2]))  # Cond.2[mS/m]
            inph2.append(float(row[idi2]))  # Inph.2[ppt]
            cond3.append(float(row[idc3]))  # Cond.3[mS/m]
            inph3.append(float(row[idi3]))  # Inph.3[ppt]

        # Storing data
        data.type = 'GRID_Survey'
        data.x = np.array(x)
        data.y = np.array(y)
        data.z = np.array(z)
        data.date = np.array(date)
        data.time = np.array(time)
        data.cond1 = np.array(cond1)
        data.cond2 = np.array(cond2)
        data.cond3 = np.array(cond3)
        data.inph1 = np.array(inph1)
        data.inph2 = np.array(inph2)
        data.inph3 = np.array(inph3)

        data.Nval = data.x.size
        self.data = data

        return self
        
    def _parse_gps_lat_survey(self, headers):
        ''' Parse CMD GPS survey in geographic coordinates. '''

        # Variable initialization
        data = Data()
        long, lat, time = [], [], []
        cond1, cond2, cond3 = [], [], []
        inph1, inph2, inph3 = [], [], []
        x, y, z, zone, letter = [], [], [], [], []
        
        # Retrieving data index
        try:
            idlong = headers.index('Longitude')
            idlat = headers.index('Latitude')
            idalt = headers.index('Altitude')
            
            idtime = headers.index('Time')
            idc1 = headers.index('Cond.1[mS/m]')
            idc2 = headers.index('Cond.2[mS/m]')
            idc3 = headers.index('Cond.3[mS/m]')
            idi1 = headers.index('Inph.1[ppt]')
            idi2 = headers.index('Inph.2[ppt]')
            idi3 = headers.index('Inph.3[ppt]')

        except:
            idlat, idlong, idalt, idtime = 0, 1, 2, 3
            idc1, idc2, idc3 = 3, 5, 7
            idi1, idi2, idi3 = 4, 6, 8

        # Retrieving data
        for row in self.rows:
            # coordinates
            long.append(float(row[idlong]))  # Longitude
            lat.append(float(row[idlat]))  # Latitude
            z.append(float(row[idalt]))  # Altitude

            coordutm = utm.from_latlon(lat[-1], long[-1])
            x.append(coordutm[0])  # Easting
            y.append(coordutm[1])  # Northing
            zone.append(coordutm[2])  # UTM zone number
            letter.append(coordutm[3])  # UTM zone letter

            # time
            time.append(row[idtime])  # Time [hh:mm:ss.dd]

            # data values
            cond1.append(float(row[idc1]))  # Cond.1[mS/m]
            inph1.append(float(row[idi1]))  # Inph.1[ppt]
            cond2.append(float(row[idc2]))  # Cond.2[mS/m]
            inph2.append(float(row[idi2]))  # Inph.2[ppt]
            cond3.append(float(row[idc3]))  # Cond.3[mS/m]
            inph3.append(float(row[idi3]))  # Inph.3[ppt]

        # Storing data
        data.type = 'GPS_Survey'
        data.long = np.array(long)
        data.lat = np.array(lat)
        data.x = np.array(x)
        data.y = np.array(y)
        data.z = np.array(z)
        data.zone = np.array(zone)
        data.letter = np.array(letter)
        data.time = np.array(time)
        data.cond1 = np.array(cond1)
        data.cond2 = np.array(cond2)
        data.cond3 = np.array(cond3)
        data.inph1 = np.array(inph1)
        data.inph2 = np.array(inph2)
        data.inph3 = np.array(inph3)

        data.Nval = data.lat.size
        self.data = data

        return self

    def _parse_gps_east_survey(self, headers):
        ''' Parse CMD GPS survey in projected coordinates. '''

        # Variable initialization
        data = Data()
        east, north, z, time = [], [], [], []
        cond1, cond2, cond3 = [], [], []
        inph1, inph2, inph3 = [], [], []

        # Retrieving data index
        try:

            ideast = headers.index('Easting')
            idnorth = headers.index('Northing')
            idalt = headers.index('Altitude')
            idtime = headers.index('Time')
            idc1 = headers.index('Cond.1[mS/m]')
            idc2 = headers.index('Cond.2[mS/m]')
            idc3 = headers.index('Cond.3[mS/m]')
            idi1 = headers.index('Inph.1[ppt]')
            idi2 = headers.index('Inph.2[ppt]')
            idi3 = headers.index('Inph.3[ppt]')

        except:
            ideast, idnorth, idalt, idtime = 0, 1, 2, 3
            idc1, idc2, idc3 = 3, 5, 7
            idi1, idi2, idi3 = 4, 6, 8

        # Retrieving data
        for row in self.rows:
            # coordinates
            east.append(float(row[ideast]))  # Easting
            north.append(float(row[idnorth]))  # Northing
            z.append(float(row[idalt]))  # Altitude

            # time
            time.append(row[idtime])  # Time [hh:mm:ss.dd]

            # data values
            cond1.append(float(row[idc1]))  # Cond.1[mS/m]
            inph1.append(float(row[idi1]))  # Inph.1[ppt]
            cond2.append(float(row[idc2]))  # Cond.2[mS/m]
            inph2.append(float(row[idi2]))  # Inph.2[ppt]
            cond3.append(float(row[idc3]))  # Cond.3[mS/m]
            inph3.append(float(row[idi3]))  # Inph.3[ppt]

        # Storing data
        data.type = 'GPS_Survey'
        data.east = np.array(east)
        data.north = np.array(north)
        data.z = np.array(z)
        data.time = np.array(time)
        data.cond1 = np.array(cond1)
        data.cond2 = np.array(cond2)
        data.cond3 = np.array(cond3)
        data.inph1 = np.array(inph1)
        data.inph2 = np.array(inph2)
        data.inph3 = np.array(inph3)

        data.Nval = data.lat.size
        self.data = data

    def tofile(self, filename, delimiter='\t'):
        ''' Write data into a dsv file.'''

        # GPS Survey
        if self.data.type == 'GPS_Survey':

            # Projected coordinates
            if hasattr(self.data, 'Easting'):
                headers = ['Easting',
                           'Northing',
                           'Altitude',
                           'Time',
                           'Cond.1[mS/m]',
                           'Inph.1[ppt]',
                           'Cond.2[mS/m]',
                           'Inph.2[ppt]',
                           'Cond.3[mS/m]',
                           'Inph.3[ppt]']

                data = np.vstack((self.data.east,
                                  self.data.north,
                                  self.data.z,
                                  self.data.time,
                                  self.data.cond1,
                                  self.data.inph1,
                                  self.data.cond2,
                                  self.data.inph2,
                                  self.data.cond3,
                                  self.data.inph3)).T

            # Geographic coordinates
            elif hasattr(self.data, 'Longitude'):
                headers = ['Longitude',
                           'Latitude',
                           'Altitude',
                           'Time',
                           'Cond.1[mS/m]',
                           'Inph.1[ppt]',
                           'Cond.2[mS/m]',
                           'Inph.2[ppt]',
                           'Cond.3[mS/m]',
                           'Inph.3[ppt]']

                data = np.vstack((self.data.long,
                                  self.data.lat,
                                  self.data.z,
                                  self.data.time,
                                  self.data.cond1,
                                  self.data.inph1,
                                  self.data.cond2,
                                  self.data.inph2,
                                  self.data.cond3,
                                  self.data.inph3)).T

        # Grid Survey
        elif self.data.type == 'GRID_Survey':
            headers = ['x[m]', 'y[m]']
            data = np.vstack((self.data.x, self.data.y))

            if len(self.data.date)>0:
                headers.extent('Date')
                data = np.vstack((data, self.data.date))

            if len(self.data.time)>0:
                headers.extent('Time')
                data = np.vstack((data, self.data.time))

            headers.extend(['Cond.1[mS/m]',
                            'Inph.1[ppt]',
                            'Cond.2[mS/m]',
                            'Inph.2[ppt]',
                            'Cond.3[mS/m]',
                            'Inph.3[ppt]'])

            data = np.vstack((data,
                              self.data.cond1,
                              self.data.inph1,
                              self.data.cond2,
                              self.data.inph2,
                              self.data.cond3,
                              self.data.inph3)).T

        rows = [headers]
        for item in data:
            rows.append(item) 

        FileWriter(filename, rows=rows, delimiter=delimiter)
    
        return self
