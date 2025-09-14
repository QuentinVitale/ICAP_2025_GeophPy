# -*- coding: utf-8 -*-
"""
   geophpy.mag.uxo
   ---------------

   Provides a registered reader for the UXO100 magnetic data format.

   :copyright: Copyright 2021-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.
"""

import os
import numpy as np
import pandas as pd
import logging 
from geophpy.core.registry import register_reader, register_writer
#from .datastructures import MagneticPointData # Import the specialized dataclass

logger = logging.getLogger(__name__)

# Define the standard column layout for a UXO file
UXO_DEFAULT_MAP = {
    'x': 'x', 'y': 'y', 'values': 'values', 'track': 'track',
    'east': 'east', 'north': 'north', 'long': 'long', 'lat': 'lat'
}

@register_reader('uxo', 'UXO 100 Magnetic Data (*.uxo)', default_map=UXO_DEFAULT_MAP)
def read_uxo(filename, encoding=None):
    ''' Read UXO 100 (.uxo) file.

    Parameters
    ----------
    filename : str
        File name.

    encoding : {None, 'utf-8'}
        File encoding.
        Maybe different from 'utf-8' for uxo file containing German character.

    Returns
    -------
    data : dict
        Dictionary containing the data.
        Dictionary keys are :
            * 'x' : local x position in meters.
            * 'y' : local y position in meters.
            * 'z' : local z position in meters.
            * 'values' : data values.
            * 'track' : track number.
            * ()'probe' : probe number.)
            * 'long' [optional] : data longitude in decimal degrees.
            * 'lat' [optional] : data latitude in decimal degrees.
            * 'alt' [optional] : data altitude.
            * 'east' [optional] : data easting in meters.
            * 'north' [optional] : data northing in meters.
            * 'elevation' [optional] : data elevation in meters.

    Notes
    -----

    Notes on UXO100 format

    Header lines starts with [xxx], where xxx is a number.
    Header lines are possibly empty ex:
        [002]project name
        [003]
        [004]
    Header lines might use column separator or not ex:
        [002]project name
        [002] project name
    Header line [041] is decimal separator used in the file.
    Header line [042] is the column separator used in the file.
    Data header line starts with [071]
    Data lines starts with [072]

    '''

    if not os.path.exists(filename):

        raise OSError(2, 'No such file or directory', filename)

    with open(filename, 'r', encoding=encoding) as file:

        # read all lines at once
        lines = file.readlines()

        data = {}
        xyzval = []
        track = []
        probe = []
        lglt = []
        alt = []
        enelev = []

        for line in lines:

            # [041]	Column separator
            if line.startswith('[041]'):

                col_sep = get_column_separator(line)

            # # [042]	Decimal separator
            # if line.startswith('[042]'):
            #     dec_sep = get_decimal_separator(line)

            # [072]	Data line
            if line.startswith('[072]'):

                # Stripping leading separator and splitting line
                data_line = line[5:]
                # Empty x, y values [MXPDA]
                if data_line.startswith(col_sep*3):
                    data_line = data_line.lstrip(col_sep)  # possible additionnal separator
                    data_line = data_line.rstrip()
                    data_line = data_line.split(col_sep)
                    data_line.insert(0,'-999') # adding fake x coordinate
                    data_line.insert(0,'-999') # adding fake y coordinate
                    data_line.insert(0,'0') # adding fake z coordinate

                else:
                    data_line = data_line.lstrip(col_sep)  # possible
                    data_line = data_line.rstrip()
                    data_line = data_line.split(col_sep)
                
                data_line = [it.replace(',', '.') for it in data_line]

                # Conversion of data to numeric type
                xyzval.append([float(it) for it in data_line[0:4]])
                if '-' in data_line[4]: # track-probe num
                    track.append(float(data_line[4].split('-')[0]))
                    probe.append(float(data_line[4].split('-')[1]))
                else:
                    track.append(float(data_line[4]))

                # Geographic coordinates
                if len(data_line) > 5:
                    lglt.append([geo_str_to_dd(it) for it in data_line[5:7]])
                    alt.append(float(data_line[7]))

                # Projected coordinates
                if len(data_line) > 9:
                    enelev.append([float(it) for it in data_line[8:]])

    data['x'] = np.asarray(xyzval)[:, 0]
    data['y'] = np.asarray(xyzval)[:, 1]
    data['z'] = np.asarray(xyzval)[:, 2]
    data['values'] = np.asarray(xyzval)[:, 3]
    data['track'] = np.asarray(track).astype(int)
    data['fields'] = ['x', 'y', 'z', 'values', 'track']

    if len(probe):
        data['probe'] = np.asarray(probe).astype(int)
        data['fields'] = data['fields'] + ['probe']

    if lglt:
        data['long'] = np.asarray(lglt)[:, 0]
        data['lat'] = np.asarray(lglt)[:, 1]
        data['alt'] = np.asarray(alt)
        data['fields'] = data['fields'] + ['long', 'lat', 'alt']

    if enelev:
        data['east'] = np.asarray(enelev)[:, 0]
        data['north'] = np.asarray(enelev)[:, 1]
        data['elevation'] = np.asarray(enelev)[:, 2]
        data['fields'] = data['fields'] + ['east', 'north', 'elevation']
    
    # 'Smart' Coordinate Handling ---
    # Check if georeferenced coordinates are present and valid
    if 'east' in data and 'north' in data and len(data['east']) > 0:
        logger.info("GNSS data found in UXO file. Prioritizing 'east' and 'north' as primary coordinates.")

        # Preserve local coordinates by renaming them
        data['local_x'] = data.pop('x')
        data['local_y'] = data.pop('y')

        # Alias georeferenced coordinates to be the primary 'x' and 'y'
        data['x'] = data['east']
        data['y'] = data['north']

#    return data
    data.pop('fields')  # the 'fields' keys is not the same length as the data a make DataFrame population to crash

    # Convert the dictionary to a DataFrame 
    try:
        # Create the DataFrame from the (potentially modified) dictionary
        df = pd.DataFrame(data)

        # Ensure the primary 'x', 'y', 'values' columns are first for convenience
        core_cols = ['x', 'y', 'values', 'track']
        other_cols = [col for col in df.columns if col not in core_cols]
        ordered_df = df[core_cols + other_cols]

        return ordered_df

    except Exception as e:
        logger.error(f"Failed to create DataFrame from UXO data: {e}")

        return None

@register_writer('uxo', 'UXO 100 Magnetic Data (*.uxo)')
def write_uxo(filename, data, encoding=None):
    ''' Write UXO 100 (.uxo) file.

    Parameters
    ----------
    filename : str
        File name.

    encoding : {None, 'utf-8'}
        File encoding.
        Maybe different from 'utf-8' for uxo file containing German character.

    data : dict
        Dictionary containing the data.
        Dictionary keys are :
            * 'x' : local x position in meters.
            * 'y' : local y position in meters.
            * 'z' : local z position in meters.
            * 'values' : data values.
            * 'track' : track number.
            * 'long' [optional] : data longitude in decimal degrees.
            * 'lat' [optional] : data latitude in decimal degrees.
            * 'alt' [optional] : data altitude.
            * 'east' [optional] : data easting in meters.
            * 'north' [optional] : data northing in meters.
            * 'elevation' [optional] : data elevation in meters.
            * 'name' [optional] : project name.
            * 'unit' [optional] : values unit.
            * 'crs' [optional] : Coordinate system (descriptive string).
        Additional keys will be ignored

    '''

    if not filename.lower().endswith('.uxo'):
        filename = os.path.splitext(filename)[0] + '.uxo'

    with open(filename, 'w', encoding=encoding) as file:

        col_sep = '\t'

        # File header
        file.write(col_sep.join(['[001]',
                                 'File Format',
                                 'UXO100'])
                   + '\n')

        name = data.get('name', None)
        if name is not None:
            file.write(col_sep.join(['[002]',
                                     'Project name',
                                     name])
                       + '\n')
        else:
            file.write(col_sep.join(['[002',
                                     'Project name'])
                       + '\n')

        file.write(col_sep.join(['[003]', 'Service provider']) + '\n')
        file.write(col_sep.join(['[004]', 'Acolyte']) + '\n')
        file.write(col_sep.join(['[005]', 'Creation Date']) + '\n')
        file.write(col_sep.join(['[006]', 'Evaluator']) + '\n')

        file.write(col_sep.join(['[007]', 'Transfer date']) + '\n')
        file.write(col_sep.join(['[008]', 'Software-Name']) + '\n')

        file.write(col_sep.join(['[009]', 'Field name']) + '\n')

        file.write(col_sep.join(['[010]', 'Field type']) + '\n')
        file.write(col_sep.join(['[011]', 'Data source type']) + '\n')
        file.write(col_sep.join(['[012]', 'Firmware/Software']) + '\n')
        file.write(col_sep.join(['[013]', 'DLG-Serial-no.']) + '\n')
        file.write(col_sep.join(['[014]', 'Calibration date']) + '\n')

        file.write(col_sep.join(['[015]', 'Detector type']) + '\n')
        file.write(col_sep.join(['[016]', 'Serial no.']) + '\n')
        file.write(col_sep.join(['[017]', 'Sensor type']) + '\n')
        file.write(col_sep.join(['[018]', 'Serial no.']) + '\n')
        file.write(col_sep.join(['[019]', 'Calibration date']) + '\n')
        file.write(col_sep.join(['[020]', 'Number of sensors']) + '\n')

        if data.get('unit', None) is not None:
            file.write(col_sep.join(['[021]',
                                     'Measurand',
                                     data.get('unit')])
                       + '\n')
        else:
            file.write(col_sep.join(['[021]',
                                     'Measurand'])
                       + '\n')

        if data.get('values', None) is not None:
            min_val = min(data['values'])
            file.write(col_sep.join(['[022]',
                                     'Measuring range min',
                                     '{:.2f}']
                                   ).format(min_val) + '\n')

            max_val = max(data['values'])
            file.write(col_sep.join(['[023]',
                                     'Measuring range max',
                                     '{:.2f}']
                                   ).format(max_val) + '\n')
        else:
            file.write(col_sep.join(['[022]',
                                     'Measuring range min'])
                       + '\n')
            file.write(col_sep.join(['[023]',
                                     'Measuring range max'])
                       + '\n')

        file.write(col_sep.join(['[024]', 'Data gap']) + '\n')
        file.write(col_sep.join(['[025]', 'Positioning']) + '\n')
        file.write(col_sep.join(['[026]', 'Measuring speed']) + '\n')


        if data.get('long', None) is not None and data.get('lat', None) is not None:

            file.write(col_sep.join(['[027]',
                                     'Coordinate system',
                                     'WGS84'])
                       + '\n')

        else:
            file.write(col_sep.join(['[027]',
                                     'Coordinate system'])
                       + '\n')


        if data.get('crs', None) is not None:
            file.write(col_sep.join(['[028]',
                                     'Local coordinates',
                                     data.get('crs')])
                       + '\n')
        else:
            file.write(col_sep.join(['[028]', 'Local coordinates']) + '\n')

        file.write(col_sep.join(['[029]', 'Sensor distance']) + '\n')

        file.write(col_sep.join(['[030]', 'Data point spacing']) + '\n')

        file.write(col_sep.join(['[031]', 'Field comment']) + '\n')

        file.write(col_sep.join(['[041]',
                                 'Column separator',
                                 'tab'])
                   + '\n')

        file.write(col_sep.join(['[042]',
                                 'Decimal separator',
                                 '.'])
                   + '\n')

        file.write(col_sep.join(['[043]', 'Coding']) + '\n')

        unit_str = ''
        if data.get('unit', None) is not None:
            unit_str = ' ' + '[' + data.get('unit') + ']'

        file.write(col_sep.join(['[070]',
                                 'X [m]',
                                 'Y [m]',
                                 'Z [m]',
                                 'Value' + unit_str,
                                 'Track',
                                 'WGS84-Lon',
                                 'WGS84-Lat',
                                 'WGS84-Alt [m]',
                                 'Local East [m]',
                                 'Local North [m]'])
                   + '\n')

        # Data header
        file.write(col_sep.join(['[071]',
                                 'x.xxx',
                                 'x.xxx',
                                 'x.xxx',
                                 'x.xx',
                                 'xxxxx',
                                 'x°xx.xxxxxxxx',
                                 'x°xx.xxxxxxxx',
                                 'x.xxx',
                                 'x.xxx',
                                 'x.xxx',
                                 'x.xxx'])
                   + '\n')

        # Data
        for i in range(len(data['values'])):
            x = data['x'][i]
            y = data['y'][i]
            z = 0
            value = data['values'][i]
            track = data['track'][i]
            xyzvaltrack_str = col_sep.join(['{:.3f}',
                                            '{:.3f}',
                                            '{:.3f}',
                                            '{:.2f}',
                                            '{:05d}']).format(x, y, z, value, track)
            llalt_str = ''
            if data['long'] is not None:
                long_str = dd_to_geo_str(data['long'][i])
                lat_str = dd_to_geo_str(data['lat'][i])
                alt = 0
                llalt_str = col_sep + col_sep.join([long_str,
                                                    lat_str,
                                                    '{:.3f}']).format(alt)

            enelev = ''
            if data['east'] is not None:
                east = data['east'][i]
                north = data['north'][i]
                #elevation = 0
                enelev = col_sep + col_sep.join(['{:.3f}',
                                                 '{:.3f}',
                                                 '{:.3f}']).format(east, north, alt)
            file.write('[072]'
                       + col_sep
                       + xyzvaltrack_str
                       + llalt_str
                       + enelev
                       + '\n')


def get_decimal_separator(line):
    ''' Returns the decimal separator. '''

    if ',' in line[5:]:
        return ','

    return  '.'


def get_column_separator(line):
    ''' Returns the column separator. '''

    if 'tab' in line[5:] or '\t' in line[5:]:
        return '\t'

    if ';' in line[5:]:
        return ';'

    if ',' in line[5:]:
        return ','

    if '|' in line[5:]:
        return '|'

    if ' ' in line[5:]:
        return ' '

    return '\t'


def geo_str_to_dd(data_string, separator='°'):
    '''Convert geographic coordinate given as a string with the format
    x°xx.xxxxxxxx to decimal degrees.
    '''

    deg, mindec = data_string.split(separator)

    return int(deg) + float(mindec)/60


def dd_to_geo_str(dd, separator='°'):
    ''' Convert geographic coordinate in decimal degrees to  a string with the format
    x°xx.xxxxxxxx.
    '''

    deg, dec = divmod(dd, 1)

    return '{:d}{}{:.8f}'.format(int(deg), separator, dec*60)
