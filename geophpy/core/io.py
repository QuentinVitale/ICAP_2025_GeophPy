# -*- coding: utf-8 -*-
'''
    geophpy.core.io
    ---------------------------

    General input and output file management.

    :copyright: Copyright 2014-2020 L. Darras, P. Marty, Q. Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

from operator import itemgetter
import csv              # for csv files treatment
import glob             # for managing severals files thanks to "*." extension
import warnings
import netCDF4 as nc4
import numpy as np
import pandas as pd
import json
import os, time, platform
import logging

#from geophpy.misc.utils import *
#from geophpy.operation.general import *
#import geophpy  # used for software version etc. in nc files saving
#from geophpy.filesmanaging.grd import SurferGrid, gridformat_list, gridtype_list
#from geophpy.filesmanaging.cmd import CMDFileReader
import geophpy.__config__ as CONFIG

#from geophpy.core.grd import read_grd, write_grd, VALID_GRID_FORMAT, VALID_GRID_LIST
#from geophpy.core.uxo import read_uxo, write_uxo

from .datastructures import GridData
from geophpy.core.registry import FILE_READERS, FILE_WRITERS
from geophpy.core.json import JSON_Indent_Encoder
from .grd import VALID_GRID_FORMAT, VALID_GRID_LIST

# This dictionary provides a user-friendly layer to guess a file format
# from its extension. It maps a file extension to a default format name
# that is registered in the FILE_READERS registry.
DEFAULT_EXTENSION_MAP = {
    '.dat': 'ascii',
    '.txt': 'ascii',
    '.csv': 'ascii',
    '.xyz': 'ascii',
    '.uxo': 'uxo',  # should be placed in mag module
    '.grd': 'surfer',
    '.pga': 'wumap',
    '.nc': 'netcdf',
    '.cdf': 'netcdf',
}

# --- IO Classe Mixin ---
logger = logging.getLogger(__name__)
class IOMixin:
    """
    Mixin containing all high-level Input/Output methods for the Survey class.
    """

    # --- Reader Functions ---

    @classmethod
    def from_file(cls, filename, fileformat=None, column_map=None, **kwargs):
        """
        Creates a Survey object by reading data from a file.

        This smart factory method dispatches to the appropriate reader and
        populates the Survey object based on the type of data returned
        (point data as a DataFrame or grid data as a dictionary).

        Parameters
        ----------
        filename : str
            The full path to the data file.
        fileformat : str, optional
            The format of the input file. If None, it's inferred from the extension.
        column_map : dict, optional
            A dictionary mapping PointData attribute names to column numbers
            for point data files. Defaults to {'x': 0, 'y': 1, 'values': 2}.
            Example: {'x': 0, 'y': 1, 'values': 4, 'track': 3}
        **kwargs
            Additional keyword arguments passed to the specific file reader.
            Legacy arguments like `x_colnum` are also handled here for
            backward compatibility.

        Returns
        -------
        Survey or None
            A new Survey object instance on success, or ``None`` on failure.
        """

        # --- 1. Handle Deprecation for legacy _colnum arguments ---
        legacy_args = ['x_colnum', 'y_colnum', 'z_colnum']
        if any(arg in kwargs for arg in legacy_args):
            warnings.warn(
                "'x_colnum', 'y_colnum', and 'z_colnum' are deprecated and will be "
                "removed in a future version. Use the 'column_map' dictionary instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # If column_map isn't provided, create it from the legacy args
            if column_map is None:
                column_map = {
                    'x': kwargs.pop('x_colnum', 0),
                    'y': kwargs.pop('y_colnum', 1),
                    'values': kwargs.pop('z_colnum', 2)
                }

        # --- 2. Determine format and find the appropriate reader ---
        if fileformat is None:
            _, ext = os.path.splitext(filename)
            fileformat = DEFAULT_EXTENSION_MAP.get(ext.lower())
            if fileformat is None:
                logger.info(f"Error: Could not determine format for extension '{ext}'.")
                return None

        reader_details = FILE_READERS.get(fileformat)
        if not reader_details:
            logger.info(f"Error: No reader registered for format '{fileformat}'.")
            return None
        reader_func = reader_details['function']

        # --- 3. Call the reader function to get the data ---
        try:
            # Pass remaining kwargs to the low-level reader
            result = reader_func(filename, **kwargs)
            if result is None:
                return None
        except Exception as e:
            logger.error(f"The reader for format '{fileformat}' failed with an error: {e}")
            return None

        # --- 4. Populate the Survey object based on the result type ---
        base_name = os.path.splitext(os.path.basename(filename))[0]
        instance = cls(name=base_name)

        if isinstance(result, pd.DataFrame):
            # Case 1: Reader returned a DataFrame (for point data)
            logger.info("Reader returned point data. Populating survey.points...")
            if column_map is None:
                reader_default_map = reader_details.get('default_map') # check if the reader provided a default map.
                if reader_default_map:
                    column_map = reader_default_map  # Use the specialist reader's default map
                else:
                    # Fallback to the generic XYZ default for simple ascii files
                    column_map = {'x': 0, 'y': 1, 'values': 2}

            try:
                for attr_name, col_identifier in column_map.items():
                    if hasattr(instance.points, attr_name):
                        # It checks if the identifier is a string (a column name)
                        # or an integer (a column position) and selects data accordingly.
                        if isinstance(col_identifier, str):
                            data_col = result[col_identifier]
                        else:
                            data_col = result.iloc[:, col_identifier]
                            
                        #data_array = pd.to_numeric(result.iloc[:, data_col], errors='coerce').to_numpy()
                        data_array = pd.to_numeric(data_col, errors='coerce').to_numpy()
                        setattr(
                            instance.points,
                            attr_name,
                            data_array)
                    else:
                        logger.warning(f"Warning: Attribute '{attr_name}' not found in PointData. Skipping column {col_num}.")
            except IndexError:
                logger.error(f"Error: A column number in column_map is out of bounds for the data file.")
                return None
            except Exception as e:
                logger.error(f"An error occurred while populating points from DataFrame: {e}")
                return None

        elif isinstance(result, dict):
            # Case 2: Reader returned a dictionary (for grid data)
            logger.info("Reader returned grid data. Populating survey.grid and survey.info...")
            try:
                instance.grid = GridData(z_image=result['values'])
                instance.info.x_min = result['xmin']
                instance.info.x_max = result['xmax']
                instance.info.y_min = result['ymin']
                instance.info.y_max = result['ymax']
                instance.info.z_min = result.get('zmin')
                instance.info.z_max = result.get('zmax')
                instance.info.x_gridding_delta = result['xsize']
                instance.info.y_gridding_delta = result['ysize']
                instance.info.gridding_interpolation = 'loaded_from_file'
            except KeyError as e:
                logger.error(f"Grid data dictionary is missing a required key: {e}")
                return None
        else:
            logger.error(f"Error: Reader returned an unsupported data type: {type(result)}")
            return None

        #instance.log_message(f"Data loaded from '{filename}' into Survey object '{instance.name}'.")
        logger.info(f"Data loaded from '{filename}' into Survey object '{instance.name}'.")
        return instance

    # @classmethod
    # def from_file(cls, filename, fileformat=None, **kwargs):
    #     """
    #     Creates a Survey object by reading data from a file.

    #     This smart factory method automatically determines the file format
    #     from the file extension if it is not explicitly provided. It then
    #     dispatches the call to the appropriate registered reader.

    #     Parameters
    #     ----------
    #     filename : str
    #         The full path to the data file.
    #     fileformat : str, optional
    #         The format of the input file (e.g., 'ascii', 'surfer'). If None,
    #         the format is inferred from the file's extension using a default
    #         mapping. This allows for resolving ambiguity (e.g., for '.dat' files).
    #         Defaults to None.
    #     **kwargs
    #         Additional keyword arguments passed to the specific file reader.
    #         For 'ascii': `delimiter`, `x_colnum`, `y_colnum`, `z_colnum`.

    #     Returns
    #     -------
    #     Survey or None
    #         A new Survey object instance on success, or ``None`` on failure.

    #     """
    #     # --- 1. Determine the file format ---
    #     if fileformat is None:
    #         # ... (code for inferring format remains the same) ...
    #         _, ext = os.path.splitext(filename)
    #         fileformat = DEFAULT_EXTENSION_MAP.get(ext.lower())
    #         if fileformat is None:
    #             print(f"Error: Could not determine format for extension '{ext}'.")
    #             return None

    #     # --- 2. Find the reader in the registry ---
    #     reader_details = FILE_READERS.get(fileformat)
    #     if not reader_details:
    #         print(f"Error: No reader registered for format '{fileformat}'.")
    #         return None
        
    #     reader_func = reader_details['function']

    #     # --- 3. Separate arguments and call the reader ---
    #     x_col = kwargs.pop('x_colnum', 0)
    #     y_col = kwargs.pop('y_colnum', 1)
    #     z_col = kwargs.pop('z_colnum', 2)

    #     # The remaining kwargs are for the reader function (e.g., pandas)
    #     reader_kwargs = kwargs

    #     df = reader_func(filename, **reader_kwargs)
    #     if df is None:
    #         return None

    #     # --- 4. Populate the Survey object ---
    #     try:
    #         #instance = cls(name=os.path.basename(filename))
    #         base_name = os.path.splitext(os.path.basename(filename))[0]
    #         instance = cls(name=base_name)
            
    #         instance.points.x = pd.to_numeric(df.iloc[:, x_col], errors='coerce').to_numpy()
    #         instance.points.y = pd.to_numeric(df.iloc[:, y_col], errors='coerce').to_numpy()
    #         instance.points.values = pd.to_numeric(df.iloc[:, z_col], errors='coerce').to_numpy()

    #         nan_count = np.isnan(instance.points.values).sum()
    #         if nan_count > 0:
    #             print(f"Warning: {nan_count} values in data column could not be converted to numbers and were set to NaN.")
            
    #         instance.log_message(f"Data loaded from '{filename}' into Survey object '{instance.name}'.")
    #         return instance
    #     except Exception as e:
    #         print(f"An error occurred while populating the Survey object: {e}")
    #         return None

    # --- Writter Functions ---
    def to_points_file(self, filename: str, column_map: dict = None, **kwargs):
        """
        Exports the ungridded point data to a file (e.g., CSV).

        This method saves attributes from `self.points` to a tabular text file.
        The user can specify which attributes to save and what their corresponding
        column headers should be.

        Parameters
        ----------
        filename : str
            The path for the output file.
        column_map : dict, optional
            A dictionary mapping the desired output column header to the
            PointData attribute name. If None, it defaults to saving the
            core 'x', 'y', 'values' and 'track' data.
            Example: {'Easting': 'east', 'Northing': 'north', 'Value': 'values'}
        **kwargs
            Additional keyword arguments passed directly to `pandas.DataFrame.to_csv`.
            For example: `sep=','`, `header=True`.
        """
        if self.points is None or self.points.x.size == 0:
            raise ValueError("No point data available to export.")

        # 1. Define the default mapping if none is provided
        if column_map is None:
            column_map = {
                'x': 'x',
                'y': 'y',
                'value': 'values',
                'track': 'track'
            }

        # 2. Build a dictionary of data to save
        data_to_save = {}
        for header, attribute_name in column_map.items():
            # Check if the requested attribute exists on the PointData object
            if hasattr(self.points, attribute_name):
                data_array = getattr(self.points, attribute_name)
                # Add the data only if it's not None and has content
                if data_array is not None and data_array.size > 0:
                    data_to_save[header] = data_array
                else:
                    logger.warning(
                        f"Attribute '{attribute_name}' is empty and will not be saved."
                    )
            else:
                logger.warning(
                    f"Attribute '{attribute_name}' not found in PointData and will be skipped."
                )

        # 3. Create a DataFrame and save it to a CSV file
        if not data_to_save:
            logger.error("No valid data found to save. File will not be written.")
            return

        df_to_save = pd.DataFrame(data_to_save)
        df_to_save.to_csv(filename, index=False, **kwargs)
        
        self.log_step('to_points_file', {'filename': filename, 'columns': list(column_map.keys())})
        logger.info(f"Point data successfully saved to '{filename}'.")

    def to_grid_file(self, filename: str, fileformat: str = None, **kwargs):
        """
        Exports the gridded data to a file (e.g., Surfer GRD, GeoTIFF).

        This smart dispatcher saves the contents of `self.grid` using the
        appropriate registered writer. If the format is 'surfer' and no
        specific version is provided, it defaults to 'surfer7bin'.

        Parameters
        ----------
        filename : str
            The path to the output file.
        fileformat : str, optional
            The format of the output file (e.g., 'surfer', 'geotiff'). If None,
            the format is inferred from the file extension.
        **kwargs
            Additional keyword arguments passed to the specific file writer.
            For 'surfer', this can include `fmt='surfer6ascii'`, etc.
        """

        if self.grid is None:
            raise ValueError("No gridded data available to export.")

        if fileformat is None:
            _, ext = os.path.splitext(filename)
            # This map should be defined at the top of the io.py module or locally
            writer_ext_map = {
                '.grd': 'surfer',
                '.tif': 'geotiff',
                '.tiff': 'geotiff',
            }
            fileformat = writer_ext_map.get(ext.lower())
            if fileformat is None:
                raise ValueError(
                    f"Cannot determine file format for extension '{ext}'. "
                    f"Please use a known extension or specify the 'fileformat' argument."
                )

        writer_details = FILE_WRITERS.get(fileformat)
        if not writer_details:
            raise ValueError(f"No writer registered for format '{fileformat}'.")
        
        writer_func = writer_details['function']
        
        # Call the registered writer, passing the survey object itself
        writer_func(self, filename, **kwargs)
        self.log_step('to_grid_file', {'filename': filename, 'fileformat': fileformat})

## USER DEFINED PARAMETERS -----------------------------------------------------

# list of file formats treated in this file
FORMAT_LIST = ["ascii", "netcdf", "surfer", "uxo", "cmd"]

# list of file common delimiters
DELIMITER_LIST = [r"\s", " ", ",", ";", "\t", "    ", "|", "-"]

# list of grid file format
gridtype_list = VALID_GRID_FORMAT
gridformat_list = VALID_GRID_LIST

FILE_FORMAT_DICT = {
    ".dat" : ["ASCII delimiter-separated format (*.dat)", "ascii"],
    ".txt" : ["ASCII delimiter-separated format (*.txt)", "ascii"],
    ".csv" : ["ASCII delimiter-separated format (*.csv)", "ascii"],
    ".xyz" : ["ASCII delimiter-separated format (*.xyz)", "ascii"],
    #".dat" : ["Gf instruments CMD files (*.dat)", "cmd"],  # erase previous results
    ".uxo" : ["UXO 100 format (*.uxo)", "uxo"],
    ".grd" : ["Golden Software Surfer grid format (*.grd)", "surfer"],
    ".pga" : ["WUMAP PGA format (*.pga)", "wumap"],
    ".nc" : ["WuMapPy format (*.nc)", "netcdf"],
    ".cdf" : ["WuMapPy format (*.cdf)", "netcdf"]}

def fileformat_getlist():
    '''Get the the list of file formats treated
    Returns :
        - FORMAT_LIST : list of file formats.
    '''
    return FORMAT_LIST


def delimiter_getlist():
    '''Get the the list of the common file's delimiters.

    Returns :

    :DELIMITER_LIST: list of the common file's delimiters.
    '''
    return DELIMITER_LIST


def gridfiletype_getlist():
    ''' Return the the the list of file available grid file type. '''

    return gridtype_list


def gridfileformat_getlist():
    ''' Return the the the list of file available grid file format. '''

    return gridformat_list



#---------------------------------------------------------------------------#
# Reading tools                                                             #
#---------------------------------------------------------------------------#
def sniff_delimiter(filename, skiplines=0):
    ''' Sniff the delimiter from the 1st line of a csv file.

    Parameters
    ----------
    filename : ``str``
        Name of the efile to analys.

    skiplines : ``int``
        Number of lines to skip.
        If skiplines>0, ``skiplines`` lines will be skipped and the
        following line will be considered as the fisrt.

    Return
    ------
    delimiter : ``str`` or ``None``
        If the sniffed delimiter is not in the
        :obj:`~geophpy.filesmanaging.files.delimiter_list`,
        None will be returned.

    '''

    delimiter = None
    with open(filename, "r", newline="") as csvfile:

        # skipping header lines
        if skiplines>0:
            for n in range(skiplines):
                csvfile.readline()

        dialect = csv.Sniffer().sniff(csvfile.readline())
        if dialect.delimiter:
            delimiter = dialect.delimiter

        if delimiter not in DELIMITER_LIST:
            delimiter = None

    return delimiter

def extent_file_list(filenames):
    ''' Extend file names list if '*.' is used in the file name list.

    Extend a given list of file names by replacing "*." by the full list of file names with the corresponding extension.
    (["GPS_ex2.dat", "*.csv"]),

    Parameter
    ---------
    filenames : ``str`` or ``list`` of ``str``
        File names or list of file name to be read.
        If the list contains ``'*.'``, the file list will be completed
        with all the corresponding file names.

    Return
    ------
    Complete ``list`` of all files.

    '''

    if isinstance(filenames, str):
        filenames = [filenames]

    full_filelist = []
    for filename in filenames:
        file_name, file_ext = os.path.splitext(filename)
        filename_extended = glob.glob(file_name + file_ext)
        full_filelist.extend(filename_extended)

    return full_filelist

#---------------------------------------------------------------------------#
# Reading routines                                                          #
#---------------------------------------------------------------------------#
def getlinesfrom_ascii(filename, delimiter="\t", skipinitialspace=True, skiprowsnb=0, rowsnb=1):
    ''' Reads lines in an ascii file
    Parameters :

    :filename: file name with extension to read, "test.dat" for example.

    :delimiter: delimiter between fields, tabulation by default.

    :skipinitialspace: if True, considers severals delimiters as only one : "\t\t" as '\t'.

    :skiprowsnb: number of rows to skip to get lines.

    :rowsnb: number of the rows to read, 1 by default.

    Returns :
        - colsnb : number of the columns in all rows, 0 if rows have not the sames columns nb
        - rows : rows.
    '''

    # Variable initialization
    rows = []
    colsnb = 0
    first=True

    # If file exist
    if os.path.isfile(filename) is True:
        # Opening file
        ## using "with open" ensures that the file is closed at the end
        with open(filename, "r") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=delimiter, skipinitialspace=skipinitialspace)

            # Skipping rows
            for i in range(skiprowsnb):
                next(csv_reader)

            # Reading "rowsnb" rows
            for i in range(rowsnb):
                row = next(csv_reader)
                rows.append(row)

                # First passage
                if first is True:
                    colsnb = len(row)
                    first = False

                # Not first passage
                ## initialises number of columns if size of row is different
                ## than size of previous rows
                else:
                    if (len(row) != colsnb):
                        colsnb = 0

    return colsnb, rows


# ...TBD... following function seems not to be used !!!
def getxylimitsfrom_ascii(filenameslist, delimiter="\t", skipinitialspace=True, x_colnum=1, y_colnum=2, skiprowsnb=1):
    '''Gets list of XY limits from a list of ascii files
    Parameters :

    :filenameslist: list of files to treat,

    ["file1.xyz", "file2.xyz", ...] or,

    ["file*.xyz"] to open all files with a filename beginning by "file", and ending by ".xyz".

    :delimiter: delimiter between fields in a line, "\t", ",", ";", ...

    :x_colnum: column number of the X coordinate of the profile, 1 by default.

    :y_colnum: column number of the Y coordinate inside the profile, 2 by default.

    :skiprowsnb: number of rows to skip to get values, 1 by default (to skip fields_row).

    Returns :
        - files nb treated.
        - list of XY limits :
          [[file1_xmin, file1_xmax, file1_ymin, file1_ymax], ..., [fileX_xmin, fileX_xmax, fileX_ymin, fileX_ymax]]
    '''

    filenames_fulllist = []                             # initialisation of the full list if files
    filesnb = 0
    for filenames in filenameslist:                     # for each filenames in the list
        filenamesextended = glob.glob(filenames)        # extension if the filenames field is like "*.txt"
        for filename in filenamesextended:              # for each filename in the extended filenames list
            filenames_fulllist.append(filename)         # adds the filename in the full list of files
            filesnb = filesnb + 1                       # increments files number

                                                        # initialisation
    values_array = []                                   # empty list
    xylimitsarray = []

    for filename in filenames_fulllist:                 # for each filename in the list
        if os.path.isfile(filename) is True:            # if file exist
            csvRawFile = csv.reader(open(filename,"r"), delimiter=delimiter, skipinitialspace=skipinitialspace)
            for i in range(skiprowsnb):
                next(csvRawFile)                        # splits header from values array
            data = []
            for line in csvRawFile:
                data.append(line)

            dataT = np.array(data).T
            xylimitsarray.append([dataT[x_colnum-1].min, dataT[x_colnum-1].max, dataT[y_colnum-1].min, dataT[y_colnum-1].max])

    return filesnb, xylimitsarray


def from_ascii(dataset, filenameslist, delimiter=None, skipinitialspace=True, x_colnum=1, y_colnum=2, z_colnum=3, skip_rows=1, fields_row=1):
    '''
    cf. dataset.py

    returned error codes:

    :0: no error

    :-1: invalid column number

    :-2: file does not exist

    :-3: no file found

    :-4: incompatible files

    '''

    # Invalid column number ####################################################
    if ((x_colnum < 1) or (y_colnum < 1) or (z_colnum < 1)):
      return -1

    # Initialization of the list of full filenames #############################
    if isinstance(filenameslist, str):
        filenameslist = [filenameslist]
    filenames_fulllist = []
    for filenames in filenameslist:
        # extension if filenames is like "*.txt"
        filenamesextended = glob.glob(filenames)
        for filename in filenamesextended:
            # if file exist
            if os.path.isfile(filename):
                # adds the filename in the full list of files
                filenames_fulllist.append(filename)
            else:
                return -2

    # No file found
    if (len(filenames_fulllist) == 0):
        return -3

    # if index of fields_row > nb of rows of the header ##############
    #if (fields_row >= skip_rows):
    if (fields_row > skip_rows):
        # fields X, Y, Z by default
        fields_row = -1

    # data initialization ############################################
    values_array = []

    # read the field names only in the first file ####################
    firstfile = True

##    # Checking files compatibility #############################################
##    compatibility = True
##    columns_nb = None
##    for file in filenames_fulllist:
##        col_nb, rows = getlinesfrom_file(file)
##        if columns_nb is not None and col_nb != columns_nb:
##            compatibility = False
##            return -4
##        else:
##            columns_nb = col_nb
    # Read each file ###########################################################
    for filename in filenames_fulllist:

        # Guess delimiter from file
        if delimiter is None:
            #delimiter, success, count = getdelimiterfrom_ascii(filename)
            delimiter = sniff_delimiter(filename)

        # Reading csv file
        ## using "with open" ensures that the file is closed at the end
        with open(filename, "r") as csvfile:
            csvRawFile = csv.reader(csvfile, delimiter=delimiter, skipinitialspace=skipinitialspace)

            # File header (skiped)
            csvHeader = []
            for row in range(skip_rows):
                csvHeader.append(next(csvRawFile))

            # File data
            csvValuesFileArray = []
            for csvRawLine in csvRawFile:
                csvValuesFileArray.append(csvRawLine)

            # Field names from first file ######################################
            if firstfile:
                #if (fields_row > -1):
                if (fields_row > 0):
                    try:
                        dataset.data.fields = [(csvHeader[fields_row-1])[x_colnum-1], (csvHeader[fields_row-1])[y_colnum-1], (csvHeader[fields_row-1])[z_colnum-1]]
                    except Exception as e:
                        dataset.data.fields = ["X", "Y", "Z"]
                        print(e)
                else:
                    dataset.data.fields = ["X", "Y", "Z"]

            firstfile = False

            # Data from each file ##############################################
            for csvValuesLine in csvValuesFileArray:
                try:
                    # add values from selected columns
                    values_array.append([float(csvValuesLine[x_colnum-1]), float(csvValuesLine[y_colnum-1]), float(csvValuesLine[z_colnum-1])])
                except:
                    try:
                        # add a "nan" value
                        values_array.append([float(csvValuesLine[x_colnum-1]), float(csvValuesLine[y_colnum-1]), np.nan])
                    except:
                        # skip line
                        continue

    # converts in a numpy array, sorted by column 0 then by column 1.
    #dataset.data.values = np.array(sorted(values_array, key=itemgetter(0,1)))
    dataset.data.values = np.array(values_array)
    dataset.name = os.path.splitext(os.path.basename(filenames_fulllist[0]))[0]
    # Note : values_array is an array of N points of measures [x, y, z], but not in a regular grid, and perhaps with data gaps.

    return 0


def to_ascii(dataset, filename, delimiter="\t", ignore_nodata=True):
    '''
    cf. dataset.py

    returned error codes:

    :0: no error

    '''

    csvfile = csv.writer(open(filename,"w", newline=""), delimiter=delimiter)
    csvfile.writerow(dataset.data.fields)
    if ignore_nodata:
        # index of valid data
        idx = [not any(np.isnan(val)) for val in dataset.data.values]
        csvfile.writerows(dataset.data.values[idx])
    else:
        # TODO : modify to write all additional data values
        csvfile.writerows(dataset.data.values)    

    return 0


def from_netcdf(dataset, filenameslist, x_colnum=1, y_colnum=2, z_colnum=3):
   '''
   cf. dataset.py

   returned error codes:

   :0: no error

   :-1: invalid column number

   :-2: file does not exist

   :-3: no file found

   '''
   # if bad number of column #########################################
   if ((x_colnum < 1) or (y_colnum < 1) or (z_colnum < 1)):
      return -1

   # initialisation of the full list of files ########################
   filenames_fulllist = []
   for filenames in filenameslist:
      # extension if filenames is like "*.txt" #######################
      filenamesextended = glob.glob(filenames)
      for filename in filenamesextended:
         # if file exist
         if (os.path.isfile(filename)):
            # adds the filename in the full list of files
            filenames_fulllist.append(filename)
         else:
            return -2
   if (len(filenames_fulllist) == 0):
      return -3

   # in case of MFDataset (Multi Files) ##############################
   if (len(filenames_fulllist) > 1):
      pass # ...TBD...

   # in case of single file ##########################################
   else:
      # Open netcdf file #############################################
      filename = filenames_fulllist[0]
      fileroot = nc4.Dataset(filename, "r", format="NETCDF4")

      # Read dataset attributes ######################################
      # for attr in fileroot.ncattrs():
      dataset_description      = fileroot.description
      dataset_originalfilename = fileroot.filename # ...TBD... à comparer avec la veleur courante de filename ?
      dataset_history          = fileroot.history
      dataset_os               = fileroot.os
      dataset_author           = fileroot.author
      dataset_source           = fileroot.source
      dataset_version          = fileroot.version
      # ...TBD... pour l'instant on ne fait rien des variables dataset_* ... les inclure dans la structure Dataset définie dans dataset.Py ?

      # Read dataset values ##########################################
      dat_grp = fileroot.groups["Data"]
      dataset_data_units = []
      dataset_data_values = []
      zimg = False
      eimg = False
      nimg = False
      fields  = list(dat_grp.variables.keys())
      if ("z_image" in fields):
         fields.remove("z_image")
         zimg = True
      if ("easting_image" in fields):
         fields.remove("easting_image")
         eimg = True
      if ("northing_image" in fields):
         fields.remove("northing_image")
         nimg = True
      for field in fields:
         val_var = dat_grp.variables[field]
         dataset_data_units.append(val_var.units)         # ...TBD...
         dataset_data_values.append(val_var)
      dataset.data.fields                 = list(fields)
      dataset.data.values                 = np.array(dataset_data_values)
      if (zimg):
         img_var = dat_grp.variables["z_image"]
         dataset.data.z_image                = np.array(img_var)
         dataset_info_z_units                = img_var.units # ...TBD...
         dataset_info_z_nanval               = img_var.missing_value
         dataset.info.x_min                  = img_var.x_min
         dataset.info.x_max                  = img_var.x_max
         dataset.info.y_min                  = img_var.y_min
         dataset.info.y_max                  = img_var.y_max
         dataset.info.z_min                  = img_var.z_min
         dataset.info.z_max                  = img_var.z_max
         dataset.info.x_gridding_delta       = img_var.grid_xdelta
         dataset.info.y_gridding_delta       = img_var.grid_ydelta
         dataset.info.gridding_interpolation = img_var.grid_interp
         dataset.info.plottype               = img_var.plottype
         dataset.info.cmapname               = img_var.cmapname
      if (eimg):
         eas_var = dat_grp.variables["easting_image"]
         dataset.data.easting_image          = np.array(eas_var)
         dataset_info_easting_units          = eas_var.units # ...TBD...
         dataset_info_easting_nanval         = eas_var.missing_value
      if (nimg):
         nor_var = dat_grp.variables["northing_image"]
         dataset.data.northing_image         = np.array(nor_var)
         dataset_info_northing_units         = nor_var.units # ...TBD...
         dataset_info_northing_nanval        = nor_var.missing_value

      # Read dataset georef data #####################################
      geo_grp = fileroot.groups["GeoRefSystem"]
      dataset.georef.active            = (geo_grp.active == 1)
      if (dataset.georef.active):
         dataset.georef.refsystem      = geo_grp.refsystem
         dataset.georef.utm_zoneletter = geo_grp.utm_zoneletter
         dataset.georef.utm_zonenumber = geo_grp.utm_zonenumber
         ptu_var = geo_grp.variables["pts_number"]
         pte_var = geo_grp.variables["pts_easting"]
         ptn_var = geo_grp.variables["pts_northing"]
         pta_var = geo_grp.variables["pts_abs"]
         pto_var = geo_grp.variables["pts_ord"]
         for p in range(len(geo_grp.dimensions["point"])):
#           dataset.georef.points_list.append(GeoRefPoint(easting=pue_var[p], northing=pun_var[p]))
            dataset.georef.points_list.append([ptu_var[p], pte_var[p], ptn_var[p], pta_var[p], pto_var[p]])

      # Close netcdf file ############################################
      fileroot.close()

   return 0


def to_netcdf(dataset, filename, description=None):
   '''
   cf. dataset.py

   returned error codes:

   :0: no error

   '''
   # netCDF does not support "None" values ###########################
   if description is None:
      description = "nil"
   if dataset.data.z_image is None:
      dataset.interpolate()

   # Open file for writing ###########################################
   # "w" means write (and clobber !), or create if necessary
   # "a" means append (does not create !)
   fileroot = nc4.Dataset(filename, "w", format="NETCDF4")

   # Create root attributes ##########################################
   fileroot.description = description
   fileroot.filename    = filename
   fileroot.history     = "Created " + time.ctime(time.time())
   fileroot.os          = platform.uname()
   fileroot.author      = os.getlogin()
   fileroot.source      = geophpy.__software__ + " " + geophpy.__version__ + " of " + geophpy.__date__
   fileroot.version     = "netCDF " + nc4.getlibversion()

   # Create data group ###############################################
   dat_grp = fileroot.createGroup("Data")
   val_len = len(dataset.data.values)
   val_dim = dat_grp.createDimension("value", val_len)
   val_typ = dataset.data.values.dtype
   f = 0
   for field in [f.replace("/","\\") for f in dataset.data.fields]:
      val_var = dat_grp.createVariable(field, val_typ, ("value",))
      val_var.units     = "nil" # ...TBD...
      val_var[:]        = dataset.data.values[:,f]
      f += 1
   if dataset.data.z_image is not None:
      nx, ny  = dataset.data.z_image.shape
      abs_dim = dat_grp.createDimension("x", nx)
      ord_dim = dat_grp.createDimension("y", ny)
      img_typ = dataset.data.z_image.dtype
      img_var = dat_grp.createVariable("z_image", img_typ, ("x","y",))
      img_var.units        = "nil" # ...TBD...
      img_var.missing_value= np.nan
      img_var.x_min        = dataset.info.x_min
      img_var.x_max        = dataset.info.x_max
      img_var.y_min        = dataset.info.y_min
      img_var.y_max        = dataset.info.y_max
      img_var.z_min        = dataset.info.z_min
      img_var.z_max        = dataset.info.z_max
      img_var.grid_xdelta  = dataset.info.x_gridding_delta
      img_var.grid_ydelta  = dataset.info.y_gridding_delta
      img_var.grid_interp  = dataset.info.gridding_interpolation
      img_var.plottype     = dataset.info.plottype
      img_var.cmapname     = dataset.info.cmapname
      img_var[...]         = dataset.data.z_image[...]
   if (dataset.data.easting_image != None):
      eas_typ = dataset.data.easting_image.dtype
      eas_var = dat_grp.createVariable("easting_image", eas_typ, ("x","y",))
      eas_var.units        = "nil" # ...TBD...
      eas_var.missing_value= np.nan
      eas_var[...]         = dataset.data.easting_image[...]
   if (dataset.data.northing_image != None):
      nor_typ = dataset.data.northing_image.dtype
      nor_var = dat_grp.createVariable("northing_image", nor_typ, ("x","y",))
      nor_var.units        = "nil" # ...TBD...
      nor_var.missing_value= np.nan
      nor_var[...]         = dataset.data.northing_image[...]

   # Create georefsys group ##########################################
   geo_grp = fileroot.createGroup("GeoRefSystem")
   if (dataset.georef.active):
      geo_grp.active         = 1
      geo_grp.refsystem      = dataset.georef.refsystem
      geo_grp.utm_zoneletter = dataset.georef.utm_zoneletter
      geo_grp.utm_zonenumber = dataset.georef.utm_zonenumber
      pts_len = len(dataset.georef.points_list)
      pts_dim = geo_grp.createDimension("point", pts_len)
      ptu_var = geo_grp.createVariable("pts_number",   "i", ("point",))
      pte_var = geo_grp.createVariable("pts_easting",  "d", ("point",))
      ptn_var = geo_grp.createVariable("pts_northing", "d", ("point",))
      pta_var = geo_grp.createVariable("pts_abs",      "d", ("point",))
      pto_var = geo_grp.createVariable("pts_ord",      "d", ("point",))
      p = 0
      for [pts_number, pts_easting, pts_northing, pts_abs, pts_ord] in dataset.georef.points_list:
         ptu_var[p] = pts_number
         pte_var[p] = pts_easting
         ptn_var[p] = pts_northing
         pta_var[p] = pts_abs
         pto_var[p] = pts_ord
         p += 1
   else:
      geo_grp.active      = 0

   # Close file ######################################################
   fileroot.close()

   # Return error code ###############################################
   return 0


def from_grd(dataset, filenameslist, gridtype=None):
    '''
    :0: no error

    :-1: invalid file encounter

    :-2: file does not exist

    :-3: no file found
    '''

    # Extension if filenames is like "*.grd"
    filenames_fulllist = extent_file_list(filenameslist)

    # # Initialization of the full list of files
    # filenames_fulllist = []

    # for filenames in filenameslist:
        # # extension if filenames is like "*.grd"
        # filenamesextended = glob.glob(filenames)

        # for filename in filenamesextended:
            # # File exists
            # if (os.path.isfile(filename)):
                # # adds the filename in the full list of files
                # filenames_fulllist.append(filename)

            # # File does noes exist
            # else:
                # return -2

    # No file found
    if (len(filenames_fulllist) == 0):
        return -3

    # Multiple files  ##########################################################
    if (len(filenames_fulllist) > 1):
        pass # ...TBD...

    # Single file ##############################################################
    # Reading Surfer grid file
    data =  read_grd(filenames_fulllist[0], frmt=gridtype)
    #grid = SurferGrid.fromfile(*filenames_fulllist)

    # Copying Surfer grid to the DataSet object
    # dataset.info.x_min = grid.getXmin()
    # dataset.info.x_max = grid.getXmax()
    # dataset.info.y_min = grid.getYmin()
    # dataset.info.y_max = grid.getYmax()
    # dataset.info.z_min = grid.getZmin()
    # dataset.info.z_max = grid.getZmax()
    # dataset.info.x_gridding_delta = grid.xSize
    # dataset.info.y_gridding_delta = grid.ySize
    # dataset.info.gridding_interpolation = "none"

    # dataset.data.z_image = grid.data
    # dataset.data.fields =  ["X", "Y", "Z"]
    dataset.info.x_min = data["xmin"]
    dataset.info.x_max = data["xmax"]
    dataset.info.y_min = data["ymin"]
    dataset.info.y_max = data["ymax"]
    dataset.info.z_min = data["zmin"]
    dataset.info.z_max = data["zmax"]
    dataset.info.x_gridding_delta = data["xsize"]
    dataset.info.y_gridding_delta = data["ysize"]
    dataset.info.gridding_interpolation = "none"

    dataset.data.z_image = data["values"]
    dataset.data.fields =  ["X", "Y", "Z"]

    # Creating false data values point #########################################
    x, y = dataset.get_xyvect()

    values_array = []
    for i, row in enumerate(dataset.data.z_image):
        for j, val in enumerate(row):
            values_array.append([x[0][j], y[0][i], val])

    dataset.data.values =  np.array(values_array)

    # Return error code ########################################################
    return 0


def to_grd(dataset, filename, gridtype=None):
    '''
    cf. :meth:`~geophpy.dataset.DataSet.to_grd`

    returned error codes:

    :0: no error

    '''

    # Copying the DataSet object to a SurferGrid object and writting grid file #
    # grid = SurferGrid(nRow=dataset.data.z_image.shape[0],
                      # nCol=dataset.data.z_image.shape[1],
                      # xLL=dataset.info.x_min,
                      # yLL=dataset.info.y_min,
                      # xSize=dataset.info.x_gridding_delta,
                      # ySize=dataset.info.y_gridding_delta,
                      # zMin=dataset.info.z_min,
                      # zMax=dataset.info.z_max,
                      # data=dataset.data.z_image,
                      # BlankValue=1.71041e38,
                      # Rotation=0,
                      # filename=filename)

    # grid.write(gridtype=gridtype)
    data = {"ncol" : dataset.data.z_image.shape[1],
            "nrow" : dataset.data.z_image.shape[0],
            "xll" : dataset.info.x_min,
            "yll" : dataset.info.y_min,
            "xmin" : dataset.info.x_min,
            "xmax" : dataset.info.x_max,
            "ymin" : dataset.info.y_min,
            "ymax" : dataset.info.y_max,
            "zmin" : dataset.info.z_min,
            "zmax" : dataset.info.z_max,
            "xsize" : dataset.info.x_gridding_delta,
            "ysize" : dataset.info.y_gridding_delta,
            "values" : dataset.data.z_image,
            }

    error =  write_grd(filename, data, frmt=gridtype)

    # or SurferGrid.tofile(grid, filename)
##  # or SurferGrid.tofile(SurferGrid(
##                                 nRow=None, nCol=None,
##                                 LL=dataset.info.x_min, yLL=dataset.info.y_min,
##                                 xSize=dataset.info.x_gridding_delta,
##                                 ySize=dataset.info.y_gridding_delta,
##                                 zMin=dataset.info.z_min, zMax=dataset.info.z_max,
##                                 data=dataset.data.z_image,
##                                 BlankValue=1.71041e38, Rotation=0
##                                 ),
##                      filename
##                      )

    return 0


def from_uxo(dataset, filenameslist):
    '''
    cf. :meth:`~geophpy.dataset.DataSet.from_file`

    returned error codes:

    :0: no error

    :-2: file does not exist

    :-3: no file found

    :-4: incompatible files

    '''

    # Extension if filenames is like "*.uxo"
    filenames_fulllist = extent_file_list(filenameslist)

    # No file found
    if (len(filenames_fulllist) == 0):
        return -3

    # Multiple files
    #...TBD...
    if (len(filenames_fulllist) > 1):
        pass

    # Single file
    
    try:
        data = read_uxo(filenames_fulllist[0])

    except Exception as e:
        data =  None
        
    if data is None:
        try:
            data = read_uxo(filenames_fulllist[0], encoding="utf-8")

        except Exception as e:
            data =  None

    if data is None:
        raise Exception("Cannot read data format from file.", filenames_fulllist[0])

    name = os.path.splitext(os.path.basename(filenames_fulllist[0]))[0]

    #track = None
    # Projected coordinates
    if data.get("east") is not None and data.get("north") is not None:
        xy = np.vstack((data["east"],
                        data["north"]))
        fields =  ["East", "North", "Value"]
        east = data["east"]
        north = data["north"]

    # Geographic coordinates
    elif     data.get("long") is not None and data.get("lat") is not None:
        xy = np.vstack((data["long"],
                        data["lat"]))
        fields =  ["Long", "Lat", "Value"]
        east = data["long"]
        north = data["lat"]

    # Local coordinates
    else:
        xy = np.vstack((data["x"],
                        data["y"]))
        fields =  ["X", "Y", "Value"]

    # Choice of parameter
    param = data["values"]
    dataset.data.values = np.vstack((xy, param)).T
    dataset.data.fields = fields
    dataset.data.x = data["x"]
    dataset.data.y = data["y"]
    dataset.data.east = data.get("east", None)
    dataset.data.north = data.get("north", None)
    dataset.data.long = data.get("long", None)
    dataset.data.lat = data.get("lat", None)
    dataset.data.track = data.get("track", None)
    dataset.name = name

    return 0


def to_uxo(dataset, filename):
    '''
    cf. :meth:`~geophpy.dataset.DataSet.to_file`

    returned error codes:

    :0: no error

    '''

    data= dataset.to_dict()
    data["values"] = data["values"][:, -1]
    data["unit"] = "nT"
    write_uxo(filename, data)

    return 0

###
##
#  Implemented in the next version
##
###
##def from_cmdfile(dataset, filenameslist, param="cond1"):
##    '''
##    cf. dataset.py
##
##    returned error codes:
##
##    :0: no error
##
##    :-1: invalid column number
##
##    :-2: file does not exist
##
##    :-3: no file found
##
##    :-4: incompatible files
##    '''
##
##    # Initialisation of the full list of files #################################
##    filenames_fulllist = []
##
##    for filenames in filenameslist:
##        # extension if filenames is like "*.grd"
##        filenamesextended = glob.glob(filenames)
##
##        for filename in filenamesextended:
##            # File exists
##            if (os.path.isfile(filename)):
##                # adds the filename in the full list of files
##                filenames_fulllist.append(filename)
##
##            # File does noes exist
##            else:
##                return -2
##
##    # No file found
##    if (len(filenames_fulllist) == 0):
##        return -3
##
##    # Multiple files  ##########################################################
##    if (len(filenames_fulllist) > 1):
##        pass # ...TBD...
##
##    # Single file ##############################################################
##    # Reading CMD dat file
##    CMDdata = CMDFileReader(filename).data
##
##    # Geographic coordinates
##    if hasattr(CMDdata, "long") and not hasattr(CMDdata, "east"):
##        coord = np.vstack((CMDdata.long,
##                           CMDdata.lat))
##
##    # Projected coordinates
##    elif hasattr(CMDdata, "east"):
##        coord = np.vstack((CMDdata.east,
##                           CMDdata.north))
##
##    # Local coordinates
##    else:
##        coord = np.vstack((CMDdata.x,
##                           CMDdata.y))
##
##    # Choice of parameter
##    ###
##    ##
##    # Or use an dictionnary to choose attribute ?
##    ##
##    ###
##    if param.lower() == "cond1":
##        param = CMDdata.cond1
##
##    elif param.lower() == "inph1":
##        param = CMDdata.inph1
##
##    elif param.lower() == "cond2":
##        param = CMDdata.cond2
##
##    elif param.lower() == "inph2":
##        param = CMDdata.inph2
##
##    elif param.lower() == "cond3":
##        param = CMDdata.cond3
##
##    elif param.lower() == "inph3":
##        param = CMDdata.inph3
##
##    dataset.data.values = np.vstack((coord, param)).T
##
##    return 0

def read_ascii(filename, 
               delimiter=None,
               x_col=1,
               y_col=2,
               z_col=None,
               val_col=3,
               skipinitialspace=True,
               skip_rows=1,
               fields_row=1,
               encoding=None):
    ''' Read ASCII delimiter-separated Values file.

    Parameters
    ----------

    filename : 

    delimiter : str
        A one-character string used to separate fields. It defaults to None.
        If None, the delimiter will be sniffed from the first line.

    x_col : int
        Column number of the X coordinate of the profile (1 by default).

    y_col : int
        Column number of the Y coordinate inside the profile (2 by default).

    z_col : int
        Column number of the Z coordinate inside the profile (None by default).
        If None, the Z coordinate will be set to 0.

    val_col : int
        Column number of data values (3 by default).

    skipinitialspace : bool
        If True, several contiguous delimiters are equivalent to one.

    skip_rows : int
        Total number of rows to skip at the beginning of the file, i.e. total number of header rows (0 by default).

    fields_row : int
        Row number to read the field names. Should be be inferior or equal to skip_rows.
        If fields_row > skip_rows. A total number of  line equals to fields_row will be skipped.
        If -1, the default field names will be "X", "Y" and "Value"

    encoding : str or None.
        Encoding used for text data. If None (by default) the encoding from the system's user preferences will be used.

    Returns
    -------
    error : {0, -1, -2, -3}
        Error code :
        *0: no error
        *-1: invalid column number
        *-2: file does not exist
        *-3: no file found

    data : dict
        Dictionary containing the data.
        Dictionary keys are :
            * "x" : local x position in meters.
            * "y" : local y position in meters.
            * "z" : local z position in meters.
            * "values" : data values.
            * "track" : track number.
            * "long" [optional] : data longitude in decimal degrees.
            * "lat" [optional] : data latitude in decimal degrees.
            * "alt" [optional] : data altitude.
            * "east" [optional] : data easting in meters.
            * "north" [optional] : data northing in meters.
            * "elevation" [optional] : data elevation in meters.

    '''

    if not os.path.exists(filename):
        raise OSError(2, "No such file or directory", filename)

    # Invalid column number
    if x_col < 1 or y_col < 1 or val_col < 1:
        raise TypeError("Invalid column number (must be >0).")

    if  fields_row > skip_rows:
        fields_row = -1

    # Guess delimiter from file
    delimiter = sniff_delimiter(filename)

    with open(filename, "r") as file:
        csvfile = csv.reader(file, delimiter=delimiter, skipinitialspace=skipinitialspace)

        # Header lines
        headers = []
        for row in range(skip_rows):
            headers.append(next(csvfile))

        # Fields line
        if fields_row > 0:
            try:
                fields = [(headers[fields_row-1])[x_col-1], (headers[fields_row-1])[y_col-1], (headers[fields_row-1])[val_col-1]]
            except Exception as e:
                fields = ["X", "Y", "Value"]
                print(e)

        # Data lines
        for csvRawLine in csvRawFile:
            csvValuesFileArray.append(csvRawLine)


def sniff_delimiter(filename, skiplines=0):
    ''' Sniff the delimiter from the 1st line of a csv file.

    Parameters
    ----------
    filename : str
        File name.

    skiplines : int
        Number of lines to skip.
        If skiplines>0, ``skiplines`` lines will be skipped and the
        following line will be considered for the analysis.

    Return
    ------
    delimiter : ``str`` or ``None``
        None will be returned if the analysis failles.

    '''

    delimiter = None
    with open(filename, "r", newline="") as csvfile:

        # skipping header lines
        if skiplines > 0:
            for n in range(skiplines):
                csvfile.readline()

        dialect = csv.Sniffer().sniff(csvfile.readline())
        if dialect.delimiter:
            delimiter = dialect.delimiter

        if delimiter not in DELIMITER_LIST:
            delimiter = None

    return delimiter


def df_to_matrix(file,sep="\t",output_file="dtm.json"):
    """
    Create the "matrix" representation of a grid dataframe.
    
    Notes
    -----
    Output is in JSON format.
    
    Parameters
    ----------
    file : str or dataframe 
        Dataframe file or loaded dataframe.
    ``[opt]`` sep : str, default : ``"\\t"``
        Dataframe separator.
    ``[opt]`` output_file : str, default : ``"dtm.json"``
        Output file name.
    
    Raises
    ------
    * File not found.
    * Wrong separator or columns not found.
    
    See also
    --------
    ``matrix_to_df``
    """
    if isinstance(file,str):
        try:
            # Chargement des données
            df = pd.read_csv(file, sep=sep)
        except FileNotFoundError:
            raise FileNotFoundError("File {} not found.".format(file))
    else:
        df = file
        
    nc_data = df.columns
    # Un peu en redite du check suivant
    if len(nc_data) <= 1:
        raise FileNotFoundError("Unable to read {}, is the separator {} correct ?".format(file,repr(sep)))
    try:
        # Sélection des colonnes
        ncx = nc_data[0]
        ncy = nc_data[1]
        nc_data = nc_data[2:]
        nb_data = len(nc_data)
    except KeyError:
        raise FileNotFoundError("Unable to read {}, is the separator {} correct ?".format(file,repr(sep)))
    
    grid_mat_row = [[] for n in range(nb_data)]
    grid_mat = [[] for n in range(nb_data)]
    # On sépare les lignes à partir de X
    curr_x = df.loc[0,ncx]
    for index, row in df.iterrows():
        if row[ncx] != curr_x:
            for n in range(nb_data):
                grid_mat[n].append(grid_mat_row[n])
            curr_x = row[ncx]
            grid_mat_row = [[] for n in range(nb_data)]
        
        # On remplit le tableau pour chaque donnée
        for n in range(nb_data):
            grid_mat_row[n].append(row[nc_data[n]])
    for n in range(nb_data):
        grid_mat[n].append(grid_mat_row[n])
    
    # Formules magiques pour recalculer les dimensions de la grille (ne me remerciez pas)
    last = len(df)-1
    pas_X = df.groupby(ncy)[ncx].apply(lambda x: x.diff().mean()).reset_index()[ncx].mean()
    pas_Y = df.groupby(ncx)[ncy].apply(lambda x: x.diff().mean()).reset_index()[ncy].mean()
    ext = [df.loc[0,ncx],df.loc[last,ncx],df.loc[0,ncy],df.loc[last,ncy]]
    pxy = [len(df[df[ncy] == ext[2]]),len(df[df[ncx] == ext[0]])]
    
    grid_mat_t = [[[np.nan for i in range(pxy[0])] for j in range(pxy[1])] for n in range(nb_data)]
    for n in range(nb_data):
        # Transposer
        for i in range(pxy[0]):
            for j in range(pxy[1]):
                grid_mat_t[n][j][i] = grid_mat[n][i][j]
    
    # Dictionaire à enregistrer
    grid_save = {"grid" : grid_mat_t, "ext" : ext, "pxy" : pxy, "step" : [pas_X,pas_Y],
                 "ncx" : ncx.split("|"), "ncy" : ncy.split("|"), "ncz" : nc_data.to_list()}
    
    # Enregistrer dans un .json
    with open(output_file, "w") as f:
        json.dump(grid_save, f, indent=None, cls=JSON_Indent_Encoder)

# Convertit le format matrice en dataframe

def matrix_to_df(file,sep="\t",output_file="mtd.dat"):
    """
    Create the dataframe representation of a "matrix".
    
    Notes
    -----
    ``sep`` is only relevant for the output.
    Weird coordinates shenanigans serves to comply with the krigind grid structure.
    
    Parameters
    ----------
    file : str
        Matrix file.
    ``[opt]`` sep : str, default : ``"\\t"``
        Output dataframe separator.
    ``[opt]`` output_file : str, default : ``"mtd.dat"``
        Output file name.
    
    Raises
    ------
    * File not found.
    * File is not a JSON.
    
    See also
    --------
    ``df_to_matrix``
    """
    try:
        with open(file, "r") as f:
            # Chargement des données
            grid_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("File {} not found.".format(file))
    except json.JSONDecodeError:
        raise FileNotFoundError("File {} is not a json.".format(file))
    
    # Sélection des données
    nb_data = len(grid_dict["ncz"])
    nb_ecarts = len(grid_dict["ncx"])
    grid = np.array(grid_dict["grid"])
    
    # On construit une colonne position qui contient les labels de toutes
    ncx = ""
    ncy = ""
    for e in range(nb_ecarts):
        ncx += grid_dict["ncx"][e]+"|"
        ncy += grid_dict["ncy"][e]+"|"
    ncx = ncx[:-1]
    ncy = ncy[:-1]
    
    # Sélection des dimensions
    min_X = grid_dict["ext"][0]
    min_Y = grid_dict["ext"][2]
    pas_X = grid_dict["step"][0]
    pas_Y = grid_dict["step"][1]
    nc_data = grid_dict["ncz"]
    df = pd.DataFrame(columns=[ncx,ncy]+nc_data)
    
    # Remplissage du dataframe
    for i in range(grid_dict["pxy"][0]):
        for j in range(grid_dict["pxy"][1]):
            # Reconstitution de la position de la case
            row_data = {ncx : round(min_X+pas_X*i,CONFIG.prec_coos), ncy : round(min_Y+pas_Y*j,CONFIG.prec_coos)}
            for n in range(nb_data):
                row_data[nc_data[n]] = grid[n][j][i]
            r = i*grid_dict["pxy"][1] + j
            df.loc[r] = pd.Series(row_data)
    
    # Enregistrement en .dat
    df.to_csv(output_file, index=False, sep=sep)

# Convertit le format matrice en .grd pour surfer

def matrix_to_grd(file,fmt,output_file=None):
    """
    Create the .grd representation of a "matrix".
    Current supported format are :
    * Golden Software Surfer 6 binary
    * Golden Software Surfer 6 ascii
    * Golden Software Surfer 7 binary
    
    Notes
    -----
    .grd is compatible with *Golden Software Surfer*
    
    Parameters
    ----------
    file : str
        Matrix file.
    fmt : str, [``"surfer6bin"``, ``"surfer6ascii"``, ``"surfer7bin"``]
        Surfer format.
    ``[opt]`` output_file : ``None`` or str, default : ``None``
        Output file name. If ``None``, set to ``file + "_grd.grd"``
    
    Raises
    ------
    * File not found.
    * File is not a JSON.
    * TypeError exception in one of ``grd.py`` write functions.
    """
    try:
        with open(file, "r") as f:
            # Chargement des données
            grid_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("File {} not found.".format(file))
    except json.JSONDecodeError:
        raise FileNotFoundError("File {} is not a json.".format(file))
    
    # Séparatio de la grille pour chaque donnée
    if output_file == None:
        output_file = file[:-5] + "_grd.grd"
    
    # Remplissage des champs pour la fonction de grd.py
    nb_data = len(grid_dict["ncz"])
    for n in range(nb_data):
        grid = np.array(grid_dict["grid"][n])
        zmin = min(grid.flatten())
        zmax = max(grid.flatten())
        data = {"ncol" : grid_dict["pxy"][0], "nrow" : grid_dict["pxy"][1], "xmin" : grid_dict["ext"][0], "xmax" : grid_dict["ext"][1], 
                "ymin" : grid_dict["ext"][2], "ymax" : grid_dict["ext"][3], "xsize" : grid_dict["step"][0], "ysize" : grid_dict["step"][1], 
                "zmin" : zmin, "zmax" : zmax, "values" : grid, "blankvalue" : np.inf}
        # Nommage des fichiers
        if nb_data == 1:
            filename = output_file
        else:
            spl = output_file.split(".")
            filename = spl[0]+"_"+str(n+1)+"."+spl[1]
        
        # Choix du format
        write_grd(filename, data, fmt)

# Convertit le format matrice en .grd pour surfer

def grd_to_matrix(file_list,fmt,output_file=None):
    """
    Create the "matrix" representation of a .grd.
    Current supported format are :
    * Golden Software Surfer 6 binary
    * Golden Software Surfer 6 ascii
    * Golden Software Surfer 7 binary
    
    Notes
    -----
    .grd is compatible with *Golden Software Surfer*
    
    Parameters
    ----------
    file : str
        .grd file.
    fmt : str, [``"surfer6bin"``, ``"surfer6ascii"``, ``"surfer7bin"``]
        Surfer format.
    ``[opt]`` output_file : ``None`` or str, default : ``None``
        Output file name. If ``None``, set to ``file + "_mtx.json"``
    
    Raises
    ------
    * File not found.
    * TypeError exception in one of ``grd.py`` write functions.
    """
    data = {}
    grid = []
    ncx = []
    ncy = []
    ncz = []
    
    # Pour chaque fichier
    for ic,file in enumerate(file_list):
        # On choisit le format adapté
        data = read_grd(file, fmt)
        
        # Restitution de la grille
        grid.append(data["values"].tolist())
        # Noms placeholder des colonnes (l"info n"est pas là)
        ncx.append("x_"+str(ic+1))
        ncy.append("y_"+str(ic+1))
        ncz.append("z_"+str(ic+1))
    
    # Nouveau format
    grid_save = {"grid" : grid, "ext" : [data["xmin"],data["xmax"],data["ymin"],data["ymax"]], 
                 "pxy" : [data["ncol"],data["nrow"]], "step" : [data["xsize"],data["ysize"]], 
                 "ncx" : ncx, "ncy" : ncy, "ncz" : ncz}
    
    # Enregistrement en .json
    if output_file == None:
        output_file = file[:-4] + "_mtx.json"
    with open(output_file, "w") as f:
        json.dump(grid_save, f, indent=None, cls=JSON_Indent_Encoder)  