# -*- coding: utf-8 -*-
"""
   geophpy.core.ascii
   ------------------

   Provides a registered reader for generic ASCII (delimiter-separated) files.

   :copyright: Copyright 2014-2025 L. Darras,  Q. Vitale, P. Marty, and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

import pandas as pd
from .registry import register_reader


ASCII_DEFAULT_MAP = {'x': 0, 'y': 1, 'values': 2}

@register_reader(
    format_name='ascii',
    description="ASCII delimited text (*.txt, *.csv, *.dat)",
    default_map=ASCII_DEFAULT_MAP
)

def read_ascii(filename, **kwargs):
    """
    Reads an ASCII file using pandas and returns a DataFrame.

    This function serves as the low-level reader for the 'ascii' format. It
    leverages the robustness of pandas for parsing and returns a DataFrame
    for further processing, or None if the read operation fails.

    Parameters
    ----------
    filename : str
        The full path to the ASCII data file.
    **kwargs
        Additional keyword arguments passed directly to `pandas.read_csv`.
        For example: `delimiter=','`, `skiprows=5`, `header=None`.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the file's data on success, or ``None`` on
        failure.

    """
    try:
        df = pd.read_csv(filename, header=None, comment='#', **kwargs)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filename}': {e}")
        return None
    
