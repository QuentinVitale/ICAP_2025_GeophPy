# -*- coding: utf-8 -*-
'''
    geophpy.core.registery
    ---------------------------

    Registery for all file reader functions.

    :copyright: Copyright 2025 Q. Vitale, L. Darras, P. Marty, and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

# A dictionary to hold all registered file reader functions and their metadata
FILE_READERS = {}
def register_reader(format_name, description, default_map=None):
    """
    A decorator that registers a reader function along with its description.

    Parameters
    ----------
    format_name : str
        The unique identifier for the format (e.g., 'ascii', 'surfer', 'cmd_dat').
    description : str
        A user-friendly description for GUIs (e.g., "ASCII delimited (*.txt, *.csv)").
    """
    def decorator(func):
        FILE_READERS[format_name] = {
            'function': func,
            'description': description,
            'default_map': default_map 
        }
        return func
    return decorator

# helper function to provide a list for a GUI
def get_reader_formats():
    """Returns a list of (description, format_name) for use in GUIs."""
    return sorted([(details['description'], name) for name, details in FILE_READERS.items()])

# Dictionary to hold all registered file writer functions
FILE_WRITERS = {}

def register_writer(format_name, description):
    """
    A decorator that registers a writer function along with its description.

    Parameters
    ----------
    format_name : str
        The unique identifier for the format (e.g., 'ascii', 'surfer').
    description : str
        A user-friendly description for GUIs (e.g., "Golden Software Surfer Grid (*.grd)").
    """
    def decorator(func):
        FILE_WRITERS[format_name] = {
            'function': func,
            'description': description
        }
        return func
    return decorator