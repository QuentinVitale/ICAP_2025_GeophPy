# -*- coding: utf-8 -*-
"""
geophpy
-------

An open source python package for sub-surface geophysical data processing.

:copyright: Copyright 2014-2025 L. Darras, Q. Vitale, P. Marty and contributors.
:license: GNU GPL v3, see LICENSE for details.
"""

from geophpy.core.io import FILE_FORMAT_DICT, FORMAT_LIST

# --- Package Metadata ---
__version__ = "0.35.O"
__author__ = "L. Darras, Q. Vitale, P. Marty,  and contributors"
__license__ = "GPLv3"
__software__ = 'GeophPy'
__date__ = '15/09/2025'
__description__ = 'Tools for sub-surface geophysical survey data processing'

# --- Plugin System Registration ---
# By importing the subpackages here, we trigger the @register_reader and
# @register_writer decorators within their modules. This populates the
# central registries before any user code is run.
# This makes the I/O system fully extensible.
from . import core
from . import electrical_mapping
from . import emi
from . import georeferencing
from . import mag
from . import magnetic_susceptibility
from . import seismic
from . import visualization


# --- Public API ---
# Expose the main classes to the top-level namespace for easy access
# by the end-user. This allows them to write `from geophpy import Survey` etc.
from .core.survey import Survey
#from geophpy.core import *

# Define what `from geophpy import *` should import.
#__all__ = ['Survey']