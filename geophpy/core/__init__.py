# -*- coding: utf-8 -*-
"""
The core subpackage of geophpy, containing the main Survey object,
data structures, and processing capabilities.

:copyright: Copyright 2014-2025 L. Darras, P. Marty, Q. Vitale and contributors.
:license: GNU GPL v3, see LICENSE for details.
"""

# --- Import core modules to trigger registrations and build the subpackage API ---
from . import registry  # Import the registry first, as other modules will use it.
from . import datastructures

# Import all modules that contain registered readers/writers to populate the registry.
from . import ascii
from . import grd
#from . import uxo

# --- Import the mixin classes so they can be assembled into the Survey class.
from .io import IOMixin
#from .operations import OperationsMixin
#from .processing import ProcessingMixin

# check if following imports are still valid
from geophpy.core.io import *
from geophpy.core.processing import *
from geophpy.core.utils import *
from geophpy.core.operation import *

import logging
# --- Basic Logging Configuration ---
# Sets up a default handler that prints INFO level messages and higher to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
