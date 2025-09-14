# -*- coding: utf-8 -*-
"""
The mag subpackage of geophpy, containing the magnetic data reading,
plotting, and processing capabilities.

:copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
:license: GNU GPL v3, see LICENSE for details.
"""

from . import uxo  # This import ensures the @register_reader in uxo.py is executed.
from . import processing  # This import loads the MagneticGridMixin and other processing methods.
from . import plot  # This import loads the specialized magnetic plotting methods.
