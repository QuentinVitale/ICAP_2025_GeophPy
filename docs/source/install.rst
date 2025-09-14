.. _chap-install-geophpy:

Installation
************

A Python (3.x) installation is necessary to install the :program:`GeophPy` package. 

Using pip
=========

You can install :program:`GeophPy` directly from the `PyPI`_ repository using ``pip``.

First, make sure you have an up-to-date version of ``pip`` using the command:

    >>>  pip install --upgrade pip
    >>>  or
    >>>  python -m pip install --upgrade pip

Then, install, upgrade (or uninstall) :program:`GeophPy` directly from `PyPI`_ repository using ``pip`` with these commands:

    >>>  pip install geophpy
    >>>  pip install --upgrade geophpy
    >>>  pip uninstall  geophpy

You can also download the zip file "GeophPy-vx.y" from the `PyPI`_ repository, and from the download folder use:

    >>>  pip install GeophPy-vx.y.zip

.. _`PyPI`: https://pypi.org/project/GeophPy/

Building from sources
=====================

Download the zip file "GeophPy-vx.y" and unzip it. 
Go to the unzipped folder and run the install script with the following command:

    >>>  python setup.py install
    >>>  or
    >>>  python -m setup.py install


.. warning::

   If you get the following error:

   **Error while finding module specification for 'setup.py' (AtributError: module 'setup' has no attribute '__path__')**
   
   you might need to enter the command without using the *'.py'* extension for the setup file

       >>>  python -m setup install

Dependencies
============

:program:`GeophPy` depends on the following Python packages:

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 0
   :align: center

   *  -  Name
      -  Version
      -  Comment
   *  -  numpy
      -  
      -  
   *  -  SciPy
      - 
      - 
   *  -  matplotlib
      -   
      -  
   *  -  netCDF4
      - 
      - 
   *  -  Pillow
      - 
      - 
   *  -  PySide
      - 
      - 
   *  -  pyshp
      - 
      - 
   *  -  simplekml
      - 
      - 
   *  -  utm
      - 
      - 
   *  -  Sphinx
      - 1.4.3 (or greater)
      - 

.. tip:: Failure on :program:`Windows`

   :program:`GeophPy` uses others Python modules and packages that should be automatically installed. 
   However, the installation of these modules may failed on :program:`Windows` (in the absence of a C++ compiler for instance).

   They can be installed independently using the useful website: http://www.lfd.uci.edu/~gohlke/pythonlibs/.
   This website provides many popular scientific Python packages precompiled for :program:`Windows` distributions.

   To install a package independently:

   #. Download the precompiled package sources corresponding to your Python version and :program:`Windows` distribution (SomePackage-vx.y-cp3x-cp3xm_winxx.whl);

      .. image:: _static/figInstallGeophPyPackages.png
                   :height: 4cm

   #. In download folder, use a command prompt and install the package using ``pip`` with the name of the downloaded archive:

       >>> python setup.py install SomePackage-vx.y-cp3x-cp3xm_winxx.whl
       >>> or
       >>> python -m setup.py install SomePackage-vx.y-cp3x-cp3xm_winxx.whl

   #. Repeat the process for all packages which installation failed before re-installing :program:`GeophPy`.

Uninstallation
==============

The Python package can simply be uninstalled using ``pip``:

    >>> pip uninstall geophpy
    >>> or
    >>>  python -m pip uninstall geophpy
