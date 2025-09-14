.. _chap-about-geophpy:

About GeophPy
*************

Introduction
============

:program:`GeophPy` is a project initiated in 2014 through cooperation between two units of the CNRS [#]_ 
(`UMR5133-Archeorient`_ and `UMR7619-Metis <https://www.metis.upmc.fr/>`_).  
Since 2017, it is also developed by `Geo-Heritage <http://geoheritage.huma-num.fr/>`_ (a cooperation between `UMR5133-Archeorient`_  and `Eveha International <http://eveha-international.com/>`_).

:program:`GeophPy` is an open source `Python <https://www.python.org/>`_ package that offers tools for sub-surface geophysical survey data processing, in the field of archaeology, geology, and other sub-surface applications.

It mainly focuses on ground surveys data and offers tools to process the data and create geophysical maps that can be imported to GIS softwares.

:program:`GeophPy` contains `general tools`, 
such as data :ref:`destaggering <chap-gen-proc-festoon-geophpy>`, 
:ref:`destriping <chap-gen-proc-destipcon-geophpy>` 
and `method-specific tools`, 
such as :ref:`reduction to the pole <chap-mag-proc-rdp-geophpy>` 
or :ref:`magnetic data continuation <chap-mag-proc-continu-geophpy>`.

:program:`GeophPy` builds a geophysical :class:`~geophpy.dataset.DataSet` object composed by series of data in the format (X,Y,Z) with (X,Y) being the point position of the geophysical value Z, in order to process and/or display as maps of Z values.

This package can be used as command line, but a specific Graphical User Interface, `WuMapPy <https://pypi.org/project/WuMapPy/>`_, is designed for it. 

.. [#] French National Center for Scientific Research

.. _`UMR5133-Archeorient`: http://www.archeorient.mom.fr/recherche-et-activites/ressources-techniques/pole-2/

Features
========

* Building dataset from one or severals data files.
* Displaying geophysical maps in 2-D or 3-D.
* Processing datasets with general or method-specific geophysical filters.
* Export processed datasets into georeferenced format.
* Compatibility with Python 3.x.

Main authors
============

* **Lionel DARRAS**

 *CNRS, UMR5133-Archeorient, Lyon, France*  
  
 lionel.darras@mom.fr

* **Philippe MARTY** 
  
 *CNRS, UMR7619-Metis, Paris, France*

 philippe.marty@upmc.fr

* **Quentin VITALE**

 *Eveha International, Lyon, France*

 quentin.vitale@eveha.fr

License
=======

:program:`GeophPy` is developed on a `GNU GPL v3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_ license.

.. include:: ..\LICENSE