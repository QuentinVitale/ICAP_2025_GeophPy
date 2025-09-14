Changelog
*********

Version 0.33
==============

Released on 2022-XX-XX.


Version 0.32.2
==============

Released on 2021-12-02.

* Documentation update.

Version 0.32
============

Released on 2019-08-01.

* HTML documentation theme changed to Read the Docs theme.
* Reading Georeferencing Ground Control Points from CSV as been extended to any delimiter-value file.
* Georeferencing now works ungridded dataset values.
* Forced equal aspect ratio for map plots.
* Added Dataset linear/polynomial detrending of dataset profile.
* Constant destriping now works on ungridded dataset values.
* Changed the euler deconvolution implementation (now allow automatic sub-windowing).
* Changed the magnetic continuation implementation.
* Changed the analytic signal implementation.
* Added import/export from/to Surfer grids.
* Changed the reduction to the pole implementation.
* Added directional filter (plough filter).
* Added 2D-Fourier spectral plot.
* Added setmin/setmax range possibility for correlation calculation in festoon filter.
* Uptaded Dataset mean cross-track plot.
* Added Dataset zero-mean/zero-median profile filter (works on both gridded and ungridded dataset values). 
* Enhanced CSV opening file method for both dataset and geoposset.
* Added Datasets merging & edge matcing methods.
* Added Dataset translation & rotation.
* Added Label display option for 2-D plots.
* Added Scatter plot of raw values.
* Added number of levels or levels values specification for 2-D contour and filled contour plots.
* Fixed Filled 2-D contour plot bug.

Version 0.31
============

Released on 2018-01-02.

* Fixed GeophPy pip installation issues and updated documentation.

Version 0.30
============

Released on 2017-12-01.

* Updated GeophPy documentation.
* Added options for constant destriping filter.
* Added Mean cross-track profile plot for destripping filters.
* Implemented Wallis filter.
* Implemented replace by profile's median in peak filtering.
* Added automatic delimiter search in delimited files.
* Fixed reading delimited file issues.
* Added non uniform shift for Festoon.
* Fixed correlation and cross-correlation map calculation.
* Added color map for histogram plot.
* Fixed bug in histo.plot.

Version 0.21
============

Released on 2016-05-01

* Initial version.
