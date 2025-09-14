.. _chap-gen-proc-geophpy:

General Processing
******************

All the available processing techniques can be apply to a dataset through a simple command line of the form:

    >>> dataset.ProcessingTechnique(option1=10, option2=True, option3='relative',...)

The available `general processing` techniques (i.e. not specific to a particular geophysical method) are listed in the sections below.

Peak filtering
==============

Replace peaks in the dataset.

Peaks are detected using an *Hampel filter* or a *decision-theoretic median filter* and replaced by either NaNs or the local median value.

Examples
--------

* replacing peaks using an *Hampel filter* to detect outliers:

    >>> # Peaks are replaced by local median
    >>> dataset.peakfilt(method='hampel', halfwidth=5, threshold=3)

    >>> # or by NaNs
    >>> dataset.peakfilt(method='hampel', halfwidth=5, threshold=3, setnan=True)

* replacing peaks using *decision-theoretic median filter*  to detect outliers:

    >>> # The threshold is a percentage of the local median value
    >>> dataset.peakfilt(method='median', halfwidth=5, threshold=.05, mode='relative')

    >>> # or a in raw unit
    >>> dataset.peakfilt(method='median', halfwidth=5, threshold=15, mode='absolute')

Principle
---------

A centered 1-D window is slided along a flattened version of the dataset and determines if the central element of the moving window is an outlier (peak) and has to be replaced. 

To dertermine outliers, the median of the elements in the moving window is computed and used as a reference value. 
If the deviation of the central element of the window to the local median is above a threshold, the central value is replaced by the local median (or NaNs depending on the filter configuration ``setnan={True|False}``). 
Two different filters are implemented to dertermine outliers (a median filter and an Hampel filter).

*Threshold value*

For the median filter, the *threshold* can be defined as a percentage of the local median (``mode='relative'``) or directly in raw units (``mode='absolute'``).

For the Hampel filter, the higher the value of the *threshold* is, the less the filter will be selective. 
A threshold value of 0 is equivalent to a *standard median filter* (each element replaced by the value of the local median). 

*Filters definition*

For :math:`f_{k}` the central element of a moving window :math:`\{ f_{k-K}, ..., f_{k}, ..., f_{k+K}\}` of half-with :math:`K`  and of local median :math:`f^{\dagger} = \mbox{median}\{f_{k-K}, ..., f_{k}, ..., f_{k+K}\}`

The Hampel filter is defined as [PeGa16]_ :
   .. math::

      \mathcal{H}_{K,t}\{f_k\} =  
      \begin{cases} 
      f^{\dagger} & \mbox{if } |f_{k} - f^{\dagger}| > t \cdot Sk, \\ 
      f_{k} & \mbox{otherwise},
      \end{cases}

   .. math::

      Sk = 1.4826 \cdot \mbox{median}\{|f_{k-K} - f^{\dagger}|, ..., |f_{k} - f^{\dagger}|, ..., |f_{k+K} - f^{\dagger}|\}

The (decision-theoretic) median filter is defined as:
   .. math::

      \mathcal{M}_{K,t}\{f_k\} =  
      \begin{cases} 
      f^{\dagger} & \mbox{if } |f_{k} - f^{\dagger}| > t, \\ 
      f_{k} & \mbox{otherwise},
      \end{cases}

where is :math:`t` the filter threshold.

Parameters
----------

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 1
   :align: center

   * - Name
     - Description
     - Type
     - Value
   * - method
     - Type of the `decision-theoretic filter` used to determine outliers.
     - str
     - 'median', 'hampel'
   * - halfwidth
     - Filter half-width
     - int
     - 5, 10 20, ...
   * - threshold
     - Filter threshold parameter. If t=0 and method='hampel', it is equal to a `standard median filter`.
     - int
     - 0, 1, 2, 3, ...
   * - mode
     - Median filter mode. If 'relative', the threshold is a percentage of the local median value. If 'absolute', the threshold is a value.
     - str
     - 'relative', 'absolute'`
   * - setnan
     - Flag to replace outliers by NaNs instead of the local median.
     - bool
     - ``True`` or ``False``
   * - valfilt
     - [For future implementation] Flag to apply filter on the ungridded data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``

See :ref:`chap-hlvl-api-geophpy` for calling details.

Thresholding
============

Dataset thresholding in a given interval.

Examples
--------

    >>> # Replacing out of range values by lower and upper bounds
    >>> dataset.threshold(setmin=-10, setmax=10)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figThreshold2.png               | .. figure:: _static/figThresholdHisto2.png                       |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Peak filter - Min, max thresholding - dataset.   |    Peak filter - Min, max thresholding - histogram.              |
+-----------------------------------------------------+------------------------------------------------------------------+

    >>> # by NaNs
    >>> dataset.threshold(setmin=-10, setmax=10, setnan=True)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figThreshold3.png               | .. figure:: _static/figThresholdHisto4.png                       |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Thresholding - NaN thresholding - dataset.       |    Thresholding - NaN thresholding - histogram.                  |
+-----------------------------------------------------+------------------------------------------------------------------+

    >>> # or by each profile's median
    >>> dataset.threshold(setmin=-10, setmax=10, setmed=True)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figThreshold4.png               | .. figure:: _static/figThresholdHisto4.png                       |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Thresholding - Median thresholding - dataset.    |    Thresholding - Median thresholding - histogram.               |
+-----------------------------------------------------+------------------------------------------------------------------+

Principle
---------

Each value of the dataset are compared to the given interval bounds. 
Depending on the filter configuration, values outside of the interval will be replaced by the interval bounds (``setmin=valmin``, ``setmax=valmax``), 
NaNs (``setnan=True``), 
or the profile's median (``setmed=True``).

Parameters
----------

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 1
   :align: center

   * - Name
     - Description
     - Type
     - Value
   * - setmin
     - Minimal interval value. All values lower than ``setmin`` will be replaced by ``setmin`` (if both ``setmed`` and ``setnan`` are ``False``).
     - float
     - -5, 10, 42.5, ...
   * - setmax
     - Maximal interval value. All values lower than ``setmax`` will be replaced by ``setmax`` (if both ``setmed`` and ``setnan`` are ``False``).
     - float
     - -5, 10, 42.5, ...
   * - setmed
     - Flag to replace out of bound data by the profile's median.
     - bool
     - ``True`` or ``False``
   * - setnan
     - Flag to replace out of bound data by NaNs.
     - bool
     - ``True`` or ``False``
   * - valfilt
     - Flag to apply filter on the ungridded data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``

See :ref:`chap-hlvl-api-geophpy` for calling details.

Median filtering
================

Apply a median filter (*decision-theoretic* or *standard*) to the dataset.

Examples
--------

    >>> # No threshold : standard median filter
    >>> dataset.medianfilt(nx=3, ny=3)

+------------------------------------------------+--------------------------------------------------------+
| .. figure:: _static/figMedianFilter1.png       | .. figure:: _static/figMedianFilter2.png               |
|    :height: 6cm                                |    :height: 6cm                                        |
|    :align: center                              |    :align: center                                      |
|                                                |                                                        |
|    Median filter - Raw dataset (no threshold). |    Median filter - Filtered dataset (no threshold).    |
+------------------------------------------------+--------------------------------------------------------+

    >>> # Threshold in raw unit : decision-theoretic median filter
    >>> dataset.medianfilt(nx=3, ny=3, gap=5)

+---------------------------------------------------------+--------------------------------------------------------------+
| .. figure:: _static/figMedianFilter1.png                | .. figure:: _static/figMedianFilter3.png                     |
|    :height: 6cm                                         |    :height: 6cm                                              |
|    :align: center                                       |    :align: center                                            |
|                                                         |                                                              |
|    Median filter - Raw dataset (threshold in raw unit). |    Median filter - Filtered dataset (threshold in raw unit). |
+---------------------------------------------------------+--------------------------------------------------------------+

Principle
---------

"Median filtering is a non linear process useful in reducing impulsive, or salt-and-pepper noise" [LimJ90]_. 
It is capable of smoothing a few out of bounds pixels while preserving image's discontinuities without affecting the other pixels.


For each pixel in the dataset, the local median of the (**nx** x **ny**) neighboring points is calculated. 

..  image:: _static/figMedianFilter.png
   :height: 5cm
   :align:  center

A threshold value is defined and if the deviation from the local median is higher than this threshold, then the center pixel value is replaced by the local median value. 
The threshold deviation from the local median can be defined: 

* in percentage (``percent=10``) or raw units (``gap=5``),
* if no threshold is given, all pixels are replaced by their local medians (*standard median filter*).

Parameters
----------

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 1
   :align: center

   * - Name
     - Description
     - Type
     - Value
   * - nx
     - Size, in number of sample, of the filer in the x-direction.
     - int
     - 5, 10, 25, ...
   * - ny
     - Size, in number of sample, of the filer in the y-direction.
     - int
     - 5, 10, 25, ...
   * - percent
     - Threshold deviation (in percents) to the local median value (for absolute field measurements).
     - float
     -  
   * - gap
     - Threshold deviation (in raw units) to the median value (for relative anomaly measurements).
     - float
     -  
   * - valfilt
     - [For future implementation] Flag to apply filter on the ungridded data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``

See :ref:`chap-hlvl-api-geophpy` for calling details.

.. _chap-gen-proc-festoon-geophpy:

Festoon filtering
=================

Dataset destaggering.

The festoon filter is a destaggering filter that reduces the positioning error along the survey profiles that result in a festoon-like effect.
An *optimum shift* is estimated based on the correlation of a particular profile and the mean of its two neighboring profiles.

This filter needs to be done as an early step in the processing flow as it needs to be done preferably before interpolation (or with no interpolation in the x-axis).

**Filter applications:** *data destaggering*

.. warning::
   
   The correlation map computation needs a regular grid WITHOUT interpolation between profiles (i.e. the x-axis direction).
   The correlation is done between each even profiles number and the mean of its two nearest neighbors. 
   In the case of an interpolation in the x-axis, an interpolated (even) profile will be equal (or quasi-equal) to the mean of its two nearest neighbors.
   The computed map will be close to an auto-correlation map, resulting in a null shift optimum shift.

Examples
--------

    >>> # Gridding data without interpolation
    >>> dataset.interpolate(interpolation='none')
    >>> 
    >>> # Uniform shift
    >>> dataset.festoonfilt(method='Crosscorr', uniformshift=True)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figFestoonFilter1.png           | .. figure:: _static/figFestoonFilter2.png                        |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Destaggering - Raw dataset (uniform shift).      |    Destaggering - Filtered dataset (uniform shift).              |
+-----------------------------------------------------+------------------------------------------------------------------+

    >>> # Non uniform shift
    >>> dataset.festoonfilt(method='Crosscorr', uniformshift=False)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figFestoonFilter1.png           | .. figure:: _static/figFestoonFilter3.png                        |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Destaggering - Raw dataset (non uniform shift).  |    Destaggering - Filtered dataset (non uniform shift).          |
+-----------------------------------------------------+------------------------------------------------------------------+

Principle
---------

For every even profiles (columns) in the dataset, an optimum shift is estimated based on the correlation of the profile and the mean of its two nearest neighbors. 
For each possible shift, a correlation map is hence computed and used to estimate the optimum shift (max of correlation).

The optimum shift can be set to be uniform throughout the map (``uniformshift=True``) or different for each profile (``uniformshift=False``). 
If the shift is set uniform, the mean correlation profile is used as correlation map.

At the top and bottom edges of the correlation map (high shift values), high correlation values can arrise from low sample correlation calculation. 
To prevent those high correlation values to drag the best shift estimation, a limitation is set to only consider correlation with at least 50% overlap between profiles. 
Similarly, a minimum correlation value (``corrmin``) can be defined to prevent profile's shift if the correlation is too low.

    >>> # Correaltion map 
    >>> dataset.correlation_plotmap()
    
    >>> # Correaltion profile
    >>> dataset.correlation_plotsum() 

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figFestoonFilterCorrMap.png     | .. figure:: _static/figFestoonFilterCorrSum.png                  |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Destaggering - Correlation map.                  |    Destaggering - Mean correlation profile.                      |
+-----------------------------------------------------+------------------------------------------------------------------+

Parameters
----------

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 1
   :align: center

   * - Name
     - Description
     - Type
     - Value
   * - method
     - Correlation method to use to compute the correlation coefficient in the correlation map.
     - str
     - 'Crosscorr', 'Pearson', 'Spearman' or 'Kendall'
   * - shift
     - Optional shift value (in pixels) to apply to the dataset profiles. If shit=0, the shift will be determined for each profile by correlation with neighbors. 
       If shift is a vector each value in shift will be applied to its corresponding even profile. 
       In that case shift must have the same size as the number of even profiles.  
     - int or array of int
     - 3, 5, 10, [2, 3, 4, ..., 3, 4] or 0
   * - corrmin
     - Minimum correlation coefficient value to allow shifting.
     - float (in the range [0-1])
     -  0.6, 0.8
   * - uniformshift
     - Flag to use a uniform shift on the map or a different one for each profile.
     - bool
     - ``True`` or ``False``
   * - setmin
     - Data values lower than ``setmin`` are ignored.
     - float
     - 12, 44.5, ..., ``None``
   * - setmax
     - Data values higher than ``setmin`` are ignored.
     - float
     - 12, 44.5, ..., ``None``
   * - valfilt
     - [For future implementation] Flag to apply filter on the ungridded data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``

See :ref:`chap-hlvl-api-geophpy` for calling details.

Detrending
==========

... To Be Developed ...

Subtracting a polynomial fit for each profile in the dataset.


Regional trend filtering
========================

... To Be Developed ...

Remove the background (or regional response) from a dataset to highlight the sub-surface features of interest.

Example
-------

Principle
---------

Parameters
----------

See :ref:`chap-hlvl-api-geophpy` for calling details.


Wallis filtering
================

The Wallis filter is a locally adaptative contrast enhancement filter.

Based on the local statistical properties of sub-window in the image, 
it adjusts brightness values (grayscale image) in the local window so that the local mean and standard deviation match target values.

**Filter applications:** *contrast enhancement*

Examples
--------

    >>> dataset.wallisfilt()

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figWallisFilter1.png            | .. figure:: _static/figWallisFilter2.png                         |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Wallis Filter - Raw dataset.                     |    Wallis Filter - Filtered dataset.                             |
+-----------------------------------------------------+------------------------------------------------------------------+

Principle
---------

A window of size (**nx**, **ny**) is slided along the image and at each pixel the Wallis operator is calculated. 
The Wallis operator is defined as  [STHH90]_:

.. math::

   \frac{A \sigma_d}{A \sigma_{(x,y)} + \sigma_d} [f_{(x,y)} - m_{(x,y)}] + \alpha m_d + (1 - \alpha)m_{(x,y)}

where:
    * :math:`A` is the amplification factor for contrast, 
    * :math:`\sigma_d` is the target standard deviation, 
    * :math:`\sigma_{(x,y)}` is the standard deviation in the current window, 
    * :math:`f_{(x,y)}` is the center pixel of the current window, 
    * :math:`m_{(x,y)}` is the mean of the current window, 
    * :math:`\alpha` is the edge factor (controlling portion of the observed mean, and brightness locally to reduce or increase the total range), 
    * :math:`m_d` is the target mean.

As the Wallis filter is design for grayscale image, the data are internally converted to brightness level before applying the filter. 
The conversion is based on the minimum and maximum value in the dataset and uses 256 levels (from 0 to 255). 
The filtered brightness level are converted back to data afterwards.

A quite large window is recommended to ensure algorithm stability.

Parameters
----------

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 1
   :align: center

   * - Name
     - Description
     - Type
     - Value
   * - nx
     - Size, in number of sample, of the filer in the x-direction.
     - int
     - 5, 10, 25, ...
   * - ny
     - Size, in number of sample, of the filer in the y-direction.
     - int
     - 5, 10, 25, ...
   * - targmean
     - The target standard deviation (in gray level).
     - int
     - 125
   * - setgain
     - Amplification factor for contrast.
     - int
     - 8
   * - limitstdev
     - imitation on the window standard deviation to prevent too high gain value if data are dispersed.
     - int
     - 25
   * - edgefactor
     - Brightness forcing factor (:math:`\alpha`), controls ratio of edge to background intensities.
     - float (in the range of [0,1])
     -  
   * - valfilt
     - [For future implementation] Flag to apply filter on the scattered data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``


See :ref:`chap-hlvl-api-geophpy` for calling details.

Ploughing filtering
===================

Directional filter.

Apply a directional filter to reduce agricultural ploughing effect in the dataset (or any other directional feature).


Examples
--------

    >>> dataset.ploughfilt(apod=0, azimuth=90, cutoff=100, width=3)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figPloughFilter1.png            | .. figure:: _static/figPloughFilter2.png                         |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Plough Filter - Raw dataset.                     |    Plough Filter - Filtered dataset.                             |
+-----------------------------------------------------+------------------------------------------------------------------+

    >>> # Raw dataset spectral plot  
    >>> dataset.spectral_plotmap(plottype='magnitude', logscale=True)
    >>> dataset.plot_directional_filter(apod=0, azimuth=90, cutoff=100, width=3)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figPloughFilter3.png            | .. figure:: _static/figPloughFilter4.png                         |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Plough Filter - Raw magnitude spectrum.          |    Plough Filter - Directional filter.                           |
+-----------------------------------------------------+------------------------------------------------------------------+

Principle
---------

The directional feature in the dataset is filtered in the spectral domain using the combination (:math:`\mathcal{F}`) of 
a *gaussian low-pass filter* of order 2 (:math:`\mathcal{F}_{GLP}`) and 
a *gaussian directional filter* (:math:`\mathcal{F}_{DIR}`) defined as [TABB01]_ :

.. math::

   \mathcal{F}(\rho, \theta, f_c) &= \mathcal{F}_{GLP}(\rho, f_c) \ast \mathcal{F}_{DIR}(\rho, \theta) \\

                                  &= e^{-(\rho / f_c)^2}          \ast ( 1-e^{-\rho^2 / \tan(\theta-\theta_0)^n} ) 

where:
    * :math:`\rho` and :math:`\theta` are the current point polar coordinates, 
    * :math:`f_c` is the gaussian low-pass filter cutoff frequency, 
    * :math:`\theta_0` is the directional filter's azimuth, 
    * :math:`n` is the parameter that controls the filter width.

The filter's width is determined by the value of :math:`n` (:numref:`figPloughFiltn=2`, :numref:`figPloughFiltn=3` and :numref:`figPloughFiltn=4`) and 
the *gaussian low-pass filter* component is neglected if no cutoff frequency is defined (``cutoff=None``, see :numref:`figPloughFiltfc=none`)

>>> dataset.plot_directional_filter(azimuth=30, cutoff=50, width=2)
>>> dataset.plot_directional_filter(azimuth=30, cutoff=50, width=3)

+------------------------------------------------------+----------------------------------------------------------------+
| .. _figPloughFiltn=2:                                | .. _figPloughFiltn=3:                                          |
|                                                      |                                                                |  
| .. figure:: _static/figPloughFilter5.png             | .. figure:: _static/figPloughFilter6.png                       |
|    :height: 6cm                                      |    :height: 6cm                                                |
|    :align: center                                    |    :align: center                                              |
|                                                      |                                                                |
|    Plough Filter - Directional Filter (**n=2**)      |    Plough Filter - Directional Filter (**n=3**)                |
+------------------------------------------------------+----------------------------------------------------------------+

>>> dataset.plot_directional_filter(azimuth=30, cutoff=50, width=4)
>>> dataset.plot_directional_filter(azimuth=30, cutoff=None, width=4)

+------------------------------------------------------+----------------------------------------------------------------+
| .. _figPloughFiltn=4:                                | .. _figPloughFiltfc=none:                                      |
|                                                      |                                                                |  
| .. figure:: _static/figPloughFilter7.png             | .. figure:: _static/figPloughFilter8.png                       |
|    :height: 6cm                                      |    :height: 6cm                                                |
|    :align: center                                    |    :align: center                                              |
|                                                      |                                                                |
|    Plough Filter - Directional Filter (**n=4**)      |    Plough Filter - Pure directional filter (**fc=None**)       |
+------------------------------------------------------+----------------------------------------------------------------+

Parameters
----------

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 1
   :align: center

   * - Name
     - Description
     - Type
     - Value
   * - apod
     - Apodization factor (%), to limit Gibbs phenomenon at jump discontinuities.
     - float
     - 0, 5, 10, 20, 25, ...
   * - azimuth
     - Filter azimuth in degree.
     - float
     - 0, 10, 33.25, ...
   * - cutoff
     - cutoff spatial frequency (in number of sample).
     - int
     - 5, 10, 100, ... or ``None``
   * - width
     - ilter width parameter.
     - int
     - 2, 3, 4, ...
   * - valfilt
     - [For future implementation] Flag to apply filter on the scattered data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``

See :ref:`chap-hlvl-api-geophpy` for calling details.

Zero-Mean Traversing
====================

Subtracts the mean (or median) of each traverse (profile) in the dataset.

**Filter applications:** *(magnetic) probe compensation, data destriping, edge matching...*

Examples
--------

    >>> # Raw data display
    >>> dataset.plot(plottype='2D-SURFACE')[0].show()
    >>> dataset.histo_plot(cmapdisplay=False, coloredhisto=True).show()

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figZeroMeanFilter1.png          | .. figure:: _static/figZeroMeanFilterHisto1.png                  |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Zero-mean traverse - Raw dataset.                |    Zero-mean traverse - Raw  histogram.                          |
+-----------------------------------------------------+------------------------------------------------------------------+

    >>> # Zero-Mean Traverse data
    >>> dataset.zeromeanprofile(setvar='mean')

+--------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figZeroMeanFilter2.png             | .. figure:: _static/figZeroMeanFilterHisto2.png                  |
|    :height: 6cm                                        |    :height: 6cm                                                  |
|    :align: center                                      |    :align: center                                                |
|                                                        |                                                                  |
|    Zero-mean traverse - Filtered dataset (zero-mean).  |    Zero-mean traverse - Filtered histogram  (zero-mean).         |
+--------------------------------------------------------+------------------------------------------------------------------+

    >>> # Zero-Median Traverse data
    >>> dataset.zeromeanprofile(setvar='median')

+----------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figZeroMeanFilter3.png               | .. figure:: _static/figZeroMeanFilterHisto3.png                  |
|    :height: 6cm                                          |    :height: 6cm                                                  |
|    :align: center                                        |    :align: center                                                |
|                                                          |                                                                  |
|    Zero-mean traverse - Filtered dataset (zero-median).  |    Zero-mean traverse - Filtered histogram  (zero-median).       |
+----------------------------------------------------------+------------------------------------------------------------------+

Principle
---------

For each traverse (profile) in the dataset, the mean (or median) is calculated and subtracted, leading to a zero-mean (or zero-median) survey [AsGA08]_.

.. note::

    This filter is strictly equivalent to the constant destriping filter in configuration *'mono'* sensor, using *'additive'* destriping method and *Nprof=0*:

    >>> dataset.zeromeanprofile(setvar='median')
    >>> dataset.destripecon(Nprof=0, method='additive', config='mono', reference='median')


Parameters
----------

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 1
   :align: center

   * - Name
     - Description
     - Type
     - Value
   * - setvar
     - Profile's statistical property be subtracted from each profile.
     - str
     - 'mean' or 'median'
   * - setmin
     - Data values lower than ``setmin`` are ignored.
     - float
     - 12, 44.5, ..., ``None``
   * - setmax
     - Data values higher than ``setmin`` are ignored.
     - float
     - 12, 44.5, ..., ``None``
   * - valfilt
     - [For future implementation] Flag to apply filter on the scattered data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``

See :ref:`chap-hlvl-api-geophpy` for calling details.

.. _chap-gen-proc-destipcon-geophpy:

Constant destriping
===================

Remove the strip noise effect from the dataset. 

The strip noise effect arises from profile-to-profile differences in sensor height, orientation, drift or sensitivity (multi-sensors array).
Constant destriping is done using Moment Matching method [GaCs00]_. 

**Filter applications:** *(magnetic) probe compensation, data destriping, edge matching...*

Examples
--------

    >>> dataset.destripecon(Nprof=4, method='additive')

+----------------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figDestriping1.png                   | .. figure:: _static/figDestriping2.png                           |
|    :height: 6cm                                          |    :height: 6cm                                                  |
|    :align: center                                        |    :align: center                                                |
|                                                          |                                                                  |
|    Destriping - Raw dataset.                             |    Destriping - Filtered dataset.                                |
+----------------------------------------------------------+------------------------------------------------------------------+

Principle
---------

In constant destriping, a linear relationship is assumed between profile-to-profile offset (means) and gain (standard deviation).
The statistical moments (mean :math:`m_i` and standard deviation :math:`\sigma_i`) of each profile in the dataset are computed and matched to reference values. 

**Reference values**

   Typical reference values are:

   * the mean (:math:`m_d`) and standard deviation (:math:`\sigma_d`) of the :math:`N` neighboring profiles (``Nprof=4``),
   * the mean and standard deviation of the global dataset (``Nprof='all'``),
   * the mean and standard deviation of each profiles (``Nprof=0``, zero-mean traverse),
   * alternatively, one can use the median and interquartile range instead of mean and standard deviation (``reference='median'``).

**Additive vs multiplicative destriping**

   The corrected value can be calculated by 

   * an additive relation (``method='additive'``) [RiJi06]_, [Scho07]_:

      .. math::
   
         f_{corr} = (f - m_i) \frac{\sigma_d}{\sigma_i} + m_d

   * or a multiplicative relation (``method='multiplicative'``): 

      .. math::

         f_{corr} = f \frac{\sigma_d}{\sigma_i} \frac{m_d}{m_i}

   where 
   :math:`f_{corr}` is the corrected value, 
   :math:`\sigma_d` is the reference standard deviation, 
   :math:`\sigma_i` is the current profile standard deviation, 
   :math:`f` is the current value,
   :math:`m_i` is the current profile and  
   :math:`m_d` is the reference mean.

   .. note::

       Ground surveys (unlike remote sensing) often demonstrate profile-to-profile offsets but rarely gain changes so that only matching profiles mean (or median) is usually appropriate (``configuration='mono'``):

       .. math::

          f_{corr} = f - m_i + m_d

       This is similar to the zero-mean filter, with the difference that the profile mean (or median) is set to a reference value instead of zero.
 
       If ``Nprof`` is set to zero for the reference computation, this is strictly equivalent to the zero-mean filter:
 
       >>> dataset.destripecon(Nprof=0, method='additive', config='mono', reference='mean')
       >>> # Strictly equals to 
       >>> dataset.zeromeanprofile(setvar='mean')

   *Summary of the destiping configurations*

   .. list-table:: 
      :header-rows: 1
      :widths: auto
      :stub-columns: 1
      :align: center

      * - Configuration / Method
        - ``'additive'``
        - ``'multiplicative'``
      * - ``'mono'``
        - :math:`f_{corr} = f - m_i + m_d`
        - :math:`f_{corr} = f \frac{m_d}{m_i}`
      * - ``'multi'``
        - :math:`f_{corr} = (f - m_i) \frac{\sigma_d}{\sigma_i} + m_d`
        - :math:`f_{corr} = f \frac{\sigma_d}{\sigma_i} \frac{m_d}{m_i}`

**Mean cross-track profile display**

   The data mean cross-track profile before and after destriping can be displayed as follow:

       >>> dataset.meantrack_plot(Nprof=4, method='additive', reference='median', plotflag='raw')
       >>> dataset.meantrack_plot(Nprof=4, method='additive', reference='median', plotflag='both')

   +----------------------------------------------------------+------------------------------------------------------------------+
   | .. figure:: _static/figDestriping3.png                   | .. figure:: _static/figDestriping4.png                           |
   |    :height: 6cm                                          |    :height: 6cm                                                  |
   |    :align: center                                        |    :align: center                                                |
   |                                                          |                                                                  |
   |    Destriping - mean cross-track profile (before).       |    Destriping - mean cross-track (after).                        |
   +----------------------------------------------------------+------------------------------------------------------------------+

Parameters
----------

.. list-table:: 
   :header-rows: 1
   :widths: auto
   :stub-columns: 1
   :align: center

   * - Name
     - Description
     - Type
     - Value
   * - Nprof
     - Number of neighboring profiles used to compute the the reference values.
     - int or 'all'
     - 0, 4 ,... or 'all'
   * - setmin
     - Data values lower than ``setmin`` are ignored.
     - float
     - 12, 44.5, ..., ``None``
   * - setmax
     - Data values higher than ``setmin`` are ignored.
     - float
     - 12, 44.5, ..., ``None``
   * - method
     - Destriping methode.
     - str
     - 'additive' or 'multiplicative'
   * - reference
     - References used for destriping.
     - str
     - 'mean' or 'median'
   * - config
     - Sensors configuration.
     - str
     - 'mono' or 'multi'
   * - valfilt
     - [For future implementation] Flag to apply filter on the scattered data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``

See :ref:`chap-hlvl-api-geophpy` for calling details.

Curve destriping
================

... To Be Developed ...

Remove from the dataset the strip noise effect by fitting and subtracting a MEAN polynomial curve to each profile on the dataset.
