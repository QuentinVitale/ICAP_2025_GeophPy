.. _chap-mag-proc-geophpy:

Magnetic Processing
*******************

All the available processing techniques specific to magnetic surveys can be apply to a dataset through a simple command line of the form:

    >>> dataset.ProcessingTechnique(option1=10, option2=True, option3='relative',...)

This section describes the different available processing specific to magnetic survey data.

.. _chap-mag-proc-genconsider-geophpy:

General consideration 
=====================

Spectral domain
---------------

As most of the computation are done in the spectral domain, classic pre-processing and post-processing operations are automatically performed:

*Filling gaps*

To use the Fast Fourier Transform algorithm, the dataset must not contain gaps or 'NaN' (Not A Number) values. 
If the data set grid is not interpolated and contains NaNs, the blanks will be automatically filled using profile by profile linear interpolation prior to the Fourier Transform. 
After the data filtering, the gaps will be automatically unfilled using NaNs.

*Apodization*

To limit Gibbs phenomenon at jump discontinuities, the dataset edges are extended and then smoothed using a cosine window. 
The factor of apodization (0, 5, 10, 15, 20 or 25%) gives the dataset extension size. 
Values in the extension will be attenuated using a cosine:

.. image:: _static/figApodisation.png	

After processing, only the filtered values in the original dataset extension are kept so that the original data size is respected.

*Directional derivatives computation*

The nth directional derivatives of the a potential field are computed in the spectral domain using [BLAK96]_:

.. math::

   \mathcal{F} \left[ \frac{\partial^n T}{\partial x^n} \right] = (ik_x)^n \mathcal{F} \left[ T \right], 
   \mathcal{F} \left[ \frac{\partial^n T}{\partial y^n} \right] = (ik_y)^n \mathcal{F} \left[ T \right], 
   \mathcal{F} \left[ \frac{\partial^n T}{\partial z^n} \right] = |k|^n \mathcal{F} \left[ T \right].

Magnetic field
--------------

*System of coordinates*

.. figure:: _static/figAnglesRepresentation.png
    :height: 6cm
    :align: center

    Reduction to the Pole - Magnetic inclination (:math:`I`), declination (:math:`D`) and azimuth (:math:`\theta`) definition. 

*Ambient field vector components*

Vector components

.. math::

   B_x &= |B| \cdot f_x\\
   B_y &= |B| \cdot f_y\\
   B_z &= |B| \cdot f_z


where are :math:`f = (f_x, f_y, f_z)` is the unit-vector in the direction of the ambient field defined as:

.. math::

   f_x &=  \cos(I) \cos(\phi)\\
   f_y &= \cos(I) \sin(\phi) \\
   f_z &= \sin(I)


Logarithmic transformation
==========================

The logarithmic transformation is contrast enhancement filter.

Originally used for geological magnetic data, it enhances information present in the data at low-amplitude values while preserving the relative amplitude information via logarithmic transformation procedure [MPBL01]_.

**Filter applications:** *(magnetic) contrast enhancement*

Examples
--------

    >>> dataset.logtransform(multfactor=1e3, setnan=False, valfilt=False)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figMagLogtransfrom1.png         | .. figure:: _static/figMagLogtransfrom2.png                      |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Log Transform - Raw dataset.                     |    Log Transform - Filtered dataset.                             |
+-----------------------------------------------------+------------------------------------------------------------------+

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figMagLogtransfromHist1.png     | .. figure:: _static/figMagLogtransfromHist2.png                  |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Log Transform - Raw dataset histogram.           |    Log Transform - Filtered dataset  histogram.                  |
+-----------------------------------------------------+------------------------------------------------------------------+

Principle
---------

Contrast in magnetic susceptibility and magnetic remanence are the physical rock properties that controls magnetic anomalies, as well as the geometry and the position of the source body. 

Magnetic mineral content variation and borehole magnetic susceptibility are know to be best represented by a log-normal distribution. 
Assuming a that a similar distribution can represent lithologies on magnetic anomaly maps, then log transformation of the magnetic data should serve to normalize the distribution and highlight features having common amplitude.

The log-normal transformation of magnetic data :math:`f` is defined as [MPBL01]_:

.. math::

    \mathcal{F}\{f\} = \left\{
      \begin{array}{ccc}
          -\log_{10} (-f) & \mbox{for} & f <-1 \\
          \log_{10} (f) & \mbox{for} & f >1 \\
               0         & \mbox{for} & -1< f >1 \\
      \end{array}
   \right.

A multiplying factor (``multfactor``) can be used to increase/decrease the number of data that falls into the condition :math:`-1< f >1`, i.e. the number of data that are nulled.

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
   * - multfactor
     - Multiplying factor that can be used to increase/decrease the number of data that falls into the condition [-1< f >1].
     - float
     - x5, x10, x20, x100...
   * - setnan
     - Flag to replace values by NaNs instead of zeros.
     - bool
     - ``True`` or ``False``
   * - valfilt
     - Flag to apply filter on the ungridded data values rather than on the gridded data.
     - bool
     - ``True`` or ``False``


.. _chap-mag-proc-rdp-geophpy:

Pole reduction
==============

Classic reduction to the pole.

The reduction to the magnetic pole is a way to facilitate magnetic data interpretation and comparison. 
It simulates the anomaly that would be measured at the north magnetic pole (inclination of the magnetic field is maximum, i.e. vertical) [BLAK96]_.

**Filter applications:** *(magnetic) ease anomaly interpretation*

Examples
--------

    >>> dataset.polereduction(apod=5, inclination=65, declination=0, azimuth=10)

Principle
---------

Due to the dipolar nature of the geomagnetic field, magnetic anomalies (if not located at the magnetic poles) are asymmetric with a geometry that depends on the ambient magnetic field inclination (:math:`I`). 

The filter symmetrize the anomalies and place them directly above the source.
In other words, the reduction to the pole simulates the anomaly that would be measured at the north magnetic pole. 
A similar processing (reduction to the equator) is used when data are recored at low magnetic inclinations.

The computation is done in the spectral (frequency) domain using Fast Fourier Transform.
For magnetization and ambient field uniform throughout the study area the transformation is given in the spectral domain by [BLAK96]_:

.. math::

  \mathcal{F}_{RTP} = \mathcal{F}_{TF} \cdot \mathcal{F}

where :math:`\mathcal{F}_{RTP}` is the Fourier Transform of the anomaly reduced to the pole, 
:math:`\mathcal{F}_{TF}` is the Fourier Transform of the measured `total field` anomaly and 
:math:`\mathcal{F}` is the pole reduction operator defined as:

.. math::

  \mathcal{F}_{m,f}\{k_x, k_y\} = \frac{|k|^2}{
      a_1 k_x^2
      + a_2 k_y^2
      + a_3 k_x k_y
      + i |k| (b1 k_x + b2 k_y)}, |k| \ne 0

with

.. math::

  a_1 &= m_z f_z - m_x f_x, \\
  a_2 &= m_z f_z - m_y f_y, \\
  a_3 &= -m_y f_x - m_x f_y, \\
  b_1 &= m_x f_z + m_z f_x, \\
  b_2 &= m_y f_z + m_z f_y, \\

where :math:`m=(m_x, m_y, m_z)` is the unit-vector in the direction of the total magnetization (remanant + induced) of the source;
:math:`f = (f_x, f_y, f_z)` is the unit-vector in the direction of the ambient field;
:math:`|k| = \sqrt{k_x^2 + k_y^2}` is the radial wavenumber and
:math:`k_x` and :math:`k_y` are the wavenumber in the x and y-direction respectively.

The remanent magnetization of the source is usually small compared to the induced magnetization and can be neglected. 
The total magnetization is hence purely inductive so :math:`m=(m_x, m_y, m_z) = f = (f_x, f_y, f_z)` and the previous equation simplifies in:

.. math::

  \mathcal{F}_{f}\{k_x, k_y\} = \frac{|k|^2}{
      \left[ |k| f_z
      + i( k_x f_x + k_y f_y ) \right]^2}, |k| \ne 0

By default, the remanent magnetization is neglected in the filter (``incl_mag=None``, ``decl_mag=None``).

Algorithm
---------

.. only:: html

   .. figure:: _static/figAlgoPoleReduction.png
      :align: center
   
      Reduction to the Pole - Reduction to the Pole Algorithm. 

      Reduction to the Pole algorithm where 
      :math:`k_x` and :math:`k_y` are the wavenumbers corresponding to the x and y-direction respectively; 
      :math:`I` is the ambient magnetic field inclination; 
      :math:`D` is the ambient magnetic field declination;
      :math:`\phi` is the survey x-axis azimuth.

.. only:: latex

   .. figure:: _static/figAlgoPoleReduction.pdf
      :align: center
   
      Reduction to the Pole - Reduction to the Pole Algorithm. 

      Reduction to the Pole algorithm where 
      :math:`k_x` and :math:`k_y` are the wavenumbers corresponding to the x and y-direction respectively; 
      :math:`I` is the ambient magnetic field inclination; 
      :math:`D` is the ambient magnetic field declination;
      :math:`\phi` is the survey x-axis azimuth.

See :ref:`chap-mag-proc-genconsider-geophpy` for the definition of filling gaps and apodization methods.

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
     - Apodization factor (in %), to limit Gibbs phenomenon at jump discontinuities.
     - float
     - 0, 5, 10, 20, 25, ...
   * - inclination
     - Magnetic inclination (:math:`I`, in degree). The angle between the horizontal plane and the ambient magnetic field vector (positive downward).
     - float
     - 65, 45 ...
   * - declination
     - Magnetic declination (:math:`D`, in degree). The angle between the geographic ('true') North and Magnetic North (the horizontal projection of the ambient magnetic field vector, positive eastward).
     - float
     -
   * - azimuth
     - Survey profiles azimuth (:math:`\phi`, in degree). The angle between the profile direction and the Geographic North (positive east of north).
     - float
     -
   * - magazimuth
     - Survey profiles magnetic azimuth (:math:`\alpha=D-\phi`, in degree). The angle between the profile direction and the Magnetic North (positive east of north).
     - float
     - 0, 10, 30..., ``None``

.. _chap-mag-proc-continu-geophpy:

Continuation
============

Upward or downward continuation of potential field data (magnetic or gravimetric).

The filter computes the data that would be measured at an upper (`upward continuation`) or lower (`downward continuation`) survey altitude. 
The computation is done in the spectral (frequency) domain using the Fast Fourier Transform  [BLAK96]_.

**Filter applications:** *(magnetic) ease anomaly interpretation, background removal, data smoothing, merging surveys at different altitudes*

Examples
--------

    >>> continuation(apod=0, configuration='TotalField', distance=+0.5)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figMagContinuation1.png         | .. figure:: _static/figMagContinuation2.png                      |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Continuation - Raw dataset.                      |    Continuation - Filtered dataset.                              |
+-----------------------------------------------------+------------------------------------------------------------------+

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figMagContinuationHist1.png     | .. figure:: _static/figMagContinuationHist2.png                  |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Continuation - Raw dataset histogram.            |    Continuation - Filtered dataset  histogram.                   |
+-----------------------------------------------------+------------------------------------------------------------------+

Principle
---------

The (upward or downward) continuation transforms the potential field measured at one altitude to the field that would be measured at another altitude, 
farther (`upward continuation`) or closer (`downward continuation`) from the magnetic sources. 

Assuming that all the magnetic sources are located below the observation surface, 
the continuation at a new observation altitude :math:`z` 
of a survey acquired at an original altitude :math:`z_0` 
is given in the spectral domain by [BLAK96]_:

.. math::

    \mathcal{F}_{\Delta z,k} = \mathcal{F}_{TF} \cdot e^{-\Delta z |k|}

where:

* :math:`\mathcal{F}_{TF}` is the Fourier Transform of the measured data at the original altitude of observation :math:`z_0`; 
* :math:`\mathcal{F}_{\Delta z,k}` is the Fourier Transform of the anomaly at the new altitude of observation :math:`z = z_0 - \Delta z`; 
* :math:`\Delta z = z_0 - z` is the altitude increase between the original and new altitude of observation; 
* :math:`|k| = \sqrt{k_x^2 + k_y^2}` is the radial wavenumber where :math:`k_x` and :math:`k_y` are the wavenumber in the x and y-direction respectively.

The given  altitude increase (:math:`\Delta z`) is an algebraic value:

* If :math:`\Delta z>0`, the new altitude of observation is above the original altitude: the operation is an `upward continuation`; 
* if :math:`\Delta z<0`, the new altitude of observation is below the original altitude: the operation is a `downward continuation`.

If the sensor configuration is in total-field vertical gradient, the data can be transformed to total-field data (``totalfieldconversionflag=True``) using :math:`\bigg( 1-e^{\Delta s|k|} \bigg)^{-1}` (see :ref:`chap-mag-proc-magconfigconv-geophpy`).

The `upward continuation` attenuates anomalies with respect to the wavelength in way that accentuates anomalies caused by deep sources and  attenuates at the anomalies caused by shallow sources. 
It is hence a smoothing operator.

The `downward continuation` accentuates the shallowest components. It reduces spread of anomalies and corrects anomalies coalescences. 
It is useful to discriminates the number of body source at the origin of a one big anomaly. 
It is an unsmoothing operator that is instable as small changes in the data can cause large and unrealistic variations so it is to be used with caution. 
Low-pass filtering before the `downward continuation` can be a solution to increase the filter stability.

.. note:: Algorithm unstability

     Unlike the `upward continuation`, the `downward continuation` is unstable process as it is an unsmoothing process that 
     tends to accentuate small changes in the shallow components. So "Any errors present and perhaps undetected in the measured data 
     may appear in the calculated field as large and unrealistic variations." [BLAK96]_

Algorithm
---------

.. only:: html

   .. figure:: _static/figAlgoContinuation.png
      :align: center
   
      Continuation - Continuation Algorithm. 

.. only:: latex

   .. figure:: _static/figAlgoContinuation.pdf
      :align: center
   
      Continuation - Continuation Algorithm.

See :ref:`chap-mag-proc-genconsider-geophpy` for the definition of filling gaps and apodization methods.

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
   * - distance
     - Continuation distance (in m). Positive for upward continuation and negative for downward continuation.
     - float
     - 0.5, 2, 10, ...
   * - totalfieldconversionflag
     - Flag to proceed to the conversion to total-field data after continuation
     - bool
     - ``True``, ``False``
   * - separation
     - Sensor separation if ``totalfieldconversionflag`` is ``True``.
     - float
     - 0.7, 1, ...

Analytic signal
===============

Computes the 3-D Analytic Signal.

The Analytic Signal (also known as the total gradient magnitude or energy envelope) is a way to ease magnetic source characterization independently from the direction of its magnetization.

**Filter applications:** *magnetic source depth estimation*

Examples
--------

    >>> dataset.analyticsignal(apod=0)

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figMagAnalyticSignal1.png       | .. figure:: _static/figMagAnalyticSignal2.png                    |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Analytic Signal - Raw dataset.                   |    Analytic Signal - Filtered dataset.                           |
+-----------------------------------------------------+------------------------------------------------------------------+

+-----------------------------------------------------+------------------------------------------------------------------+
| .. figure:: _static/figMagAnalyticSignalHist1.png   | .. figure:: _static/figMagAnalyticSignalHist2.png                |
|    :height: 6cm                                     |    :height: 6cm                                                  |
|    :align: center                                   |    :align: center                                                |
|                                                     |                                                                  |
|    Analytic Signal - Raw dataset histogram.         |    Analytic Signal - Filtered dataset  histogram.                |
+-----------------------------------------------------+------------------------------------------------------------------+

Principle
---------

For a magnetization and an ambient field uniform throughout the study area, the amplitude of the analytic signal (or total gradient magnitude or energy envelope) of a **potential field** anomaly :math:`T` is given by [RoVP92]_:

.. math::

   |A(x, y, z)| = \sqrt{ \left( \frac{\partial T}{\partial x} \right)^2
                    + \left( \frac{\partial T}{\partial y} \right)^2
                    + \left( \frac{\partial T}{\partial z} \right)^2 }

The directional derivatives are computes in the spectral domain (see :ref:`chap-mag-proc-genconsider-geophpy`) and transformed back to the spatial domain using an inverse Fourier Transform to computed the analytic signal amplitude. 

Algorithm
---------

.. only:: html

   .. figure:: _static/figAlgoAnalyticSignal.png
      :align: center
   
      Analytic Signal - Analytic Signal Algorithm. 

.. only:: latex

   .. figure:: _static/figAlgoAnalyticSignal.pdf
      :align: center
   
      Analytic Signal - Analytic Signal Algorithm. 

See :ref:`chap-mag-proc-genconsider-geophpy` for the definition of filling gaps and apodization methods.

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

Euler deconvolution
===================

Classic Euler deconvolution.

Euler deconvolution is a method to estimate the depth of magnetic sources that do not required reduced-to-the-pole data.

Examples
--------

**Manual target picking**

    >>> target_extents = [[15,25,10,25], [35,42,8,15]]  # [[xmin, xmax, ymin, xmax],...]
    >>> results = dataset.eulerdeconvolution(apod=0, structind=3, window=target_extents)

    +-----------------------------------------------------+------------------------------------------------------------------+
    | .. figure:: _static/figMagEulerDeconvolution1.png   | .. figure:: _static/figMagEulerDeconvolutionDepth1.png           |
    |    :height: 6cm                                     |    :height: 6cm                                                  |
    |    :align: center                                   |    :align: center                                                |
    |                                                     |                                                                  |
    |    Euler Deconvolution - Manual target picking.     |    Euler Deconvolution - Manual target picking                   |
    +-----------------------------------------------------+------------------------------------------------------------------+

**Sliding window target picking**

    >>> # Sliding window of xstep*xstep samples
    >>> results = dataset.eulerdeconvolution(apod=0, structind=3, xstep=10)

    +-----------------------------------------------------+------------------------------------------------------------------+
    | .. figure:: _static/figMagEulerDeconvolution2.png   | .. figure:: _static/figMagEulerDeconvolutionDepth2.png           |
    |    :height: 6cm                                     |    :height: 6cm                                                  |
    |    :align: center                                   |    :align: center                                                |
    |                                                     |                                                                  |
    |    Euler Deconvolution - Manual target picking.     |    Euler Deconvolution - Manual target picking                   |
    +-----------------------------------------------------+------------------------------------------------------------------+

**Displaying results**

    The :meth:`~geophpy.dataset.DataSet.eulerdeconvolution` method returns the results of the Euler deconvolution as a list containing 
    the source position, the associated structural index, the residual of the least-square estimation and the used dataset sub-window extent:

    >>> print(results[0])
    [20.078, 17.80, 0.99, 3, 0.94343, 15.0, 25.0, 10.0, 25.0]

    The sub-window extents and source positions can be display onto the dataset using the following commands sequence:

    >>> # Importing the sub-window-to-rectangle conversion tool
    >>> import geophpy.plotting.plot as gplt

    >>> # Extracting source positions and sub-windows extents
    >>> sources = np.asarray(results)[:,:2].tolist()
    >>> window_extents = np.asarray(results)[:,5:].tolist()

    >>> # Converting sub-windows extents to rectangles
    >>> rects = gplt.extents2rectangles(window_extents)

    >>> # Plotting results
    >>> dataset.plot(plottype='2D-SURFACE', rects=rects, points=sources)

Principle
---------

Euler deconvolution is a method to estimate position and depth of magnetic sources. 
It is based on the assumption that magnetic field `is homogeneous of degree N`, 
so that the total magnetic field and its originating magnetic source are related by the following equation [RAGM90]_:

.. math::

   N(B-T) = (x - x_0) \frac{\partial T}{\partial x} 
      + (y - y_0) \frac{\partial T}{\partial y}
      + (z - z_0) \frac{\partial T}{\partial z} 

where 
:math:`(x_0, y_0, z_0)` is the position of a source; 
:math:`T` is the total-field detected at :math:`(x, y, z)`; 
:math:`B` is the regional value of the field and 
:math:`N` is a non-negative integer known as the structural index (or degree of homogeneity). 

For a set of :math:`n` observation points of the total-field, the equation can be reformulated as a linear system of equations and expressed in matrix form as:

.. math::

   &\begin{pmatrix}
      \frac{\partial T_1}{\partial x}  & \frac{\partial T_1}{\partial y} & \frac{\partial T_1}{\partial z} &  N\\    
                  \vdots            &              \vdots             &              \vdots         &  \vdots \\
      \frac{\partial T_i}{\partial x}  & \frac{\partial T_i}{\partial y} & \frac{\partial T_i}{\partial z} &  N\\   
                  \vdots            &              \vdots             &              \vdots         &  \vdots \\
      \frac{\partial T_n}{\partial x}  & \frac{\partial T_n}{\partial y} & \frac{\partial T_n}{\partial z} &  N\\
      \end{pmatrix}
   \begin{pmatrix}
      x_0 \\
      y_0 \\
      z_0 \\
       B
   \end{pmatrix}
   =
   \begin{pmatrix}
      x_1 \frac{\partial T_1}{\partial x}
         + y_1 \frac{\partial T_1}{\partial y}
         + z_1 \frac{\partial T_1}{\partial z}
         + NT_1 \\
      \vdots \\
      x_n \frac{\partial T_i}{\partial x}
         + y_i \frac{\partial T_i}{\partial y}
         + z_i  \frac{\partial T_i}{\partial z}
         + NT_i \\
      \vdots \\
      x_n \frac{\partial T_n}{\partial x}
         + y_n \frac{\partial T_n}{\partial y}
         + z_n  \frac{\partial T_n}{\partial z}
         + NT_n
   \end{pmatrix} \\
   &i = 1,...,n

where :math:`T_i = T(x_i, y_i, z_i)` is the `ith` observation of the total-field anomaly at the coordinates :math:`(x_i, y_i, z_i)`.
  
This system of has the form :math:`\mathbf{A}\mathbf{x}=\mathbf{b}` and can be solved in least-square sense in the overdetermined case, 
leading to least-square estimates of :math:`(x_0, y_0, z_0, B)`.


*Directional derivatives computation*

The directional derivatives are computes in the spectral domain (see :ref:`chap-mag-proc-genconsider-geophpy`) and transformed back to the spatial domain using an inverse Fourier Transform.

*Structural Index (SI)*

The structural index is a measure of the rate of change with distance of a field. 
It is an integer value that depends on the source model geometry and is "not a tuning parameter" [ReEW14]_. 
Valid structural index values are referenced in the following table.

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0
   :align: center

   * - Source Model
     -
     - SI
   * - Point, sphere
     -
     -  3
   * - Line, cylinder, thin bed fault
     -
     - 2
   * - Thin sheet edge, thin sill, thin dike
     -
     - 1
   * - Contact of infinite depth extent
     -
     - 0

A wrong value of the SI will lead to errors in the depth estimation of the source: a too high value will yield over-estimated depths and vice versa.

.. note:: Structural Index estimation

   Although it is not recommended, the structural index can be estimated in least-square sense as a parameter of the linear system by setting ``structind=None``. 

   For now, this estimation may results in non-realistic (non integer) values for the structural index. In the future it will be constrained to select only allowed (integer) values.

*Set of observation points*

The set of observation points can be:

* specified manually by giving a set of windows extent (xmin, xmax, ymin and xmax for each window of interest),
* or automatically by using a sliding windows of a given size (``xstep`` * ``ystep``).

Algorithm
---------

... TBD ...

See :ref:`chap-mag-proc-genconsider-geophpy` for the definition of filling gaps and apodization methods.

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
   * - structuind
     - Structural Index, depends on the source geometry.
     - int
     - 0, 1, 2, 3 or ``None``
   * - windows
     - List containing the satial extent of each sub-window to be considered for the deconvolution 
     - list (of float)
     - [15,25,10,25], [35,42,8,15], ...] or ``None``
   * - xstep
     - Size, in number of sample, of the sub-windows in the x-direction. Only if ``windows=None``.
     - float
     - 10, 33, ... or ``None``
   * - ystep
     - Size, in number of sample, of the sub-windows in the y-direction. Only if ``windows=None``.
     - float
     - 10, 33, ... or ``None``

.. _chap-mag-proc-magconfigconv-geophpy:

Sensor configuration conversion
===============================

Conversion between the different sensor's configurations.

Magnetic data are all derived from the same potential and thereby contains in the same information, making theoretical conversion from one sensor configuration to another is possible.

Examples
--------

Principle
---------

Du to the "potential field nature" of the magnetic field, the same information is theoretically contained in the measure of the total-field anomaly, the total-field gradient and the vertical component of the anomaly. 
Conversion from one sensor configuration to another is theoretically possible even in practice the (inevitable) measurement noise can make it difficult.

*Sensor's configurations*

Different sensor's configurations exist to conduct magnetic surveys. Three most commonly used are the *total-field*, the *total-field vertical gradient* and the *fluxgate* sensor's configurations :

* **Total-field magnetometers** measure the magnitude of the total magnetic field regardless of the magnetic vector direction. They give a *scalar measure* of the total strength of an ambient magnetic field at any given point and are hence sometimes referred to as *scalar magnetometers* [AsGA08]_. 

* **Total-field gradiometers** simply refers to *total-field magnetometers* in *vertical gradient configuration*, *i.e* two total-field sensors one on top of the other.

* **Fluxgate magnetometers** measure the component of the ambient magnetic field in a particular direction (the sensor's axis only) and are hence sometimes referred to as *vector magnetometers* [AsGA08]_. *Fluxgate gradiometers* are composed of two sensors one on top of the other and measure directly the difference of the vertical component of the ambient magnetic field.

*Conversion between configurations*

A simple linear transformation combining a :ref:`chap-mag-proc-continu-geophpy` and a subtraction allow the conversion between the different sensor's configuration. 
For an ambient field uniform throughout the study area the transformation of the different sensor's configurations is defined in the spectral domain by [TaDD97]_:

**Total-field and total-field vertical gradient**
   Conversion of a measure obtained using a *total-field magnetometer* to the measure that would be obtained using a *total-field magnetometer in vertical gradient configuration* (i.e. with two sensors, one on top of the ohers) is defined in the spectral domain as:

   .. math::

      e^{-\Delta h |k|} - e^{-(\Delta h+\Delta s) |k|}

   The reverse transformation (from total-field vertical gradient to total-field) can simply be computed using the inverse expression: 
   
   .. math::

      \Big( e^{-\Delta h |k|} - e^{-(\Delta h+\Delta s) |k|} \Big)^{-1}

**Total-field vertical gradient and fluxgate**
   Conversion of a measure obtained using a *total-field magnetometer in vertical gradient configuration* to the measure that would be obtained using a *fluxgate gradiometer* is defined in the spectral domain as:

   .. math::

      \frac{|k|}{ |k| f_y + i(k_x f_x + k_y f_y) }

   The reverse transformation (from fluxgate vertical gradient to total-field vertical gradient) can simply be computed using the inverse expression: 

   .. math::

      \bigg( \frac{|k|}{ |k| f_y + i(k_x f_x + k_y f_y) } \bigg)^{-1}

**Total-field and fluxgate**
   Conversion of a measure obtained using a *total-field magnetometer* to the measure that would be obtained using a *fluxgate gradiometer* is defined in the spectral domain as the combination of the previous transformation :

   .. math::

      \Big( e^{-\Delta h |k|} - e^{-(\Delta h+\Delta s) |k|} \Big)
      \cdot
      \frac{|k|}{
                 |k| f_y + i(k_x f_x + k_y f_y)
                 }

   The reverse transformation (from fluxgate vertical gradient to total-field) can simply be computed using the inverse expression:

   .. math::

      \Bigg(
            \Big( e^{-\Delta h |k|} - e^{-(\Delta h+\Delta s) |k|} \Big)
            \cdot
            \frac{|k|}{
                      |k| f_y + i(k_x f_x + k_y f_y)
                      }
      \Bigg)^{-1}

Where

* :math:`\Delta h` is the difference of altitude between the bottom sensors in the initial and final configuration; 
* :math:`\Delta s` is the difference of altitude between top and bottom sensors in gradient configuration; 
* :math:`|k| = \sqrt{k_x^2 + k_y^2}` is the radial wavenumber with :math:`k_x` and :math:`k_y` are the wavenumbers in the x and y-direction respectively; 
* :math:`I` is the ambient magnetic field inclination; 
* :math:`D` is the ambient magnetic field declination;
* :math:`\phi` is the survey x-axis azimuth;
* :math:`f_x = \cos(I) \cdot cos(D-\phi)` is the unit-vector in the x-direction of the ambient field; 
* :math:`f_y = \cos(I) \cdot sin(D-\phi)` is the unit-vector in the y-direction of the ambient field;
* :math:`f_z = \sin(I)` is the unit-vector in the z-direction of the ambient field.

Algorithm
---------

See :ref:`chap-mag-proc-genconsider-geophpy` for the definition of filling gaps and apodization methods.

Parameters
----------

Equivalent Susceptibility
=========================

Examples
--------

... TBD ...

Principle
---------

... TBD ...

Algorithm
---------

... TBD ...

See :ref:`chap-mag-proc-genconsider-geophpy` for the definition of filling gaps and apodization methods.

Parameters
----------

... TBD ...