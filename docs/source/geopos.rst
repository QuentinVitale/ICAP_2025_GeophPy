.. _chap-geopos-geophpy:

Geographic Positionning
***********************

All geographic position information are manage via a :class:`~geophpy.geoposset.GeoPosSet` class.

You can open a file containing the geographic coordinates of the geophysical survey corresponding to a :class:`~geophpy.dataset.DataSet` object from both :

* an ascii file** (*.csv* for instance)
* or a shapefile (*.shp*)

**Reading positions from file**

To open your a :class:`~geophpy.geoposset.GeoPosSet` from a file (ASCII or shapefile) use:

    >>> from geophpy.geoposset import GeoPosSet
    
    >>> # file type determined from extension
    >>> gpos = GeoPosSet.from_file('GCPs.txt')

    >>> # file type forced manually
    >>> gpos = GeoPosSet.from_file('GCPs.shp', filetype='shapefile', )	

**Retrieve position**

Retrieve list of Ground Control Points via:

    >>> gcp_list = gpos.points_list 
    >>> # or
    >>> gcp_list = gpos.GCPs()

    >>> print(gcp_list)
    >>> [[0, 32.52, 34.70], [1, 32.52, 34.70]]	# with [num, x or lon, y or lat]

**Plot GCPs**

Plot the Ground Control Points:

    >>> fig = gpos.plot(long_label=False)
    >>> fig.show()

.. image:: _static/figGps1.png
   :width: 50%
   :align: center

**Saving GCPs**

Ground Control Points can be saved as an ascii file or a kml file:

    >>> # saving as an ascii file
    >>> gpos.to_ascii("MyGCPs.dat", delimiter=',')

To be saved as a kml file, the coordinates must be converted into lat,long first. 
If the original geographic system is UTM you can directly do:

    >>> # Converting UTM to WGS84
    >>> import geophpy.geoposset as pset
    >>> num, east, north, xlocal, ylocal = gpos.GCPs().T
    >>> lt, lg = pset.utm_to_wgs84(east, north, 32,'T')
    >>> gcp_wgs84 = np.stack((num, xlocal, ylocal, lt, lg)).T

    >>> # saving as kml
    >>> gpos_conv = pset.GeoPosSet(refsystem='WGS84', points_list=gcp_wgs84)
    >>> gpos.to_kml("ConvGCPs.kml")

.. image:: _static/figGps2.png
   :width: 50%
   :align: center

**Dataset georeferencing**

Dataset georeferencing is possible with at less 4 points.

    >>> # Georeferencing ungridded values
    >>> dataset.setgeoref('UTM', gpos.points_list, 'T', 32)

    >>> # Georeferencing gridded values
    >>> dataset.interpolate(x_step=0.5, y_step=0.25)
    >>> dataset.setgeoref('UTM', gpos.points_list, 'T', 32)

Plotting georeferenced data position (example for ungridded values)

    >>> import geophpy.dataset as dset
    >>> import geophpy.geoposset as pset

    >>> # Reading files
    >>> success, dataset = dset.DataSet.from_file('Mag_ex1.dat')
    >>> success, gpos = pset.GeoPosSet.from_file('GPS_ex1.csv')

    >>> # Displaying GCps
    >>> gpos.plot(long_label=True)

    +-------------------------------------------------+
    | .. figure:: _static/figGeorefGCPs.png           |    
    |    :height: 6cm                                 |
    |    :align: center                               |
    |                                                 |
    |    Georeferencing - Ground Control Points.      |
    +-------------------------------------------------+

    >>> # Georeferencing
    >>> dataset.setgeoref(gpos.refsystem, gpos.points_list, 'T', 32)

    >>> # Displaying in local system
    >>> box = np.stack(dataset.get_boundingbox())  # bounding box corners
    >>> x, y = dataset.get_xyvalues()  # data sample position

    >>> box_gcp = gpos.points_list.T[1:3].T  # georef bounding box corners
    >>>  east, north = dataset.get_georef_xyvalues() # data sample georef position
    
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, y, 'r.', markersize=0.25) # or east, north
    >>> ax.plot(box.T[0], box.T[1], 'bo') # or box_gcp.T[0], box_gcp.T[1]
    >>> ax.set_aspect('equal')

    >>> # adding GCPs label
    >>> for point in  box:
    >>>     strg = '(%s, %s)' % (point[0], point[1])
    >>>     dx, dy = 3, 0
    >>>     plt.text(point[0]+dx, point[1]+dy, strg,
    >>>              bbox=dict(fc='white',ec='none', alpha=0.5))

    +-------------------------------------------------+---------------------------------------------------+
    | .. figure:: _static/figGeorefLocalSystem.png    | .. figure:: _static/figGeorefRefSystem.png        |     
    |    :height: 6cm                                 |    :height: 6cm                                   |
    |    :align: center                               |    :align: center                                 |
    |                                                 |                                                   |
    |    Georeferencing - Dataset in local system.    |    Georeferencing - Dataset in geographic system. |
    +-------------------------------------------------+---------------------------------------------------+

With the data set georeferenced, it is possible to export the dataset as a kml file:

    >>> dataset.to_kml('2D-SURFACE', 'gray_r', "prospection.kml",
	cmmin=-10, cmmax=10, dpi=600)

.. image:: _static/figGeoref1.png
   :width: 50%
   :align: center

Exporting the data set as a raster in a SIG application (as ArcGis, QGis, Grass, ...) is possible with severals picture file format ('jpg', 'png', 'tiff'):

    >>> dataset.to_raster('2D-SURFACE', 'gray_r', "prospection.png",
	cmmin=-10, cmmax=10, dpi=600)

.. image:: _static/figGeoref2.png
   :width: 50%
   :align: center

A world file containing positioning informations of the raster is created ('jgw' for JPG, 'pgw' dor.png, and 'tfw' for TIFF picture format) with:

    Line 1: A: pixel size in the x-direction in map units/pixel

    Line 2: D: rotation about y-axis

    Line 3: B: rotation about x-axis

    Line 4: E: pixel size in the y-direction in map units, almost always negative[3]

    Line 5: C: x-coordinate of the center of the upper left pixel

    Line 6: F: y-coordinate of the center of the upper left pixel

Example:

    0.0062202177595

    -0.0190627320737

    0.0131914192417

    0.00860610262817

    660197.8178

    3599813.97056

**ASCII GCPs file format**

    If the `geographic reference system` is known, it must be written on the first line. 
    The other lines contain the `point number` followed by its `GPS coordinates` (`Longitude` and `Latitude` or `Easting` and `Northing`)
    and eventually the corresponding local `X`, `Y` -coordinates.

    >>> # WGS84 file without local coordinates
    >>> # (delimiter is tabulation)
    WGS84		
    1	66.84617533	37.74956917
    2	66.84649517	37.7489535
    3	66.8472475	37.74972867
    4	66.84689417	37.7491385
    5	66.84691867	37.7491025
    ...

    >>> # UTM file can also contain the corresponding local coordinates
    >>> # (delimiter is tabulation)
    UTM
    1	745038.191	4656005.727	150	0	
    2	745068.172	4656045.663	150	50	
    3	745028.43	4656076.057	100	50	
    4	744988.466	4656105.978	50	50	
    5	744998.428	4656036.093	100	0	
    ...

    >>> # UTM file with zone letter and number
    >>> # (delimiter is tabulation)
    UTM 	L	32
    1	745038.191	4656005.727	150	0	
    2	745068.172	4656045.663	150	50	
    3	745028.43	4656076.057	100	50	
    4	744988.466	4656105.978	50	50	
    5	744998.428	4656036.093	100	0	
    ...

    >>> # Unknown geographic system with missing local position
    >>> # (delimiter is ";")
    1;745038.191;4656005.727;150;0	
    2;745068.172;4656045.663;150;50	
    3;745028.43;4656076.057;;        # missing local position
    4;744988.466;4656105.978;50;50	
    5;744998.428;4656036.093;100;0	
    ...
