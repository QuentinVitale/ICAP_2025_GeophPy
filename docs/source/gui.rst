.. _chap-gui-geophpy:

Using a GUI
***********

All plots in :program:`GeophPy` are actually made using `Matplotlib <https://matplotlib.org/>`_. 

To embed Matplotlib's plots in your own GUI, you have to select the proper backend:

   >>> import matplotlib
   >>> #for Qt4 and Pyside 
   >>> matplotlib.use('Qt4Agg')

   >>> #for Qt5 and Pyside2
   >>> matplotlib.use('Qt5Agg')

   >>> #for Tkinter
   >>> matplotlib.use('TkAgg')

As specified in the `SciPy Cookbook on PySide and Matplotlib <https://scipy-cookbook.readthedocs.io/items/Matplotlib_PySide.html>`_, 
"in case of problems for PySide, try to change the rcParam entry “backend.qt4” to "PySide" (e.g. by in the matplotlibrc file):
 
   >>> matplotlib.rcParams['backend.qt4']='PySide' 

.. warning::

   ``rcParams`` has been depreciated in recent matplotlib versions

.. note:: 

 * in Windows environment, this file ``matplotlibrc`` is in the "C:\\PythonXY\\Lib\\site-packages\\matplotlib\\mpl-data" directory.

 * in Linux environment, it is in the "/etc" directory.
 
You can also create widgets to embed Matplotlib's plot : 
see `Embedding Matplotlib in Qt <https://matplotlib.org/3.1.0/gallery/user_interfaces/embedding_in_qt_sgskip.html>`_  
and  
`Embedding Matplotlib in Tk <https://matplotlib.org/3.1.0/gallery/user_interfaces/embedding_in_tk_sgskip.html>`_ 
for complete examples.


You can also plot data in a windows with several color maps:

    >>> from geophpy.dataset import *
    >>>			# to get the list of available color maps 
    >>> list = colormap_getlist()	
    >>> first = True				# first plot
    >>> fig = None				# no previous figure
    >>> cmap = None				# no previous color map
    >>> success, dataset = DataSet.from_file("DE11.dat", delimiter=' ',
        z_colnum=5)
    >>> if (success == True):			# if file opened
    >>>			# for each color map name in the list
    >>>    for colormapname in list :		
    >>>        fig, cmap = dataset.plot('2D-SURFACE', 'gist_rainbow',
               dpi=600, axisdisplay=True, cmapdisplay=True, cmmin=-10,
               cmmax=10)
    >>>        if (first == True):		# if first plot
    >>>           fig.show()			# displays figure windows
    >>>           first = False			# one time only
    >>>			# updates the plot in the figure windows
    >>>        p.draw()				
    >>>			# removes it to display the next
    >>>        cmap.remove()		
    >>>			# waits 3 seconds before display the plot
    >>>			# with the next color map
    >>>        time.sleep(3)