# -*- coding: utf-8 -*-
'''
    geophpy.plotting.colormap
    -------------------------

    Module regrouping color map managing functions.

    :copyright: Copyright 2014-2019 Lionel Darras, Philippe Marty, Quentin Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import matplotlib.pyplot as plt
import numpy as np
import os

import geophpy.core.utils as gputils
import matplotlib._cm as cm

# list of available colormaps
##cmap_list = ['Blues',
##             'BrBG',
##             'BuGn',
##             'BuPu',
##             'CMRmap',
##             'GnBu',
##             'Greens',
##             'Greys',
##             'OrRd',
##             'Oranges',
##             'PRGn',
##             'PiYG',
##             'PuBu',
##             'PuOr',
##             'PuRd',
##             'Purples',
##             'RdBu',
##             'RdGy',
##             'RdPu',
##             'RdYlBu',
##             'RdYlGn',
##             'Reds',
##             'Spectral',
##             'Wistia',
##             'YlGn',
##             'YlGnBu',
##             'YlOrBr',
##             'YlOrRd',
##             'afmhot',
##             'autumn',
##             'binary',
##             'bone',
##             'bwr',
##             'copper',
##             'gist_earth',
##             'gist_gray',
##             'gist_heat',
##             'gist_yarg',
##             'gnuplot',
##             'gray',
##             'hot',
##             'hsv',
##             'jet',
##             'ocean',
##             'pink',
##             'spectral',
##             'terrain']

cmap_list = [name for name in cm.datad if not name.endswith("_r")]  # all available colormaps from matplotlib


# list of available colormaps icons
colormap_script_path = os.path.dirname(os.path.abspath(__file__))
colormap_icon_path = os.path.join(colormap_script_path, 'colormapicons')

#cmap_icon_list = [name for name in os.listdir(colormap_icon_path) if os.path.isfile(os.path.join(colormap_icon_path, name)) and if name.endswith('.png')]
cmap_icon_list = [name for name in os.listdir(colormap_icon_path) if os.path.isfile(os.path.join(colormap_icon_path, name))]
cmap_icon_list = [name for name in cmap_icon_list if name.endswith('.png')]
#cmap_icon_list = [name for name in colormap_icon_list if name.endswith('.png')

##cmap_icon_list = ['CMAP_Blues.png', 'CMAP_BrBG.png', 'CMAP_BuGn.png', 'CMAP_BuPu.png', 'CMAP_CMRmap.png', 'CMAP_GnBu.png', 
##                  'CMAP_Greens.png', 'CMAP_Greys.png', 'CMAP_OrRd.png', 'CMAP_Oranges.png', 'CMAP_PRGn.png', 'CMAP_PiYG.png',
##                  'CMAP_PuBu.png', 'CMAP_PuOr.png', 'CMAP_PuRd.png', 'CMAP_Purples.png' ,'CMAP_RdBu.png' ,'CMAP_RdGy.png',
##                  'CMAP_RdPu.png', 'CMAP_RdYlBu.png', 'CMAP_RdYlGn.png' ,'CMAP_Reds.png' ,'CMAP_Spectral.png' ,'CMAP_Wistia.png',
##                  'CMAP_YlGn.png', 'CMAP_YlGnBu.png', 'CMAP_YlOrBr.png' ,'CMAP_YlOrRd.png' ,'CMAP_afmhot.png' ,'CMAP_autumn.png',
##                  'CMAP_binary.png', 'CMAP_bone.png', 'CMAP_bwr.png', 'CMAP_copper.png', 'CMAP_gist_earth.png', 'CMAP_gist_gray.png',
##                  'CMAP_gist_heat.png', 'CMAP_gist_yarg.png', 'CMAP_gnuplot.png', 'CMAP_gray.png', 'CMAP_hot.png', 'CMAP_hsv.png',
##                  'CMAP_jet.png', 'CMAP_ocean.png', 'CMAP_pink.png', 'CMAP_spectrals.png', 'CMAP_terrain.png']

def getlist(sort=True):
   '''Return the available colormaps list. '''

   colormapslist = [m for m in plt.cm.datad if not m.endswith("_r")]
   # Sorting ingoring reverse and sring case
   if sort:
      colormapslist = sorted(colormapslist, key = str.lower)

   # Removing the 'spectral' cmap (is an alias of 'nipy_spectral' that causes problem on Windows
   ## if writing png icon on the same directory 'Spectral.png' and 'spectral.png'
   try:
      colormapslist.remove('spectral')
   except:
      pass

   return colormapslist


def get_icon_path():
    ''' Return the colormap icon path. '''

    colormapiconspath = colormap_icon_path

    return colormapiconspath


def get_icon_list():
    ''' Return the colormap icon list. '''

#    colormapiconslist = sorted(name for name in cmap_icon_list)
    colormapiconslist = sorted(cmap_icon_list, key = lambda x:  x.split("_")[1].lower())

    #colormapiconslist = cmap_icon_list

    return colormapiconslist


def make_icon(cmapname, path='', creversed=False, dpi=None):
    ''' Make an icon of the given colormap name.

    cf. :meth:`~geophpy.dataset.colormap_make_icon

    '''

    cmapname = gputils.make_sequence(cmapname)

    for cmname in cmapname:
        if creversed:
           cmname = cmname + '_r'

        # alias for backward compatibility
        ## On Windows cant have a 'Spectral.png' and 'spectral.png' file in the same directory.
        if 'spectral' in cmname:
           cmname = 'nipy_spectral'

        name = 'CMAP_'+ cmname + '.png'
        filename = os.path.join(path, name)

        fig = plt.figure( figsize=(1,0.1))
        plot(cmname, fig=fig, creversed=creversed, filename=filename, dpi=dpi)
        plt.close(fig)  # closing to prevent mpl from opening too much fig

    return


def plot(cmname, creversed=False, fig=None, filename=None, dpi=None, transparent=False):
   '''
   Plot the colormap.
    
   Parameters :

   :cmname: Name of the colormap, 'gray_r' for example.

   :creversed: True to add '_r' at the cmname to reverse the color map

   :filename: Name of the color map file to save, None if no file to save. 
    
   :dpi: 'dot per inch' definition of the picture file if filename is not None

   :transparent: True to manage the transparency.

   Returns : figure and color map plot object.
   '''
   
   # Levels from 0 to 255
   gradient = np.linspace(0, 1, 256)
   gradient = np.vstack((gradient, gradient))

   # Figure for display
   if fig is None:
      fig = plt.figure( figsize=(1,0.1))
   else :
      fig.clf()

   ax = fig.add_subplot(1,1,1)
   fig.subplots_adjust(top=0.99, bottom=0.01, left=0.001, right=0.999)

   if creversed:
      cmname = cmname + '_r'

   ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmname))
   ax.set_axis_off()

   if filename is not None:
      plt.savefig(filename, dpi=dpi, transparent=transparent)

   return fig
