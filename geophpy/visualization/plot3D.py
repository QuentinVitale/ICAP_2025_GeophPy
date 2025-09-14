# -*- coding: utf-8 -*-
'''
    geophpy.plotting.plot3D
    -----------------------

    Map Plotting in 3D dimensions.

    :copyright: Copyright 2014 Lionel Darras, Philippe Marty and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from geophpy.core.utils import *

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def plot_surface(dataset, cmapname, creversed=False, fig=None, filename=None, cmmin=None, cmmax=None, interpolation='bilinear', cmapdisplay=True, axisdisplay=True, dpi=600, transparent=False, logscale=False):
   """
   plotting in 2D-dimensions a surface
   """

   if (creversed == True):             # if reverse flag is True
      cmapname = cmapname + '_r'       # adds '_r' at the colormap name to use the reversed colormap

   cmap = plt.get_cmap(cmapname)       # gets the colormap
   
   if (fig == None) :                  # if first display
      fig = plt.figure()               # creates fthe figure
   else :                              # if not first display
      fig.clf()                        # clears figure
      
   ax = fig.add_subplot(111, projection='3d')

   x_decimal_maxnb = getdecimalsnb(dataset.info.x_gridding_delta)
   y_decimal_maxnb = getdecimalsnb(dataset.info.y_gridding_delta)
   
   Z = dataset.data.z_image
   xi = np.linspace(dataset.info.x_min, dataset.info.x_max, 1 + (np.around((dataset.info.x_max-dataset.info.x_min)/dataset.info.x_gridding_delta)), endpoint = True)
   yi = np.linspace(dataset.info.y_min, dataset.info.y_max, 1 + (np.around((dataset.info.y_max-dataset.info.y_min)/dataset.info.y_gridding_delta)), endpoint = True)
   xi, yi = np.around(xi, decimals=x_decimal_maxnb), np.around(yi, decimals=y_decimal_maxnb)

   X, Y = np.meshgrid(xi, yi)                          # builds the (X, Y) gridded matrix
   
   if (axisdisplay == True):           # displays the axis if required
      plt.xlabel(dataset.data.fields[0])
      plt.ylabel(dataset.data.fields[1])
      plt.title(dataset.data.fields[2])
      ax.get_xaxis().set_visible(True)
      ax.get_yaxis().set_visible(True)
   else:
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

   if (logscale == True):
      norm = colors.LogNorm()
   else:
      norm = colors.Normalize()
                                       # builds the image
#   im = ax.imshow(Z, extent=(dataset.info.x_min,dataset.info.x_max,dataset.info.y_min,dataset.info.y_max), origin='lower', interpolation=interpolation, cmap=cmap, vmin=cmmin, vmax=cmmax, norm=norm)
   im = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, vmin=cmmin, vmax=cmmax, norm=norm)

   if (cmapdisplay == True):           # displays the color bar if required
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.1)
      colormap = fig.colorbar(im,cax=cax)
   else:
      colormap = None

   ax.set_xlim(dataset.info.x_min, dataset.info.x_max)
   ax.set_ylim(dataset.info.y_min, dataset.info.y_max)

   if (filename != None):
      if ((axisdisplay == False) and (cmapdisplay == False)):        # the plotting has to be displayed in the full picture         
         plt.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches='tight', pad_inches=0)
      else:
         plt.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches='tight')

   fig.canvas.draw()
   return fig, colormap

