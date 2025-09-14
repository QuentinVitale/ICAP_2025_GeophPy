# -*- coding: utf-8 -*-
'''
    geophpy.plotting.spectral
    -------------------------

    Module regrouping Fourier transform plotting functions.

    :copyright: Copyright 2018-2019 Lionel Darras, Quentin Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable

import geophpy.core.operation as cop
import geophpy.core.processing as cproc

import numpy as np


#-----------------------------------------------------------------------------#
# Module's parameters definition                                              #
#-----------------------------------------------------------------------------#
# list of spectrum plots
SPECTRUM_PLOTTYPE_LIST = ['magnitude', 'logmagnitude', 'power', 'logpower', 'phase', 'real', 'imag']

def getspectrum_plottype_list():
    ''' Return the list of available spectrum plot types. '''
    
    return SPECTRUM_PLOTTYPE_LIST

def plotmeantrace():
    pass


def plotmap(dataset, fig=None, plottype='magnitude', filename=None, dpi=None, transparent=False, logscale=False, cmapdisplay=True, axisdisplay=True, rects=None, lines=None):
    '''Plot Dataset 2-D Fourier Transform.'''

    # Figure for the display ###################################################
    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()
      
    # Figure ax creation #######################################################
    ax = fig.add_subplot(111)

    # Data 2-D Fourier Transform computation ###################################
    Z = np.copy(dataset.data.z_image)
    nan_idx = np.asarray([])
    Z = cop.fillnanvalues(Z, indexout=nan_idx)
    ZTF = np.fft.fft2(Z)
    ZTF = np.fft.fftshift(ZTF)   # Centered Fourier Transform Spectrum

    nx = Z.shape[1]
    dx = dataset.info.x_gridding_delta
    fnyq_x = 1/(2*dx)
    u = np.fft.fftfreq(nx, d=dx)
    u = np.fft.fftshift(u)   # Centered frequencies

    ny = Z.shape[0]
    dy = dataset.info.y_gridding_delta
    fnyq_y = 1/(2*dy)
    v = np.fft.fftfreq(ny, d=dy)
    v = np.fft.fftshift(v)   # Centered frequencies

##    fmin = max(u.min(), v.min())
##    fmax = min(u.max(), v.max())

    fmin = -min(fnyq_x, fnyq_y)
    fmax = min(fnyq_x, fnyq_y)

    # Logarithmic scale ########################################################
    if logscale:
        norm = colors.LogNorm()
    else:
        norm = colors.Normalize()

    # Plot Type ################################################################
    if plottype in ['magnitude', 'amplitude']:
        Zdisp = np.abs(ZTF)/ZTF.size
        title = 'Magnitude Spectrum'
        cbartitle = '|F(u,v)|'

    elif plottype in ['logmagnitude', 'logamplitude']:
        Zdisp = np.log10(np.abs(ZTF)/ZTF.size)
        title = 'Magnitude Spectrum (Log)'
        cbartitle = 'Log(|F(u,v)|)'

    elif plottype == 'power':
        Zdisp = np.abs(ZTF)**2/ZTF.size
        title = 'Power Spectrum'
        cbartitle = '|F(u,v)|^2'

    elif plottype == 'logpower':
        Zdisp = np.log10(np.abs(ZTF)**2/ZTF.size)
        title = 'Power Spectrum (log)'
        cbartitle = 'Log(|F(u,v)|^2)'

    elif plottype == 'phase':
        Zdisp = np.unwrap(np.angle(ZTF))
        title = 'Phase Spectrum'
        norm = colors.Normalize()

    elif plottype == 'real':
        Zdisp = np.real(ZTF)
        title = 'Real Spectrum'
        norm = colors.Normalize()
        cbartitle = 'Re[F(u,v)]'

    elif plottype == 'imag':
        Zdisp = np.imag(ZTF)
        title = 'Imaginary Spectrum'
        norm = colors.Normalize()
        cbartitle = 'Im[F(u,v)]'

    else:
        Zdisp = np.imag(ZTF)
        Zdisp = np.abs(ZTF)
        title = 'Magnitude Spectrum'
        cbartitle = '|F(u,v)|'

    # Building image ###########################################################
    x0, x1 = u.min(), u.max()
    y0, y1 = v.min(), v.max()
    
    extent = (x0, x1, y0, y1)


##    im = ax.imshow(Z, extent=(dataset.info.x_min-(dataset.info.x_gridding_delta/2),
##                              dataset.info.x_max+(dataset.info.x_gridding_delta/2),
##                              dataset.info.y_min-(dataset.info.y_gridding_delta/2),
##                              dataset.info.y_max+(dataset.info.y_gridding_delta/2)),
##                   interpolation=interpolation, cmap=cmap, vmin=cmmin, vmax=cmmax,
##                   norm=norm, origin='lower', aspect='auto')
    im = ax.imshow(Zdisp, extent=extent, norm=norm, aspect='auto', interpolation='bilinear')

    # Label and title display ##################################################
    xlabel = 'Spatial frequency (u)'
    ylabel = 'Spatial frequency (v)'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.xlim(fmin, fmax)
    plt.ylim(fmin, fmax)

    # Colorbar display #########################################################
    if cmapdisplay:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        colormap = plt.colorbar(im, cax=cax)
        colormap.ax.set_title(cbartitle)

    else:
        colormap = None

    # Selection rectangle display ##############################################
    ###
    #rects = [[fmin + (fmax-fmin)/4, fmin + (fmax-fmin)/4 , (fmax-fmin)/2, (fmax-fmin)/2]]
    ###
    if rects is not None:
        for rect in rects:
            if len(rect) == 4:
                x, y, w, h = rect

                if x < u.min():
                    x = u.min()
                    
                if (x+w) > u.max():
                    w = u.max() - x
                    
                if y < v.min():
                    y = v.min()
                    
                if (y+h) > v.max():
                    h = v.max() - y

                ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="red"))

    # Additional line display ##################################################
    ###
    #lines = [[[0, 1], [0, 1]]]
    ###
    if lines is not None:
        for line in lines:
            x, y = line
##                if x0 < u.min():
##                    x0 = u.min()
##                    
##                if x1 > u.max():
##                    x1 = u.max()
##                    
##                if y < v.min():
##                    y = v.min()
##                    
##                if y1 > v.max():
##                    y1 = v.max()

            ax.add_line(mlines.Line2D(x, y, color="black"))

    # Saving figure to file ####################################################
    if filename is not None:
        # Map alone
        if not axisdisplay and not cmapdisplay:
            # the plotting has to be displayed in the full picture
            plt.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches='tight', pad_inches=0)

        # Map and axis
        else:
            plt.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches='tight')

    fig.canvas.draw()  # redraw current figure after it has been modified

    return fig, colormap


def plot_directional_filter(dataset, cmapname=None, creversed=False, fig=None, filename=None, dpi=None, cmapdisplay=True, axisdisplay=True, paramdisplay=False, cutoff=100, azimuth=0, width=2):
    '''Plot 2-D directional filter.'''

    # Colormap #################################################################
    if creversed and cmapname is not None:
        cmapname = cmapname + '_r'       # adds '_r' at the colormap name to use the reversed colormap

    cmap = plt.get_cmap(cmapname)       # gets the colormap

    # Figure for the display ###################################################
    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()
      
    # Figure ax creation #######################################################
    ax = fig.add_subplot(111)

    # Fourier Domain computation ###############################################
    Z = dataset.data.z_image
    nx = Z.shape[1]
    dx = dataset.info.x_gridding_delta
    fnyq_x = 1/(2*dx)
    u = np.fft.fftfreq(nx, d=dx)
    u = np.fft.fftshift(u)   # Centered frequencies

    ny = Z.shape[0]
    dy = dataset.info.y_gridding_delta
    fnyq_y = 1/(2*dy)
    v = np.fft.fftfreq(ny, d=dy)
    v = np.fft.fftshift(v)   # Centered frequencies

##    fmin = max(u.min(), v.min())
##    fmax = min(u.max(), v.max())

    fmin = -min(fnyq_x, fnyq_y)
    fmax = min(fnyq_x, fnyq_y)

    # Directional Filter Computation ###########################################
    Filt = cproc._gaussian_lowpass_dir_filter(Z.shape, fc=cutoff,
                                                azimuth=azimuth, n=width)
    Filtdisp = np.fft.fftshift(Filt)

    # Building image ###########################################################
    x0, x1 = u[0], u[-1]
    y0, y1 = v[0], v[-1]
    extent = (x0, x1, y0, y1)
    norm = colors.Normalize()

    ax.set_adjustable('box-forced') # 
    im = ax.imshow(Filtdisp, extent=extent, norm=norm, cmap=cmap, aspect='auto', interpolation='bilinear')
    
    # Label and title display ##################################################
    if paramdisplay:
        if cutoff is None:
            title = r'Directional Filter ($\theta = %d$°, $n = %d$)' %(azimuth, width)
        else:
            title = r'Directional Filter ($\theta = %d$°, $f_c = %d$, $n = %d$)' %(azimuth, cutoff, width)
    else:
        title = 'Directional Filter'

    xlabel = 'Spatial frequency (u)'
    ylabel = 'Spatial frequency (v)'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.xlim(fmin, fmax)
    plt.ylim(fmin, fmax)

    # Colorbar display #########################################################
    if cmapdisplay:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        colormap = plt.colorbar(im, cax=cax)
        #colormap.ax.set_title(cbartitle)

    else:
        colormap = None

    # Saving figure to file ####################################################
    if filename is not None:
        # Map alone
        if not axisdisplay and not cmapdisplay:
            # the plotting has to be displayed in the full picture
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)

        # Map and axis
        else:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

    fig.canvas.draw()  # redraw current figure after it has been modified

    return fig, colormap
