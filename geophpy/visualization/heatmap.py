#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    geophpy.plotting.heatmap
    ---------------------

    Module regrouping heatmap plotting functions.

    :copyright: Copyright 2025 L. Darras, P. Marty, Q. Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import numpy as np
import matplotlib.pyplot as plt

import geophpy.__config__ as CONFIG

def heatmap_plot(don,grid_final,grid,ncx,ncy,ext,pxy,w_exp):
    """
    Plotting the heatmap results.
    To the left : the heatmap of density of each tile.
    To the right : the subgrid where only tiles of value > 1 are selected, 
    which would be the selected area for interpolation/kriging.
    
    Parameters
    ----------
    don : dataframe
        Active dataframe.
    grid_final : list (dim 2) of float
        Contains the grid density score.
    grid : np.ndarray (dim 2) of float
        Contains the number of points located in each tile.
    ncx : list of str
        Names of every X columns.
    ncy : list of str
        Names of every Y columns.
    ext : [float, float, float, float]
        Extend of the grid. Contains ``[min_X, max_X, min_Y, max_Y]``.
    pxy : [float, float]
        Size of the grid for each axis. Contains ``[prec_X, prec_Y]``.
    w_exp : float
        Exponent of the function used to compute the detection window coefficients.
        Can be negative, only used for plot display.

    Notes
    -----
    The ``don``parameter only serves to plot points and does not modify the grids.
    
    See also
    --------
    ``dat_to_grid``
    """
    print("=== HEATMAP ===")

    plt.style.use('_mpl-gallery-nogrid')
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(CONFIG.fig_width,CONFIG.fig_height))
    
    # Gauche : Heatmap
    Q_l = [z for z in grid_final.flatten() if z == z]
    Q = np.quantile(Q_l,[0.05,0.95])
    ims = ax[0].imshow(grid_final, origin='lower', cmap='YlOrRd', vmin = Q[0], vmax=Q[1] , extent=ext)
    ax[0].set_title("Heatmap")
    ax[0].set_xlabel(ncx)
    ax[0].set_ylabel(ncy)
    ax[0].set_aspect('equal')
    plt.colorbar(ims,ax=ax[0])
    
    # Droite : Découpage induit de la heatmap
    grid_restr = np.array([[np.nan for i in range(pxy[0])] for j in range(pxy[1])])
    for j in range(pxy[1]):
        for i in range(pxy[0]):
            if grid_final[j][i] > 1 or grid[j][i] != 0:
                grid_restr[j][i] = 0
    ims = ax[1].imshow(grid_restr, origin='lower', cmap='gray', extent=ext)
    try:
        ax[1].scatter(don[ncx],don[ncy],marker='s',c=don["Num fich"], cmap='YlOrRd',s=5)
    except:
        ax[1].scatter(don[ncx],don[ncy],marker='s', cmap='YlOrRd',s=5)
    ax[1].set_title("Interpolation area (w_exp = {})".format(w_exp))
    ax[1].set_xlabel(ncx)
    ax[1].set_ylabel(ncy)
    ax[1].set_aspect('equal')
    plt.colorbar(ims,ax=ax[1])
    plt.show(block=False)
    # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
    # pour accélérer la vitesse de l'input
    plt.pause(CONFIG.fig_render_time)