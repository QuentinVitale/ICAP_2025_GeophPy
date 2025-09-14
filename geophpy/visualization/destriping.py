# -*- coding: utf-8 -*-
'''
    geophpy.plotting.destrip
    ------------------------

    Destriping Mean Cross-Track Plot Management module.

    :copyright: Copyright 2017-2019 Lionel Darras, Quentin Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import matplotlib.pyplot as plt
import geophpy.core.processing as cproc
import numpy as np


def plot_mean_track(dataset, fig=None, filename=None,
                    Nprof='all', setmin=None, setmax=None, method='additive',
                    reference='mean', config='mono', Ndeg=None, plotflag='raw',
                    dpi=None, transparent=False):
    '''
    Plotting the mean cross-track (mean of each profile).

    cf. :func:`~geophpy.dataset.DataSet.meantrack_plot`
    '''

    if fig is None :                      # if first display
        fig = plt.figure()                  # creates the figure
    else :                                  # if not first display
        fig.clf()                           # clears figure

    ax = fig.add_subplot(111)

    # Dataset copy before destriping ###########################################
    DatasetDstp = dataset.copy()
    DatasetDstp.threshold(setmin=None, setmax=None, setnan=True, valfilt=False)
    Z    =  DatasetDstp.data.z_image
    cols = range(Z.shape[1])
    
    # Dataset statistics before destriping #####################################
    # Mean and standard deviation
    if reference == 'mean':
        # Per profile
        ZMOY = np.nanmean(Z, axis=0, keepdims=True)
        ZSTD = np.nanstd(Z, axis=0, keepdims=True)

        # Survey Global values
        MOY = np.nanmean(Z)
        STD = np.nanstd(Z)

    # Median and interquartile range
    elif reference == 'median':
        # Per profile
        ZMOY = np.nanmedian(Z, axis=0, keepdims=True)
        q75_i, q25_i = np.nanpercentile(Z, [75,25], axis=0, keepdims=True)
        ZSTD = q75_i - q25_i
        # or directly
        # ZSTD = np.subtract(*np.nanpercentile(Z, [75,25], axis=0, keepdims=True))

        # Survey Global values
        MOY = np.nanmedian(Z)
        STD = np.subtract(*np.percentile(Z, [75, 25]))

    # Data destriping ##########################################################
    # Constant destriping
    if Ndeg is None:
        cproc.destripecon(DatasetDstp, Nprof=Nprof, setmin=None, setmax=None, method=method, reference=reference, config=config, valfilt=False)

    # Curve destriping
    else:
        cproc.destripecub(DatasetDstp, Nprof=Nprof, setmin=None, setmax=None, Ndeg=Ndeg, valfilt=False)

    # Reference mean and std dev ###############################################
    # Moments of the global map
    if Nprof == 'all':
            MOYR = MOY + 0*ZMOY
            STDR = STD + 0*ZSTD

    # Moments fo Nprof=0 profile
    elif Nprof==0:
        MOYR = ZMOY
        STDR = ZSTD

    # Moments on Nprof profile
    else:
        MOYR = np.zeros(ZMOY.shape)
        STDR = np.zeros(ZSTD.shape)
        kp2  = Nprof // 2
        # Mean of Nprof cols centered the profile
        if reference == 'mean':
            for jc in cols:
                jc1 = max(0,jc-kp2)
                jc2 = min(Z.shape[1]-1,jc+kp2)
                MOYR[0,jc] = np.nanmean(Z[:,jc1:jc2])
                STDR[0,jc] = np.nanstd(Z[:,jc1:jc2])

        elif reference == 'median':
            for jc in cols:
                jc1 = max(0,jc-kp2)
                jc2 = min(Z.shape[1]-1,jc+kp2)
                MOYR[0,jc] = np.nanmedian(Z[:,jc1:jc2])
                STDR[0,jc] = np.subtract(*np.percentile(Z[:,jc1:jc2], [75, 25]))

    # Data after destriping ##########################################
    Zdsp    = DatasetDstp.data.z_image

    # Mean and standard deviation
    if reference == 'mean':
        ZMOYdsp = np.nanmean(Zdsp, axis=0, keepdims=True)
        ZSTDdsp = np.nanstd(Zdsp, axis=0, keepdims=True)

    # Median and interquartile range
    elif reference == 'median':
        ZMOYdsp = np.nanmedian(Zdsp, axis=0, keepdims=True)
        ZSTDdsp = np.subtract(*np.nanpercentile(Z, [75,25], axis=0, keepdims=True))

    # Build the image ################################################
    GlobalRefLabel = ' '.join(['Global', reference])
    RefLabel = ' '.join(['Reference', reference])

    # Plot raw data
    if plotflag=='raw' or plotflag=='both':
        x = np.arange(ZMOY.size).reshape((-1,1))
        y = ZMOY.reshape((-1,1))
        
        ax.plot(x, y, 'bo--', linewidth=1, markerfacecolor='None', label='Original')
        ax.plot([0, ZMOY.size-1], [MOY, MOY], 'k--', linewidth=3, label=GlobalRefLabel)

    # Plot destriped data
    if plotflag=='destriped' or plotflag=='both':
        xdsp = np.arange(ZMOYdsp.size).reshape((-1,1))
        ydsp = ZMOYdsp.reshape((-1,1))

        xref = np.arange(MOYR.size).reshape((-1,1))
        yref = MOYR.reshape((-1,1))
        
        ax.plot(xdsp, ydsp, 'r-', linewidth=2, label='Destriped')
        ax.plot(xref, yref, 'go--', linewidth=2, markersize=3, label=RefLabel)
    
    # Axis labels
    ax.set_title('Mean cross-track profile')
    ax.set_xlabel('Profile number')
    ax.set_ylabel('Value')

    # Upper center legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    #ax.legend(frameon=False, loc=9, ncol=3, mode='expand')
    ax.legend(frameon=False, loc=9, ncol=2)

    # Saving into a file #############################################
    if filename is not None:
       plt.savefig(filename, dpi=dpi, transparent=transparent)

    return fig
