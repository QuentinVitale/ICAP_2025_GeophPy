# -*- coding: utf-8 -*-
"""
   geophpy.core.spectral
   ---------------------

   Provides low-level utility functions for spectral analysis and
   Fourier domain filtering.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.
"""

import numpy as np

# --- Low-Level Functions ---
def calculate_wavenumbers(nx: int, ny: int, dx: float, dy: float) -> (np.ndarray, np.ndarray):
    """
    Calculates the 2D spatial frequencies (wavenumbers) for a grid.

    Parameters
    ----------
    nx : int
        Number of samples in the x-direction (columns).
    ny : int
        Number of samples in the y-direction (rows).
    dx : float
        Sample spacing in the x-direction.
    dy : float
        Sample spacing in the y-direction.

    Returns
    -------
    u : np.ndarray
        2D array of the x-component of the wavenumber.
    v : np.ndarray
        2D array of the y-component of the wavenumber.
    """
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    u, v = np.meshgrid(kx, ky)
    return u, v

def apodize_grid(image: np.ndarray, width_percent: float = 0.1) -> np.ndarray:
    """
    Applies a tapering window (apodization) to a grid to reduce FFT edge effects.

    Parameters
    ----------
    image : np.ndarray
        The 2D grid to be windowed.
    width_percent : float, optional
        The width of the taper as a percentage of the grid size. Defaults to 0.1 (10%).

    Returns
    -------
    np.ndarray
        The windowed (apodized) grid.
    """
    from scipy.signal.windows import tukey
    ny, nx = image.shape
    
    window_x = tukey(nx, alpha=width_percent * 2)
    window_y = tukey(ny, alpha=width_percent * 2)
    window_2d = np.outer(window_y, window_x)
    
    return image * window_2d

# --- Mixin Class ---

class SpectralGridMixin:
    """
    Mixin for spectral analysis methods that work on gridded data.
    """

    def get_fft2(self, demean: bool = True, apodize: bool = True, **kwargs):
        """
        Computes the 2D Fast Fourier Transform (FFT) of the gridded data.

        Parameters
        ----------
        demean : bool, optional
            If True, the mean of the grid is subtracted before the FFT to
            remove the zero-frequency component. Defaults to True.
        apodize : bool, optional
            If True, a tapering window is applied to the grid to reduce
            edge effects. Defaults to True.
        **kwargs
            Additional keyword arguments for `apodize_grid`, e.g., `width_percent`.

        Returns
        -------
        np.ndarray
            A 2D complex array of the FFT result.
        """
        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available for FFT.")

        image = self.grid.z_image.copy()
        
        # Handle NaNs
        nan_mask = np.isnan(image)
        mean_val = np.nanmean(image) if demean else 0
        image[nan_mask] = mean_val

        # Pre-processing
        if demean:
            image -= mean_val
        if apodize:
            image = apodize_grid(image, **kwargs)

        # Perform and return the FFT
        return np.fft.fft2(image)
    
