# -*- coding: utf-8 -*-
"""
   geophpy.mag.processing
   ----------------------
   Provides Mixins for magnetism-specific processing algorithms.

   :copyright: Copyright 2014-2025 Q. Vitale, L. Darras, P. Marty and contributors.
   :license: GNU GPL v3, see LICENSE for details.

"""

from ..core import spectral

import copy
import numpy as np
#from scipy.linalg import lu_solve, lu_factor
#from geophpy.operation.general import *
import geophpy.core.processing as cproc


class MagneticGridMixin:
    """
    Mixin for advanced processing algorithms specific to magnetic grid data.
    """

    def log_transform_grid(self, mult_factor: float = 5, inplace: bool = False):
        """
        Applies a logarithmic transformation to the gridded data.

        This filter is useful for compressing the dynamic range of the data,
        which can help in visualizing both low and high amplitude anomalies
        simultaneously. The transformation is calculated as:
        `sign(z) * log(1 + abs(z) * mult_factor)`

        Parameters
        ----------
        mult_factor : float, optional
            A factor to multiply the data by before taking the logarithm,
            controlling the intensity of the transformation. Defaults to 5.
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the transformed grid if `inplace` is False.
        """

        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available to apply log transform.")

        target = self if inplace else copy.deepcopy(self)
        image = target.grid.z_image.copy()

        # Apply the logarithmic transformation, preserving the sign
        with np.errstate(divide='ignore', invalid='ignore'):
            transformed_image = np.sign(image) * np.log1p(np.abs(image) * mult_factor)
        
        # Restore NaNs that might have been converted to 0 by np.sign
        transformed_image[np.isnan(image)] = np.nan

        target.grid.z_image = transformed_image
        target.log_step('log_transform_grid', {'mult_factor': mult_factor})

        if not inplace:
            return target
        
    def analytic_signal_grid(self, apod_percent: float = 0.1, inplace: bool = False):
        """
        Calculates the analytic signal of the gridded magnetic data.

        The analytic signal is useful for locating the edges of magnetic
        sources. It is calculated in the frequency domain from the spatial
        derivatives of the magnetic field.

        Parameters
        ----------
        apod_percent : float, optional
            The width of the taper as a percentage of the grid size for apodization.
            Defaults to 0.1 (10%).
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the analytic signal grid if `inplace` is False.
        """
        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available to calculate analytic signal.")

        target = self if inplace else copy.deepcopy(self)
        image = target.grid.z_image.copy()

        # --- Pre-processing for FFT ---
        nan_mask = np.isnan(image)
        mean_val = np.nanmean(image)
        image[nan_mask] = mean_val # De-mean and infill NaNs
        
        # Apodize to reduce edge effects (using our new utility)
        image_processed = spectral.apodize_grid(image - mean_val, width_percent=apod_percent)

        # --- Core FFT Algorithm ---
        # 1. Get wavenumbers (using our new utility)
        ny, nx = image.shape
        dx = target.info.x_gridding_delta
        dy = target.info.y_gridding_delta
        u, v = spectral.calculate_wavenumbers(nx, ny, dx, dy)
        
        # 2. Perform FFT
        fft_data = np.fft.fft2(image_processed)
        
        # 3. Calculate derivatives in the frequency domain
        k = 2j * np.pi
        dZ_dx_fft = k * u * fft_data
        dZ_dy_fft = k * v * fft_data
        dZ_dz_fft = np.sqrt(u**2 + v**2) * 2 * np.pi * fft_data
        
        # 4. Inverse FFT to get spatial derivatives
        dZ_dx = np.fft.ifft2(dZ_dx_fft).real
        dZ_dy = np.fft.ifft2(dZ_dy_fft).real
        dZ_dz = np.fft.ifft2(dZ_dz_fft).real
        
        # 5. Calculate the analytic signal amplitude
        analytic_signal = np.sqrt(dZ_dx**2 + dZ_dy**2 + dZ_dz**2)
        
        # Restore NaNs
        analytic_signal[nan_mask] = np.nan

        target.grid.z_image = analytic_signal
        target.log_step('analytic_signal_grid', {})

        if not inplace:
            return target

    def continue_field_grid(self, height: float, apod_percent: float = 0.1, 
                            inplace: bool = False):
        """
        Performs upward or downward continuation on the gridded data.

        This filter recalculates the potential field as if it were measured
        at a different vertical height. It operates in the frequency domain.

        Parameters
        ----------
        height : float
            The vertical distance to continue the field, in the data's
            coordinate units.
            - Positive values (`height` > 0) correspond to upward continuation
              (smoothing).
            - Negative values (`height` < 0) correspond to downward continuation
              (sharpening).
        apod_percent : float, optional
            The width of the taper as a percentage of the grid size for apodization.
            Defaults to 0.1 (10%).
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the continued grid if `inplace` is False.
        """
        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available for field continuation.")

        target = self if inplace else copy.deepcopy(self)
        image = target.grid.z_image.copy()

        # --- Pre-processing for FFT ---
        nan_mask = np.isnan(image)
        mean_val = np.nanmean(image)
        image[nan_mask] = mean_val
        image_processed = spectral.apodize_grid(image - mean_val, width_percent=apod_percent)

        # --- Core FFT Algorithm ---
        # 1. Get wavenumbers
        ny, nx = image.shape
        dx = target.info.x_gridding_delta
        dy = target.info.y_gridding_delta
        u, v = spectral.calculate_wavenumbers(nx, ny, dx, dy)
        
        # 2. Perform FFT
        fft_data = np.fft.fft2(image_processed)

        # 3. Apply the continuation operator in the frequency domain
        # The operator is exp(-h * 2 * pi * k), where k is radial wavenumber
        radial_wavenumber = np.sqrt(u**2 + v**2)
        continuation_operator = np.exp(-height * 2 * np.pi * radial_wavenumber)  #  [Blakely 1996, eq12.8, p317]
        
        fft_continued = fft_data * continuation_operator

        # 4. Inverse FFT to return to the spatial domain
        continued_image = np.fft.ifft2(fft_continued).real
        
    # Conversion total field DATA ??
    #     #  habit in wumap to convert in tota-field but continuation applies
    #     #  on gardient data to.
    #     #  It is recomanded convert in toal-field for donwnward continuation though
    #     #  [Tabbagh 1999]
    #     ##
    #     ###
    #     if totalfieldconversionflag:
    #         ds = separation  # sensor's separation
    #         valTF_filt[indk] = valTF_filt[indk] / ( 1. - np.exp(-ds * k[indk]) )  #  conversion from gradient to total-field
    #       # equivalent to
    #       # valTF_filt = valTF * _grad_to_field(ds, k)      

        # 5. Restore mean and NaNs
        continued_image += mean_val
        continued_image[nan_mask] = np.nan

        target.grid.z_image = continued_image
        target.log_step('continue_field_grid', {'height': height})

        if not inplace:
            return target
        
    def reduction_to_the_pole_grid(self, inclination: float, declination: float, 
                                  apod_percent: float = 0.1, inplace: bool = False):
        """
        Applies a Reduction to the Pole (RTP) filter to the gridded data.

        This filter recalculates the magnetic anomaly as if the magnetization
        and ambient field were both vertical (inclination = 90). This
        simplifies the interpretation of magnetic features by centering
        anomalies directly over their sources.

        Parameters
        ----------
        inclination : float
            The geomagnetic inclination of the ambient field in degrees.
        declination : float
            The geomagnetic declination of the ambient field in degrees.
        apod_percent : float, optional
            The width of the taper as a percentage of the grid size for apodization.
            Defaults to 0.1 (10%).
        inplace : bool, optional
            If True, perform the operation in-place. Defaults to False.

        Returns
        -------
        Survey or None
            A new Survey object with the RTP-filtered grid if `inplace` is False.
        """
        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available for Reduction to the Pole.")

        target = self if inplace else copy.deepcopy(self)
        image = target.grid.z_image.copy()

        # --- Pre-processing for FFT ---
        nan_mask = np.isnan(image)
        mean_val = np.nanmean(image)
        image[nan_mask] = mean_val
        image_processed = spectral.apodize_grid(image - mean_val, apod_percent)

        # --- Core FFT Algorithm ---
        ny, nx = image.shape
        dx = target.info.x_gridding_delta
        dy = target.info.y_gridding_delta
        u, v = spectral.calculate_wavenumbers(nx, ny, dx, dy)
        
        fft_data = np.fft.fft2(image_processed)

        # Convert angles to radians
        inc_rad = np.deg2rad(inclination)
        dec_rad = np.deg2rad(declination)

        # Radial wavenumber (k)
        k = np.sqrt(u**2 + v**2)

        # --- RTP Operator in the frequency domain ---
        # Direction cosines for the ambient field and magnetization vector
        F = (np.sin(inc_rad) + 1j * np.cos(inc_rad) * np.cos(dec_rad - np.arctan2(v, u)))
        M = (np.sin(inc_rad) + 1j * np.cos(inc_rad) * np.cos(dec_rad - np.arctan2(v, u)))
        
        # Avoid division by zero, especially for the zero-frequency component
        F[F == 0] = 1e-9
        M[M == 0] = 1e-9

        rtp_operator = 1.0 / (F * M)
        
        # Apply the operator
        fft_rtp = fft_data * rtp_operator
        
        # Set the zero-frequency component back to zero
        fft_rtp[0, 0] = 0

        # --- Inverse FFT and Post-processing ---
        rtp_image = np.fft.ifft2(fft_rtp).real
        rtp_image += mean_val
        rtp_image[nan_mask] = np.nan

        target.grid.z_image = rtp_image
        target.log_step('reduction_to_the_pole_grid', {'inclination': inclination, 'declination': declination})

        if not inplace:
            return target
    
    def euler_deconvolution(self, structural_index: float, window_size: int = 10):
        """
        Performs Euler deconvolution to estimate source locations and depths.

        This method solves Euler's homogeneity equation over a moving window
        to find solutions for the (x, y, z) position of potential sources.

        Parameters
        ----------
        structural_index : float
            The structural index (SI), which describes the geometry of the
            magnetic source (e.g., 0 for a contact, 1 for a dike, 2 for a pipe).
        window_size : int, optional
            The size of the moving window in pixels. Defaults to 10.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the Euler solutions, with columns for
            the x, y, and z (depth) of each solution, as well as the
            estimated background field.
        """
        if self.grid is None or self.grid.z_image is None:
            raise ValueError("No gridded data available for Euler deconvolution.")

        image = self.grid.z_image
        
        # --- 1. Calculate Spatial Derivatives (using FFT) ---
        # (This logic is similar to the analytic_signal method)
        ny, nx = image.shape
        dx = self.info.x_gridding_delta
        dy = self.info.y_gridding_delta
        u, v = spectral.calculate_wavenumbers(nx, ny, dx, dy)
        
        fft_data = np.fft.fft2(np.nan_to_num(image))
        k = 2j * np.pi
        dZ_dx = np.fft.ifft2(k * u * fft_data).real
        dZ_dy = np.fft.ifft2(k * v * fft_data).real
        dZ_dz = np.fft.ifft2(np.sqrt(u**2 + v**2) * 2 * np.pi * fft_data).real

        # --- 2. Solve Euler's Equation in a moving window ---
        solutions = {'x': [],
                     'y': [],
                     'x0': [],
                     'y0': [],
                     'z0_depth': [],
                     'background': [],
                     'error':[]}
#        solutions = []
        x_coords, y_coords = self.get_xygrid()
        
        half_win = window_size // 2
        for j in range(half_win, ny - half_win):
            for i in range(half_win, nx - half_win):
                # Extract the window from the grid and its derivatives
                win_x = x_coords[j-half_win:j+half_win, i-half_win:i+half_win].flatten()
                win_y = y_coords[j-half_win:j+half_win, i-half_win:i+half_win].flatten()
                win_T = image[j-half_win:j+half_win, i-half_win:i+half_win].flatten()
                win_dx = dZ_dx[j-half_win:j+half_win, i-half_win:i+half_win].flatten()
                win_dy = dZ_dy[j-half_win:j+half_win, i-half_win:i+half_win].flatten()
                win_dz = dZ_dz[j-half_win:j+half_win, i-half_win:i+half_win].flatten()

                # Set up the system of linear equations A*m = b
                A = np.vstack([win_dx, win_dy, win_dz, np.ones(win_dx.size) * structural_index]).T
                b = win_x * win_dx + win_y * win_dy + win_T * structural_index

                # Solve for the model parameters [x0, y0, z0, B]
                try:
                    m, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    # Error estimation
                    n = A.shape[0]  # Number of data points (rows in A)
                    p = A.shape[1]  # Number of parameters (columns in A)
                    dof = n - p  # Degrees of freedom
                    sum_sq_residuals = residuals[0]  # The sum of squared residuals is the first (and only) item in the 'residuals' array
                    s_sq = sum_sq_residuals / dof  # Residual variance
                    # Calculate the Covariance Matrix
                    try:
                        covariance_matrix = s_sq * np.linalg.inv(A.T @ A)

                        # Step C: Extract the Standard Errors
                        # -----------------------------------
                        # The standard error for each parameter is the square root of its variance.
                        # The variances are the diagonal elements of the covariance matrix.
                        param_variances = np.diag(covariance_matrix)
                        param_errors = np.sqrt(param_variances) #  'p3', 'p2', 'p1', 'p0

                    except np.linalg.LinAlgError:
                        print("Error: The matrix (A.T @ A) is singular and cannot be inverted.")
                        print("This can happen if your columns in A are linearly dependent (collinearity).")

                    
                    # Add the solution for the center of the window
                    solutions['x'].append(win_x[len(win_x)//2])
                    solutions['y'].append(win_y[len(win_y)//2])
                    solutions['x0'].append(m[0])
                    solutions['y0'].append(m[1])
                    solutions['z0_depth'].append(-m[2])
                    solutions['background'].append(m[3])
                    solutions['error'].append(param_errors[2])
                    # solutions.append({'x': win_x[len(win_x)//2], 'y': win_y[len(win_y)//2], 
                    #                   'x0': m[0], 'y0': m[1], 'z0_depth': -m[2], 'background': m[3]})
                except np.linalg.LinAlgError:
                    continue

        #return pd.DataFrame(solutions)
        return solutions

#------------------------------------------------------------------------------#
# User defined parameters                                                      #
#------------------------------------------------------------------------------#
# list of available magnetic survey configuraiton
#prosptechlist = ["Magnetic field", "Magnetic field gradient", "Vertical component gradient"]
sensorconfiglist = ["TotalField", "TotalFieldGradient", "Fluxgate"]
structuralindexlist = ['-1','1','2','3']

# def logtransform(dataset, multfactor=5, setnan=True, valfilt=False):
#     ''' Apply a logarihtmic transformation to the dataset.

#     cf. :meth:`~geophpy.dataset.DataSet.logtransform`

#     '''
# ##   # Calculation of lines and columns numbers
# ##   ny = len(val)
# ##   nx = len(val[0])
# ##   
# ##   for ix in range(nx):                                        # for each column
# ##      for iy in range(ny):                                     # for each line
# ##         if (val[iy][ix] != np.nan) :                          # if valid value
# ##            val[iy][ix] = val[iy][ix] * multfactor             # multiplying
# ##
# ##   for ix in range(nx):                                        # for each column
# ##      for iy in range(ny):                                     # for each line
# ##         if ((val[iy][ix] != np.nan) and (val[iy][ix] < -1)):
# ##            val[iy][ix] = -np.log10(-val[iy][ix])
# ##
# ##   for ix in range(nx):                                        # for each column
# ##      for iy in range(ny):                                     # for each line
# ##         if ((val[iy][ix] != np.nan) and (val[iy][ix] > 1)):
# ##            val[iy][ix] = np.log10(val[iy][ix])
# ##
# ##   for ix in range(nx):                                        # for each column
# ##      for iy in range(ny):                                     # for each line
# ##         if ((val[iy][ix] != np.nan) and (val[iy][ix] > -1) and (val[iy][ix] < 1)) :
# ##            val[iy][ix] = np.nan

#     # Value to replace data between -1 and 1 ###################################
#     if setnan:
#         replace_val = np.nan
#     else:
#         replace_val = 0

#     # Filter values ...TBD... ##################################################
#     if valfilt:
#         pass

#     # Filter zimage ############################################################
#     else:
#         val = dataset.data.z_image*multfactor
#         indx = np.where(val<-1)
#         val[indx] = -np.log10(-val[indx])

#         indx = val>1
#         val[indx] = np.log10(val[indx]) 

#         condition = np.logical_and(val>-1, val<1)
#         indx = np.where(condition)
#         val[indx] = replace_val

#         dataset.data.z_image = val

#     return dataset


# def polereduction(dataset, apod=0, inclination=65, declination=0, azimuth=0,
#                   magazimuth=None, incl_magn=None, decl_magn=None):
#     ''' Dataset Reduction to the magnetic Pole.

#     cf. :meth:`~geophpy.dataset.DataSet.polereduction`

#     '''
# ##########
# ##
# ## Original code has been re-implemented in a less fortranic way
# ## and source remanent magnetization option has been added
# ##
# ##########
# ##
# ##   val = dataset.data.z_image
# ##
# ##   # Calculation of lines and columns numbers
# ##   ny = len(val)
# ##   nx = len(val[0])
# ##   # Calculation of total values number
# ##   nt = nx * ny
# ##
# ##   nan_indexes = np.isnan(val)                                 # searches the 'nan' indexes in the initial array
# ##   if(np.isnan(val).any()):                                    # if at least 1 'nan' is detected in the array,
# ##      
# ##      x = np.linspace(dataset.info.x_min, dataset.info.x_max, nx, endpoint=True)
# ##      y = np.linspace(dataset.info.y_min, dataset.info.y_max, ny, endpoint=True)
# ##      _fillnanvalues(x, y, val.T)                              # fills the 'nan' values by interpolated values
# ##
# ##   if (apod > 0):
# ##      apodisation2d(val, apod)
# ##
# ##   # Meaning calculation
# ##   mean = val.mean()
# ##
# ##   # complex conversion, val[][] -> cval[][]
# ##   cval = np.array(val, dtype=complex)
# ##
# ##   # fast fourier series calculation
# ##   cvalfft = np.fft.fft2(cval)
# ##
# ##   # filter application
# ##   # deg->rad angle conversions
# ##   rinc = (inclineangle*np.pi)/180 # inclination in radians
# ##   ralpha = (alphaangle*np.pi)/180  # alpha angle in radians
# ##
# ##   cosu = np.absolute(np.cos(rinc) * np.cos(ralpha))
# ##   cosv = np.absolute(np.cos(rinc) * np.sin(ralpha))
# ##   cosz = np.sin(rinc)
# ##
# ##   deltax = dataset.info.x_gridding_delta
# ##   deltay = dataset.info.y_gridding_delta
# ##
# ##   for ix in range(nx):                                        # for each column
# ##      for iy in range(ny):                                     # for each line
# ##         if ((ix != 0) or (iy != 0)):                          # if not first point of array
# ##            cu, cv = _freq(ix, iy, deltax, deltay, nx, ny)     # complex u and v calculation
# ##            cz = np.sqrt(np.square(cu) + np.square(cv))
# ##            cred = complex(cz*cosz, cu*cosu + cv*cosv)
# ##            cvalfft[iy][ix] = ((cvalfft[iy][ix] * np.square(cz))/np.square(cred))
# ##
# ##   # FFT inversion
# ##   icvalfft = np.fft.ifft2(cvalfft)
# ##   val = icvalfft.real + mean                                  # real part added to mean calculation
# ##
# ##   val[nan_indexes] = np.nan                                   # set the initial 'nan' values, as in the initial array
# ##
# ##   dataset.data.z_image = val                                  # to get the 'z_image' array in (x,y) format.

#     val = dataset.data.z_image

#     # Transformation to sprectral Domain #######################################
#     # Apodization before FT
#     if (apod > 0):
#         cproc.apodisation2d(val, apod)

#     # Filling NaNs
#     valnan = np.copy(val)
#     nan_idx = np.isnan(valnan)  # index of NaNs in the original dataset
#     valfill = cproc.fillnanvalues(valnan, indexout=False)  # Filled dataset

#     # De-meaning
#     valmean = np.nanmean(val)
#     valfill = valfill-valmean

#     # Fourier Transform computation
#     valTF = np.fft.fft2(valfill)  # Frequency domain

#     # Wavenumbers computation
#     ny, nx = val.shape
#     dx = dataset.info.x_gridding_delta
#     dy = dataset.info.y_gridding_delta

#     kx, ky = cproc.wavenumber(nx, ny, dx, dy)  # x, y-directed wavenumber matrices
#     k = np.sqrt(kx**2 + ky**2)  # radial wavenumber matrix
#     indk = k!=0  # index k valid for the RTP

#     # Magnetic azimuth #########################################################
#     ## The angle between the survey profile direction and the ambient magnetic
#     ## field is phi = declination-azimuth.
#     ## if the given angle is directly the angle between the ambient magnetic
#     ## field the survey direction (phi), we assume
#     ## declinaison = 0 so -phi = azimuth
#     if magazimuth is not None:
#         # ambient field
#         phi = magazimuth  # given azimuth is magnetic azimuth
#         declination = 0   # declination is set to 0 for the _dircos routine
#         azimuth  = -phi

#     # Reduction to the Pole Operator ###########################################
#     F = np.empty([ny, nx], dtype=complex)
#     # Neglecting source remanent magnetization
#     if incl_magn is None or decl_magn is None:

#         # Ambient field unit-vector components
#         fx, fy ,fz = _dircos(inclination, declination, azimuth)

#         # RTP operator
#         # Simplified form of Blakely 1996, eq12.31, p331, for mx=fx, my=fy and mz=mz
#         F[indk] = k[indk]**2 / ( k[indk]*fz + 1j*(kx[indk]*fx + ky[indk]*fy) )**2

#     # With sourceremanent magnetization
#     else:
#         # Ambient field and total magnetization unit-vectors components
#         fx, fy ,fz = _dircos(inclination, declination, azimuth)
#         mx, my ,mz = _dircos(incl_magn, decl_magn, azimuth)

#         # RTP operator  [Blakely 1996, eq12.31, p331]
#         a1 = mz*fz - mx*fx
#         a2 = mz*fz - my*fy
#         a3 = -my*fx - mx*fy
#         b1 = mx*fz + mz*fx
#         b2 = my*fz + mz*fy

#         F[indk] = k[indk]**2 / (a1*kx[indk]**2
#                               + a2*ky[indk]**2
#                               + a3*kx[indk]*ky[indk]
#                               + 1j*k[indk]*(b1*kx[indk] + b2*ky[indk]) )

#     # Applying filter###########################################################
#     valTF_filt = valTF*F

#     # Transformation nack to spatial domain ####################################
#     valfilt = np.fft.ifft2(valTF_filt)  # Spatial domain
#     valfilt = np.real(valfilt) + valmean  # Re-meaning
#     valfilt[nan_idx] = np.nan  # unfilled dataset

#     dataset.data.z_image = valfilt

#     return dataset


# def continuation(dataset, apod=0, distance=2, totalfieldconversionflag=False, separation=0.7):
#     ''' Dataset continuation (upward/downward).

#     cf. :meth:`~geophpy.dataset.DataSet.continuation`

#     '''
# ##########
# ##
# ## Original code has been re-implemented in a less fortranic way
# ##
# ##########
# ##
# ##   val = dataset.data.z_image
# ##
# ##   # Calculation of lines and columns numbers
# ##   ny = len(val)
# ##   nx = len(val[0])
# ##   # Calculation of total values number
# ##   nt = nx * ny
# ##
# ##   nan_indexes = np.isnan(val)                                 # searches the 'nan' indexes in the initial array
# ##   if(np.isnan(val).any()):                                    # if at least 1 'nan' is detected in the array,
# ##      
# ##      x = np.linspace(dataset.info.x_min, dataset.info.x_max, nx, endpoint=True)
# ##      y = np.linspace(dataset.info.y_min, dataset.info.y_max, ny, endpoint=True)
# ##      _fillnanvalues(x, y, val.T)                                # fills the 'nan' values by interpolated values
# ##
# ##   if (apod > 0):
# ##      apodisation2d(val, apod)
# ##
# ##   dz = downsensoraltitude - upsensoraltitude     
# ##   zp = - continuationvalue + downsensoraltitude
# ##
# ##   # complex conversion, val[][] -> cval[][]
# ##   cval = np.array(val, dtype=complex)
# ##
# ##   # fast fourier series calculation
# ##   cvalfft = np.fft.fft2(cval)
# ##
# ##   deltax = dataset.info.x_gridding_delta
# ##   deltay = dataset.info.y_gridding_delta
# ##
# ##   for ix in range(nx):                                        # for each column
# ##      for iy in range(ny):                                     # for each line
# ##         if ((ix != 0) or (iy != 0)):                          # if not first point of array
# ##            cu, cv = _freq(ix, iy, deltax, deltay, nx, ny)     # complex u and v calculation
# ##            cz = np.sqrt(np.square(cu) + np.square(cv)) * (2. * np.pi)
# ##            if (continuationflag == True):
# ##               ce = np.exp(zp * cz)
# ##               cvalfft[iy][ix] = (cvalfft[iy][ix] * complex(ce, 0.))
# ##            if (prosptech == prosptechlist[1]):                # if prospection technic is magnetic field gradient
# ##               cdz = 1. - np.exp(dz * cz)
# ##               cvalfft[iy][ix] = (cvalfft[iy][ix] / complex(cdz, 0.))
# ##
# ##   # FFT inversion
# ##   icvalfft = np.fft.ifft2(cvalfft)
# ##   val = icvalfft.real
# ##
# ##   val[nan_indexes] = np.nan                                   # set the initial 'nan' values, as in the initial array
# ##
# ##   dataset.data.z_image = val                                  # to get the 'z_image' array in (x,y) format.

#     val = dataset.data.z_image

#     # Transformation to sprectral Domain #######################################
#     # Apodization before FT
#     if (apod > 0):
#         cproc.apodisation2d(val, apod)

#     # Filling NaNs
#     valnan = np.copy(val)
#     nan_idx = np.isnan(valnan)  # index of NaNs in the original dataset
#     valfill = cproc.fillnanvalues(valnan, indexout=False)  # Filled dataset

# ##    # De-meaning
# ##    valmean = np.nanmean(val)
# ##    valfill = valfill-valmean

#     # Fourier Transform computation
#     valTF = np.fft.fft2(valfill)  # Frequency domain

#     # Wavenumbers computation
#     ny, nx = val.shape
#     dx = dataset.info.x_gridding_delta
#     dy = dataset.info.y_gridding_delta

#     kx, ky = cproc.wavenumber(nx, ny, dx, dy)  # x, y-directed wavenumber matrices
#     k = np.sqrt(kx**2 + ky**2)  # radial wavenumber matrix
#     indk = k!=0  # index k valid for the RTP

#     # Continuation #############################################################
#     dh = distance  # continuation distance
#     valTF_filt = valTF * np.exp(-dh*k)  #  [Blakely 1996, eq12.8, p317]
#     # equivalent to
#     # valTF_filt = valTF * _continu(dh, k, direction='auto')

#     # Magnetic total-field gradient survey #####################################
#     ###
#     ##
#     #...TBD... ??
#     # Should we keep the conversion ??
#     #  habit in wumap to convert in tota-field but continuation applies
#     #  on gardient data to.
#     #  It is recomanded convert in toal-field for donwnward continuation though
#     #  [Tabbagh 1999]
#     ##
#     ###
#     if totalfieldconversionflag:
#         ds = separation  # sensor's separation
#         valTF_filt[indk] = valTF_filt[indk] / ( 1. - np.exp(-ds * k[indk]) )  #  conversion from gradient to total-field
#       # equivalent to
#       # valTF_filt = valTF * _grad_to_field(ds, k)      

#     # Transformation back to spatial domain ####################################
#     valfilt = np.fft.ifft2(valTF_filt)  # Spatial domain
#     valfilt = np.real(valfilt) #+ valmean  # Re-meaning
#     valfilt[nan_idx] = np.nan  # unfilled dataset

#     dataset.data.z_image = valfilt

#     return dataset


def eulerdeconvolution(dataset, apod=0, structind=None, windows=None, xstep=None, ystep=None):
    ''' Classic Euler deconvolution.

    cf. :meth:`~geophpy.dataset.DataSet.eulerdeconvolution`

    '''

##########
##
## Original code has been re-implemented in a less fortranic way
##
##########
##   val = dataset.data.z_image
##
##   # Calculation of lines and columns numbers
##   ny = len(val)
##   nx = len(val[0])
##   # Calculation of total values number
##   nt = nx * ny
##
##   x = np.linspace(dataset.info.x_min, dataset.info.x_max, nx, endpoint=True)
##   y = np.linspace(dataset.info.y_min, dataset.info.y_max, ny, endpoint=True)
##
##   nan_indexes = np.isnan(val)                                 # searches the 'nan' indexes in the initial array
##   if(np.isnan(val).any()):                                    # if at least 1 'nan' is detected in the array,     
##      _fillnanvalues(x, y, val.T)                                # fills the 'nan' values by interpolated values
##
##   if (apod > 0):
##      apodisation2d(val, apod)
##
##   # complex conversion, val[][] -> cval[][]
##   cval = np.array(val, dtype=complex)
##   val = cval.real
##
##   deltax = dataset.info.x_gridding_delta
##   deltay = dataset.info.y_gridding_delta
##
##   # fast fourier series calculation
##   cvalfft = np.fft.fft2(cval)
##   cvalfftdx = np.array(np.zeros((ny,nx)), dtype=complex)
##   cvalfftdy = np.array(np.zeros((ny,nx)), dtype=complex)
##   cvalfftdz = np.array(np.zeros((ny,nx)), dtype=complex)
##
##   # derivations calculation
##   for iy in range(ny):                                        # for each column
##      for ix in range(nx):                                     # for each line
##         if ((ix != 0) or (iy != 0)):                          # if not first point of array
##            cu, cv = _freq(ix, iy, deltax, deltay, nx, ny)     # complex u and v calculation
##            cz = np.sqrt(np.square(cu) + np.square(cv))
##            # x derivation 
##            cvalfftdx[iy][ix] = cvalfft[iy][ix]*complex(0., cu*np.pi*2.)
##            # y derivation 
##            cvalfftdy[iy][ix] = cvalfft[iy][ix]*complex(0., cv*np.pi*2.)
##            # z derivation 
##            cvalfftdz[iy][ix] = cvalfft[iy][ix]*cz*np.pi*2.
##         else:
##            cvalfftdx[0][0] = complex(0., 0.)
##            cvalfftdy[0][0] = complex(0., 0.)
##            cvalfftdz[0][0] = complex(0., 0.)
##
##   # FFT inversion
##   valdx = (np.fft.ifft2(cvalfftdx)).real
##   valdy = (np.fft.ifft2(cvalfftdy)).real
##   valdz = (np.fft.ifft2(cvalfftdz)).real
##
##   ixmin, xnearestmin = _searchnearest(x, xmin)
##   ixmax, xnearestmax = _searchnearest(x, xmax)
##   
##   iymin, ynearestmin = _searchnearest(y, ymin)
##   iymax, ynearestmax = _searchnearest(y, ymax)
##
##   if ((nflag == 0) or (nvalue==0)):   # automatic calculation structural index
##                                       # 4 unknowns equation to solve
##      b = np.zeros(4)
##      A = np.zeros((4,4))
##      print('Estimated SI')
##      for l in range(iymin, iymax+1):
##         for c in range(ixmin, ixmax+1):
##            coef = x[c] * valdx[l][c] + y[l] * valdy[l][c]
##            b[0] += coef * val[l][c]
##            b[1] += coef * valdx[l][c]
##            b[2] += coef * valdy[l][c]
##            b[3] += coef * valdz[l][c]
##
##            A[0][0] += np.square(val[l][c])
##            A[0][1] += val[l][c] * valdx[l][c]
##            A[0][2] += val[l][c] * valdy[l][c]
##            A[0][3] += val[l][c] * valdz[l][c]
##            A[1][1] += np.square(valdx[l][c])
##            A[1][2] += valdx[l][c] * valdy[l][c]
##            A[1][3] += valdx[l][c] * valdz[l][c]
##            A[2][2] += np.square(valdy[l][c])
##            A[2][3] += valdy[l][c] * valdz[l][c]
##            A[3][3] += np.square(valdz[l][c])
##
##      # symmetry
##      A[1][0] = A[0][1]
##      A[2][0] = A[0][2]
##      A[3][0] = A[0][3]
##      A[2][1] = A[1][2]
##      A[3][1] = A[1][3]
##      A[3][2] = A[2][3]
##
##      x = _gaussresolution(A, b)
##      nvalue = -x[0]
##      xpt = x[1]
##      ypt = x[2]
##      depth = x[3]
##      
##   else :                              # fixed structural index
##                                       # 3 unknowns equation to solve
##      b = np.zeros(3)
##      A = np.zeros((3,3))
##      print('Fixed SI')
##      count= 0
##      for l in range(iymin, iymax+1):
##         for c in range(ixmin, ixmax+1):
##            coef = nvalue * val[l][c] + x[c] * valdx[l][c] + y[l] * valdy[l][c]
##            count += 1
##            b[0] += coef * valdx[l][c]
##            b[1] += coef * valdy[l][c]
##            b[2] += coef * valdz[l][c]
##
##            A[0][0] += np.square(valdx[l][c])
##            A[0][1] += valdx[l][c] * valdy[l][c]
##            A[0][2] += valdx[l][c] * valdz[l][c]
##            A[1][1] += np.square(valdy[l][c])
##            A[1][2] += valdy[l][c] * valdz[l][c]
##            A[2][2] += np.square(valdz[l][c])
##
##      # symmetry
##      A[1][0] = A[0][1]
##      A[2][0] = A[0][2]
##      A[2][1] = A[1][2]
##
##      x = _gaussresolution(A, b)
##      xpt = x[0]
##      ypt = x[1]
##      depth = x[2]

    val = dataset.data.z_image

    # Transformation to sprectral Domain #######################################
    # Apodization before FT
    if (apod > 0):
        cproc.apodisation2d(val, apod)

    # Filling NaNs
    valnan = np.copy(val)
    nan_idx = np.isnan(valnan)  # index of NaNs in the original dataset
    valfill = cproc.fillnanvalues(valnan, indexout=False)  # Filled dataset

##    # De-meaning
##    valmean = np.nanmean(val)
##    valfill = valfill-valmean

    # Fourier Transform computation
    valTF = np.fft.fft2(valfill)  # Frequency domain

    # Wavenumbers computation
    ny, nx = val.shape
    dx = dataset.info.x_gridding_delta
    dy = dataset.info.y_gridding_delta

    kx, ky = cproc.wavenumber(nx, ny, dx, dy)  # x, y-directed wavenumber matrices
    k = np.sqrt(kx**2 + ky**2)  # radial wavenumber matrix
    indk = k!=0  # index of valid k for the RTP

    # Spatial derivatives in the frequency domain ##############################
    # ... TBD...
    ## Use spatial computation fo the horizontal derivatices
    ## to limit noise as suggested by [Cooper 2002] ?
    dval_dxTF = valTF*1j*kx  # x-directed gradient [Blakely 1996, eq12.15, p324]
    dval_dyTF = valTF*1j*ky  # y-directed gradient [Blakely 1996, eq12.16, p324]
    dval_dzTF = valTF*k  #  vertical gradient [Blakely 1996, eq12.13, p323]

    # Potential field derivatives in the spatial domain ########################
    dval_dx = np.real(np.fft.ifft2(dval_dxTF))
    dval_dy = np.real(np.fft.ifft2(dval_dyTF))
    dval_dz = np.real(np.fft.ifft2(dval_dzTF))

    # Data sub-set creation ####################################################
    # Grid coordinates
    xvect = dataset.get_xvect().ravel()  # x- coordinates vect
    yvect = dataset.get_yvect().ravel()  # y- coordinates vect
    xgrid = dataset.get_xgrid()  # x-grid coordinates vect
    ygrid = dataset.get_ygrid()  # y-grid coordinates vect

    # Creating a 2-D sliding window over the datset grid
    if windows is None:
        if xstep is None and ystep is None:
            xstep = 7
            ystep = round(xstep*dx/dy)

        elif xstep is None and ystep is not None:
            xstep = round(ystep*dy/dx)

        elif xstep is not None and ystep is None:
            ystep = round(xstep*dx/dy)

        # x windows extent
        xbounds = xvect[::xstep]
        xmins = xbounds[:-1]
        xmaxs = xbounds[1:]

        # y windows extent
        ybounds = yvect[::ystep]
        ymins = ybounds[:-1]
        ymaxs = ybounds[1:]

        # Extending windows to dataset max spatial extent
        ## result in an smaller window extent at the dataset border
        if xmaxs[-1]!=xvect.max():
            xmins = np.append(xmins, xmaxs[-1])
            xmaxs = np.append(xmaxs, xvect.max())

        if ymaxs[-1]!=yvect.max():
            ymins = np.append(ymins, ymaxs[-1])
            ymaxs = np.append(ymaxs, yvect.max())

        # windows extent
        windows = []
        for idx, x in enumerate(xmins):
            for idy, y in enumerate(ymins):
                windows.append([xmins[idx], xmaxs[idx], ymins[idy], ymaxs[idy]])

    # Euler's deconvolution ####################################################
    if np.asarray(windows).ndim ==1:  # Unique window provided
        windows = [windows]

    results = []
    for extent in windows:
        # Current spatial sub-window
        xmin, xmax, ymin, ymax = extent 

        indx = np.logical_and(xgrid<=xmax, xgrid>=xmin)
        indy = np.logical_and(ygrid<=ymax, ygrid>=ymin)
        indval = np.logical_and(indx, indy)

        xi = xgrid[indval].reshape((-1,1))
        yi = ygrid[indval].reshape((-1,1))
        zi = 0*xi.reshape((-1,1))
        vali = valfill[indval].reshape((-1,1))

        dvdxi = dval_dx[indval].reshape((-1,1))
        dvdyi = dval_dy[indval].reshape((-1,1))
        dvdzi = dval_dz[indval].reshape((-1,1))

        # Estimated Structural Index
        if structind is None or structind not in [0, 1, 2, 3]:
            A = np.hstack((dvdxi, dvdyi, dvdzi, -vali))
            b = xi*dvdxi + yi*dvdyi + zi*dvdzi

            X, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

            residuals = float(residuals) 
            x0, y0, z0, N = [float(i) for i in X] 

        # Fixed Structural Index
        else:
            N = structind
            A = np.hstack((dvdxi, dvdyi, dvdzi))
            b = xi*dvdxi + yi*dvdyi + zi*dvdzi + N*vali

            X, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

            residuals = float(residuals)
            x0, y0, z0 = [float(i) for i in X] 

        # Storing results
        results.append([x0, y0, z0, N, residuals,
                        xi.min(), xi.max(), yi.min(), yi.max()])

    return results


def _searchnearest(array, value):
    ''' Return the index of the nearest value of val in array the value itself. '''

    # 1-D arrays
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    # 2-D arrays
    if len(array.shape)>1:
        idx = np.unravel_index(idx, array.shape)

    return idx, array[idx]

##def _searchnearest(array, val):
##   '''
##   Search the nearset value of val, in the array
##
##   Parameters :
##
##   :array:  1D array ordered to find the nearest value of val
##
##   :val: value to find
##
##   Returns :
##
##   :index: index in the array of the nearest value found
##
##   :valnearest: value of the nearest value found
##
##   '''
##   n = len(array)
##   for i in range(n):
##      gap = abs(val-array[i])
##      if ((i==0) or (gap<mingap)):
##         mingap = gap
##         index = i
##   return index, array[index]            
      
##def _gaussresolution(A, b):
##   ''' Solve Ax=b equation using Gauss pivot and LU factorisation. '''
##
##   n=len(A)
##   # gauss pivot calculation
##   for i1 in range(n-1):
##      # find the partial pivot
##      l=i1      
##      xam = np.absolute(A[i1][i1])
##      for j in range(i1+1,n):
##         x = np.absolute(A[j][i1])
##         if (x>xam):
##            xam = x
##            l=j
##
##      # set pivot at its place, swapping pivot and current lines
##      if (l>i1):
##         aux = b[l]
##         b[l] = b[i1]
##         b[i1] = aux
##         for j in range(n):
##            aux = A[l][j]
##            A[l][j] = A[i1][j]
##            A[i1][j] = aux
##
##   # solves Ax=b using LU factorisation
##   x = lu_solve(lu_factor(A), b)
##
##   return x



# def analyticsignal(dataset, apod=0):
#     ''' Dataset Analytic Signal.

#     cf. :meth:`~geophpy.dataset.DataSet.analyticsignal`

#     '''
# ##########
# ##
# ## Original code has been re-implemented in a less fortranic way
# ##
# ##########
# ##    val = dataset.data.z_image
# ##
# ##    # Calculation of lines and columns numbers
# ##    ny = len(val)
# ##    nx = len(val[0])
# ##    # Calculation of total values number
# ##    nt = nx * ny
# ##
# ##    nan_indexes = np.isnan(val)                                 # searches the 'nan' indexes in the initial array
# ##    if(np.isnan(val).any()):                                    # if at least 1 'nan' is detected in the array,
# ##      
# ##       x = np.linspace(dataset.info.x_min, dataset.info.x_max, nx, endpoint=True)
# ##       y = np.linspace(dataset.info.y_min, dataset.info.y_max, ny, endpoint=True)
# ##       _fillnanvalues(x, y, val.T)                              # fills the 'nan' values by interpolated values
# ##
# ##    if (apod > 0):
# ##       apodisation2d(val, apod)
# ##
# ##    # complex conversion, val[][] -> cval[][]
# ##    cval = np.array(val, dtype=complex)
# ##
# ##    deltax = dataset.info.x_gridding_delta
# ##    deltay = dataset.info.y_gridding_delta
# ##
# ##    # fast fourier series calculation
# ##    cvalfft = np.fft.fft2(cval)
# ##    cvalfftdx = np.fft.fft2(cval)
# ##    cvalfftdy = np.fft.fft2(cval)
# ##    cvalfftdz = np.fft.fft2(cval)
# ##
# ##    # dx derivations calculation
# ##    cvalfftdx[0][0] = complex(0., 0.)
# ##    for ix in range(nx):                                        # for each column
# ##       for iy in range(ny):                                     # for each line
# ##          if ((ix != 0) or (iy != 0)):                          # if not first point of array
# ##             cu, cv = _freq(ix, iy, deltax, deltay, nx, ny)     # complex u and v calculation
# ##             # x derivation 
# ##             cvalfftdx[iy][ix] = cvalfft[iy][ix]*complex(0., cu*np.pi*2)
# ##         
# ##    # FFT inversion
# ##    icvalfft = np.fft.ifft2(cvalfftdx)
# ##    val = np.square(icvalfft.real)
# ##
# ##    # dy derivations calculation
# ##    cvalfftdy[0][0] = complex(0., 0.)
# ##    for ix in range(nx):                                        # for each column
# ##       for iy in range(ny):                                     # for each line
# ##          if ((ix != 0) or (iy != 0)):                          # if not first point of array
# ##             cu, cv = _freq(ix, iy, deltax, deltay, nx, ny)     # complex u and v calculation
# ##             # y derivation 
# ##             cvalfftdy[iy][ix] = cvalfft[iy][ix]*complex(0., cv*np.pi*2)
# ##         
# ##    # FFT inversion
# ##    icvalfft = np.fft.ifft2(cvalfftdy)
# ##    val = val + np.square(icvalfft.real)
# ##
# ##    # dz derivations calculation
# ##    cvalfftdz[0][0] = complex(0., 0.)
# ##    for ix in range(nx):                                        # for each column
# ##       for iy in range(ny):                                     # for each line
# ##          if ((ix != 0) or (iy != 0)):                          # if not first point of array
# ##             cu, cv = _freq(ix, iy, deltax, deltay, nx, ny)     # complex u and v calculation
# ##             cz = np.sqrt(np.square(cu) + np.square(cv))
# ##             # z derivation 
# ##             cvalfftdz[iy][ix] = cvalfft[iy][ix]*cz*np.pi*2
# ##         
# ##    # FFT inversion
# ##    icvalfft = np.fft.ifft2(cvalfftdz)
# ##    val = np.sqrt(val + np.square(icvalfft.real))
# ##
# ##    val[nan_indexes] = np.nan                                   # set the initial 'nan' values, as in the initial array
# ##
# ##    dataset.data.z_image = val                                  # to get the 'z_image' array in (x,y) format.

#     val = dataset.data.z_image

#     # Transformation to sprectral Domain #######################################
#     # Apodization before FT
#     if (apod > 0):
#         cproc.apodisation2d(val, apod)

#     # Filling NaNs
#     valnan = np.copy(val)
#     nan_idx = np.isnan(valnan)  # index of NaNs in the original dataset
#     valfill = cproc.fillnanvalues(valnan, indexout=False)  # Filled dataset

#     # De-meaning
#     valmean = np.nanmean(val)
#     valfill = valfill-valmean

#     # Fourier Transform computation
#     valTF = np.fft.fft2(valfill)  # Frequency domain

#     # Wavenumbers computation
#     ny, nx = val.shape
#     dx = dataset.info.x_gridding_delta
#     dy = dataset.info.y_gridding_delta

#     kx, ky = cproc.wavenumber(nx, ny, dx, dy)  # x, y-directed wavenumber matrices
#     k = np.sqrt(kx**2 + ky**2)  # radial wavenumber matrix
#     indk = k!=0  # index of valid k for the RTP

#     # Spatial derivatives in the frequency domain ##############################
#     #
#     # ... TBD ... make functions for deriv_x, _y, _z of order n of a potential field ?
#     # 
#     # x-directed 1st derivative
#     dval_dxTF = valTF*1j*kx  # x-directed gradient [Blakely 1996, eq12.15, p324]
#     # y-directed 1st derivative
#     dval_dyTF = valTF*1j*ky  # y-directed gradient [Blakely 1996, eq12.16, p324]
#     # vertical 1st derivative
#     dval_dzTF = valTF*k  #  vertical gradient [Blakely 1996, eq12.13, p323]

#     # Potential field derivatives in the spatial domain ########################
#     dval_dx = np.real(np.fft.ifft2(dval_dxTF))
#     dval_dy = np.real(np.fft.ifft2(dval_dyTF))
#     dval_dz = np.real(np.fft.ifft2(dval_dzTF))

#     # Analytic signal amplitude in the spatial domain ##########################
#     A = np.sqrt( dval_dx**2 + dval_dy**2 + dval_dz**2)

#     # Unfilling dataset ########################################################
#     A[nan_idx] = np.nan

#     dataset.data.z_image = A

#     return dataset


def magconfigconversion(dataset, fromconfig, toconfig, apod=0, FromBottomSensorAlt=0.3, FromTopSensorAlt=1.0, ToBottomSensorAlt=0.3, ToTopSensorAlt=1.0, inclination=65, declination=0, azimuth=0, magazimuth=None):
    ''' Conversion between the different sensors configurations.

    cf. :meth:`~geophpy.dataset.DataSet.magconfigconversion`

    '''
##########
##
## Original code has been re-implemented in a less fortranic way
##
############
##   
##   case, sense = _definecaseandsense(prosptechused, prosptechsim)
##
##   if ((case != None) and (sense != None)):
##
##      val = dataset.data.z_image
##
##      # Calculation of lines and columns numbers
##      ny = len(val)
##      nx = len(val[0])
##      # Calculation of total values number
##      nt = nx * ny
##
##      nan_indexes = np.isnan(val)                                 # searches the 'nan' indexes in the initial array
##      if(np.isnan(val).any()):                                    # if at least 1 'nan' is detected in the array,
##      
##         x = np.linspace(dataset.info.x_min, dataset.info.x_max, nx, endpoint=True)
##         y = np.linspace(dataset.info.y_min, dataset.info.y_max, ny, endpoint=True)
##         _fillnanvalues(x, y, val.T)                              # fills the 'nan' values by interpolated values
##
##      if (apod > 0):
##         apodisation2d(val, apod)
##
##      # complex conversion, val[][] -> cval[][]
##      cval = np.array(val, dtype=complex)
##
##      deltax = dataset.info.x_gridding_delta
##      deltay = dataset.info.y_gridding_delta
##
##      # fast fourier series calculation
##      cvalfft = np.fft.fft2(cval)
##
##      if (case >= 2):
##         # deg->rad angle conversions
##         rinc = (inclineangle*np.pi)/180 # inclination in radians
##         ralpha = (alphaangle*np.pi)/180  # alpha angle in radians
##         cosu = np.absolute(np.cos(rinc) * np.cos(ralpha))
##         cosv = np.absolute(np.cos(rinc) * np.sin(ralpha))
##         cosz = np.sin(rinc)
##
##      for ix in range(nx):                                        # for each column
##         for iy in range(ny):                                     # for each line
##            if ((ix != 0) or (iy != 0)):                          # if not first point of array
##               cu, cv = _freq(ix, iy, deltax, deltay, nx, ny)     # complex u and v calculation
##               cz = np.sqrt(np.square(cu) + np.square(cv))
##
##               if (case == 1):
##                  c1 = (downsensoraltsim - downsensoraltused)*2.*np.pi*cz
##                  c2 = (upsensoraltsim - downsensoraltsim)*2.*np.pi*cz
##                  c = np.exp(-c1) - np.exp(-c2)
##               elif (case == 2):
##                  c1 = (downsensoraltsim - downsensoraltused)*2.*np.pi*cz
##                  c2 = (upsensoraltsim - downsensoraltsim)*2.*np.pi*cz
##                  cred = complex(cz*cosz, cu*cosu + cv*cosv)
##                  c = ((np.exp(-c1) - np.exp(-c2))*cz)/cred
##               else :                  # case = 3
##                  cred = complex(cz*cosz, cu*cosu + cv*cosv)
##                  c = cz/cred
##                  
##               if (sense == 0):
##                  coef = c
##               else:
##                  coef = 1./c
##
##               cvalfft[iy][ix] = cvalfft[iy][ix] * coef
##               
##      # FFT inversion
##      icvalfft = np.fft.ifft2(cvalfft)
##      val = icvalfft.real
##
##      val[nan_indexes] = np.nan                                   # set the initial 'nan' values, as in the initial array
##   
##      dataset.data.z_image = val                                  # to get the 'z_image' array in (x,y) format.

    val = dataset.data.z_image

    # Transformation to sprectral Domain #######################################
    # Apodization before FT
    if (apod > 0):
        cproc.apodisation2d(val, apod)

    # Filling NaNs
    valnan = np.copy(val)
    nan_idx = np.isnan(valnan)  # index of NaNs in the original dataset
    valfill = cproc.fillnanvalues(valnan, indexout=False)  # Filled dataset

##    # De-meaning
##    valmean = np.nanmean(val)
##    valfill = valfill-valmean

    # Fourier Transform computation
    valTF = np.fft.fft2(valfill)  # Frequency domain

    # Magnetic azimuth #########################################################
    ## The angle between the survey profile direction and the ambient magnetic
    ## field is phi = declination-azimuth.
    ## if the given angle is directly the angle between the ambient magnetic
    ## field the survey direction (phi), we assume
    ## declinaison = 0 so -phi = azimuth
    if magazimuth is not None:
        # ambient field
        phi = magazimuth  # given azimuth is magnetic azimuth
        declination = 0
        azimuth  = -phi

    # Wavenumbers computation  #################################################
    ny, nx = val.shape
    dx = dataset.info.x_gridding_delta
    dy = dataset.info.y_gridding_delta

    kx, ky = cproc.wavenumber(nx, ny, dx, dy)  # x, y-directed wavenumber matrices
    k = np.sqrt(kx**2 + ky**2)  # radial wavenumber matrix
    indk = k!=0  # index k valid for the RTP

    # Configuration transformation ###############################################
    ###
    # ...TBD... use a dictionnary to choose the convertion method ?
    #
    # _gradmagfield_conversion_chooser(config_in, config_out)
    #
    ###
    # Total-field to total-field vertical gradient
    if fromconfig=='TotalField' and toconfig=='TotalFieldGradient':
        z0 = FromBottomSensorAlt
        znew = ToBottomSensorAlt 
        separation = ToTopSensorAlt - ToBottomSensorAlt
        
        F = _field_to_grad(separation, k, z0=z0, znew=znew)

    # Total-field vertical gradient to total-field 
    elif fromconfig=='TotalFieldGradient' and toconfig=='TotalField':
        z0 = FromBottomSensorAlt
        znew = ToBottomSensorAlt
        separation = FromTopSensorAlt - FromBottomSensorAlt

        F = _grad_to_field(separation, k, z0=z0, znew=znew)

    # Total-field vertical gradient to fluxgate
    elif fromconfig=='TotalFieldGradient' and toconfig=='Fluxgate':
        F = _grad_to_fluxgate(inclination, declination, azimuth, kx, ky)

    # Fluxgate to total-field vertical gradient
    elif fromconfig=='Fluxgate' and toconfig=='TotalFieldGradient':
        F = _fluxgate_to_grad(inclination, declination, azimuth, kx, ky)

    # Total-field to fluxgate
    elif fromconfig=='TotalField' and toconfig=='Fluxgate':
        z0 = FromBottomSensorAlt
        znew = ToBottomSensorAlt 
        separation = ToTopSensorAlt - ToBottomSensorAlt

        F = _field_to_fluxgate(inclination, declination, azimuth, kx, ky, separation, z0=z0, znew=znew)

    # Fluxgate to total-field
    elif fromconfig=='Fluxgate' and toconfig=='TotalField':
        z0 = FromBottomSensorAlt
        znew = ToBottomSensorAlt
        separation = FromTopSensorAlt - FromBottomSensorAlt

        F = _fluxgate_to_field(inclination, declination, azimuth, kx, ky, separation, z0=z0, znew=znew)

    # Unknown configuration conversion
    else:
        F = 1
        # Error unknown conversion

    # Applying filter###########################################################
    valTF_filt = valTF*F

    # Transformation nack to spatial domain ####################################
    valfilt = np.fft.ifft2(valTF_filt)  # Spatial domain
    valfilt = np.real(valfilt) #+ valmean  # Re-meaning
    valfilt[nan_idx] = np.nan  # unfilled dataset

    dataset.data.z_image = valfilt

    return dataset


##def _definecaseandsense( prosptechused, prosptechsim):
##   '''
##   define case and sense of conversion
##   case 1 = "Magnetic field" <-> "Magnetic field gradient"
##   case 2 = "Magnetic field" <-> "Vertical component gradient"
##   case 3 = "Magnetic field gradient" <-> "Vertical component gradient"
##   '''
##
##   case = None
##   sense = None
##   if ((prosptechused == prosptechlist[0]) and (prosptechsim == prosptechlist[1])) :
##      case = 1
##      sense = 0
##   elif ((prosptechused == prosptechlist[0]) and (prosptechsim == prosptechlist[2])) :
##      case = 2
##      sense = 0
##   elif ((prosptechused == prosptechlist[1]) and (prosptechsim == prosptechlist[0])) :
##      case = 1
##      sense = 1
##   elif ((prosptechused == prosptechlist[1]) and (prosptechsim == prosptechlist[2])) :
##      case = 3
##      sense = 0
##   elif ((prosptechused == prosptechlist[2]) and (prosptechsim == prosptechlist[0])) :
##      case = 2
##      sense = 1
##   elif ((prosptechused == prosptechlist[2]) and (prosptechsim == prosptechlist[1])) :
##      case = 3
##      sense = 1
##
##   return case, sense


##def _gradmagfield_conversion_chooser(config_in, config_out):
##    '''
##    Return the appropriate configuration converter.
##
##    Parameters
##    ----------
##
##    config_in, config_out : str, {"TotalField", "FieldGradient", "VerticalComponentGradient"} (list from getsensorconfiglist())
##        Input and ouput magnetic survey configuration.
##
##    '''
##
##    # Checking for unvalid survey configuration
##    valid_config = ', '.join(getsensorconfiglist())
##    for config in [config_in, config_out]:
##        if config not in getsensorconfiglist():
##            raise ValueError(('Invalid survey configuration encountered. '
##                              'Valid survey configurations are %s, '
##                              'but %s encountered') %(valid_config, config)
##                             )
##
##    # Converter chooser definition
##    config_chooser = {
##        'TotalField'         : 'Field',
##        'TotalFieldGradient' : 'Grad',
##        'Fluxgate'           : 'Fluxgate',
##        }
##
##    converter_chooser ={
##        'FieldToGrad'     : _field_to_grad,
##        'GradToField'     : _grad_to_field,
##        'FieldToFluxgate' : _field_to_fluxgate,
##        'FluxgateToField' : _vertical_to_field,
##        'GradToFluxgate'  : _grad_to_fluxgate,
##        'FluxgateToGrad'  : _fluxgate_to_grad,
##        }
##
##    # Choosing input configuration
##    FromConfig = config_chooser[config_in]
##    ToConfig = config_chooser[config_out]
##    conversion_type = ''.join([FromConfig,'To',ToConfig])
##
##    return converter_chooser[conversion_type]


def susceptibility(dataset, prosptech, apod=0, downsensoraltitude = 0.3, upsensoraltitude = 1.0, calculationdepth=.0, stratumthickness=1.0, inclineangle = 65, alphaangle = 0):
   '''
   Dataset magnetic susceptibilty of the equivalent stratum.

   cf. :meth:`~geophpy.dataset.DataSet.susceptibility`

   '''

##   val = dataset.data.z_image
##   
##   # Calculation of lines and columns numbers
##   ny = len(val)
##   nx = len(val[0])
##   # Calculation of total values number
##   nt = nx * ny
##
##   nan_indexes = np.isnan(val)                                 # searches the 'nan' indexes in the initial array
##   if(np.isnan(val).any()):                                    # if at least 1 'nan' is detected in the array,
##      
##      x = np.linspace(dataset.info.x_min, dataset.info.x_max, nx, endpoint=True)
##      y = np.linspace(dataset.info.y_min, dataset.info.y_max, ny, endpoint=True)
##      _fillnanvalues(x, y, val.T)                                # fills the 'nan' values by interpolated values
##
##   if (apod > 0):
##      cproc.apodisation2d(val, apod)
##
##   # complex conversion, val[][] -> cval[][]
##   coef = 1./(400.*np.pi)
##   cval = np.array(val*coef, dtype=complex)
##
##   # fast fourier series calculation
##   cvalfft = np.fft.fft2(cval)
##
##   # filter application
##   # deg->rad angle conversions
##   rinc = (inclineangle*np.pi)/180 # inclination in radians
##   ralpha = (alphaangle*np.pi)/180  # alpha angle in radians
##
##   cosu = np.absolute(np.cos(rinc) * np.cos(ralpha))
##   cosv = np.absolute(np.cos(rinc) * np.sin(ralpha))
##   cosz = np.sin(rinc)
##
##   dz = downsensoraltitude - upsensoraltitude
##   zp = calculationdepth + downsensoraltitude
##   
##   deltax = dataset.info.x_gridding_delta
##   deltay = dataset.info.y_gridding_delta
##
##   if (inclineangle == 0):
##      teta = np.pi/2
##   elif (inclineangle == 90) :    # pi/2
##      teta = np.pi
##   else :
##      teta = np.pi + np.arctan(-2/np.tan(rinc))

   # Field Module, medium field along I inclination calcultation if bipolar field
   FM = 14.722 * (4*(np.square(np.cos(teta)) + np.square(np.sin(teta))))

   cvalfft[0][0] = 0.
   for ix in range(nx):                                        # for each column
      for iy in range(ny):                                     # for each line
         if ((ix != 0) or (iy != 0)):                          # if not first point of array
            cu, cv = _freq(ix, iy, deltax, deltay, nx, ny)     # complex u and v calculation
            # continuation
            cz = np.sqrt(np.square(cu) + np.square(cv))
            cvalfft[iy][ix] = cvalfft[iy][ix] * complex(np.exp(2*np.pi*cz*zp), 0.)
            # pole reduction with potential calculation(and not with field as standard pole reduction)
            cred = complex(cz*cosz, cu*cosu + cv*cosv)
            cvalfft[iy][ix] = (cvalfft[iy][ix] * complex(cz,0))/(2*np.pi*np.square(cred))

            if (prosptech == prosptechlist[1]):                         # if prospection technic is magnetic field gradient
               cvalfft[iy][ix] = cvalfft[iy][ix] / (1-np.exp(dz*cz))         

   # FFT inversion
   icvalfft = np.fft.ifft2(cvalfft)

   # Equivalent stratum thickness
   val = (icvalfft.real*2*100000)/(FM*stratumthickness)         # 100000 because susceptibiliy in 10^-5 SI

   val[nan_indexes] = np.nan                                   # set the initial 'nan' values, as in the initial array

   dataset.data.z_image = val                                  # to get the 'z_image' array in (x,y) format.



def getsensorconfiglist():
    ''' Returns the list of available magnetic sensor configurations. '''

    return sensorconfiglist

def getstructuralindexlist():
    ''' Returns the list of available tructural index for Euler deconvolution. '''

    return structuralindexlist

##def getprosptechlist():
##   '''
##   Get list of prospection technicals availables
##
##   Returns : list of prospection technicals
##   
##   '''
##
##   return prosptechlist


def _continu(z, k, direction='auto'):
    r''' Upward/Downward continuation operator in the spectral domain.

    Parameters
    ----------
    z : float
        The continuation distance.

    k : array
        The radial wavenumber (:math:`k = \sqrt{k_x^2 + k_y^2 }.

    direction : str {'auto', 'up', 'down'}, optional
        Direction of the continuation. If 'auto', the direction is determined
        from the continuation distance sign (if z>0, 'up'; if z<0, 'down').
    '''

    if direction=='auto':
        if z>=0:
            direction = 'up'
        else:
            direction = 'down'

    dir_selector = {'up': _contiup, 'down': _contidwn}

    return dir_selector[direction](z,k)


def _contiup(z, k):
    r''' Upward continuation operator in the spectral domain.

    Parameters
    ----------
    z : float
        The continuation distance.

    k : array
        The radial wavenumber (:math:`k = \sqrt{k_x^2 + k_y^2 }`.
    '''

    dz = np.abs(z)

    return np.exp(-dz*k)  # [Blakely 1996, eq12.8, p317]


def _contidwn(z, k):
    r''' Downward continuation operator in the spectral domain.

    Parameters
    ----------
    z : float
        The continuation distance.

    k : array
        The radial wavenumber (:math:`k = \sqrt{k_x^2 + k_y^2 }`.
    '''

    dz = np.abs(z)

    return np.exp(+dz*k)  # [Blakely 1996, p320]


def _field_to_grad(separation, k, z0=None, znew=None):
    r''' Total-field to gradient operator.

    Total-field magnitude to vertical gradient of the total-field magnitude
    operator in the spectral domain.

    The vertical gradient of the total-field (magnitude) is a simple subtraction of two
    total-field (magnitude) measures at two different heights.

    Parameters
    ----------
    separation : float
        Top-Bottom sensors separation.
        If **separation > 0**, the classic **bottom-top** gradient operator is returned. 
        If **separation < 0**, the **top-bottom** gradient operator is returned.

    k : array
        The radial wavenumber (:math:`k = \sqrt{k_x^2 + k_y^2 }`.
        
    z0 : float
        Bottom sensor altitude.
        
    znew : float
        Bottom sensor new wanted altitude for the gradient.

    Returns
    -------
    F :  array
        The total-field to gradient operator in the spectral domain.

    Notes
    -----
    If current and new survey altitudes are given, the vertical gradient of the
    total-field is computed using three simple steps:

    1. simulation of the bottom sensor total-field to the new altitude
    by continuation;
    2. simulation of the top sensor total-field to the new altitude + separation
    by continuation;
    3. subtraction of the simulated top and bottom sensor total-field.

    References
    ----------

    See also
    --------
    _grad_to_field, _grad_to_fluxgate, _fluxgate_to_grad, _field_to_fluxgate, _fluxgate_to_field

    '''

    # top-bottom sensor separation
    ds = separation

    # Changes in altitude ######################################################
    if z0 is not None and znew is not None:
        # Bottom sensor distance to new altitude
        dh = znew-z0

        # bottom sensor at new altitude - top sensor (with new altitude)
        F = _continu(dh, k) - _continu(dh+ds, k)

    # No changes in altitude ###################################################
    else:
        # bottom sensor - top sensor
        F = 1 - _continu(dz, k)

    return F


def _grad_to_field(separation, k, z0=None, znew=None):
    r''' Gradient to total-field operator.

    Vertical gradient of the total-filed magnitude to
    magnitude of the total-field operator in the spectral domain.

    Parameters
    ----------
    separation : float
        Sensor separation.
        If **separation > 0**, the classic **bottom-top** gradient operator is returned. 
        If **separation < 0**, the **top-bottom** gradient operator is returned.

    k : array
        The radial wavenumber (:math:`k = \sqrt{k_x^2 + k_y^2 }`.
        
    z0 : float
        Bottom sensor altitude.
        
    znew : float
        Bottom sensor new wanted altitude for the gradient.

    Returns
    -------
    F :  array
        The gradient to total-field operator in the spectral domain.

    Notes
    -----

    References
    ----------

    See also
    --------
    _field_to_grad, grad_to_fluxgate, _fluxgate_to_grad, _field_to_fluxgate, _fluxgate_to_field

    '''

    indk = k!=0  # index k valid for the RTP
    F = np.empty(k.shape, dtype=complex)
    F[indk] = ( _field_to_grad(separation, k, z0=z0, znew=znew)[indk] )**-1

    return F


def _grad_to_fluxgate(inclination, declination, azimuth, kx, ky):
    ''' Gradient to fluxgate operator.

    Vertical gradient of the total-filed magnitude to fluxgate
    (gradient of the vertical component of the field) operator
    in the spectral domain.

    Parameters
    ----------
    inclination : float,
        Inclination in degrees positive below horizontal.

    declination : float,
        Declination in degrees positive east of geographic north.
        
    azimuth : float,
        Azimuth of x axis in degrees positive east of north.

    kx, ky : array_like
        The wavenumbers coordinates in the x and y-directions.

    Returns
    -------
    F :  array
        The gradient to fluxgate operator in the spectral domain.

    Notes
    -----
    

    See also
    --------
    _fluxgate_to_grad, _field_to_grad, _grad_to_field, _field_to_fluxgate, _fluxgate_to_field

    '''

    # Unit-vector components of the ambient field
    fx, fy, fz = _dircos(inclination, declination, azimuth)

    # Radial wavenumber matrix
    k = np.sqrt(kx**2 + ky**2)
    indk = k!=0  # index k valid for the RTP

    # RTP operator for the anomaly of the vertical component
    # from [Tabbagh et al 1997]
    F = np.empty(k.shape, dtype=complex)
    F[indk] = k[indk] / ( k[indk]*fz + 1j*(kx[indk]*fx + ky[indk]*fy) )

    return F


def _fluxgate_to_grad(inclination, declination, azimuth, kx, ky):
    ''' Fluxgate to gradient operator.

    Fluxgate (gradient of the vertical component of the field) to 
    vertical gradient of the total-filed magnitude operator 
    in the spectral domain.

    Parameters
    ----------
    inclination : float,
        Inclination in degrees positive below horizontal.

    declination : float,
        Declination in degrees positive east of geographic north.
        
    azimuth : float,
        Azimuth of x axis in degrees positive east of north.

    kx, ky : array_like
        The wavenumbers coordinates in the x and y-directions.

    Returns
    -------
    F :  array
        The fluxgate to gradient operator in the spectral domain.

    Notes
    -----

    See also
    --------
    _grad_to_fluxgate, _field_to_grad, _grad_to_field, _field_to_fluxgate, _fluxgate_to_field

    '''

    indk = k!=0  # index k valid for the RTP
    F = np.empty(k.shape, dtype=complex)
    F[indk] = ( _grad_to_fluxgate(inclination, declination, azimuth, kx, ky)[indk] )**-1

    return F


def _field_to_fluxgate(inclination, declination, azimuth, kx, ky, separation, z0=None, znew=None):
    ''' Total-field to fluxgate operator.

    Total-field magnitude to fluxgate (gradient of the vertical
    component of the field) operator in the spectral domain. 

    Parameters
    ----------
    inclination : float,
        Inclination in degrees positive below horizontal.

    declination : float,
        Declination in degrees positive east of geographic north.
        
    azimuth : float,
        Azimuth of x axis in degrees positive east of north.

    kx, ky : array_like
        The wavenumbers coordinates in the x and y-directions.

    separation : float
        Sensor separation.
        If **separation > 0**, the classic **bottom-top** gradient operator is returned. 
        If **separation < 0**, the **top-bottom** gradient operator is returned.

    k : array
        The radial wavenumber (:math:`k = \sqrt{k_x^2 + k_y^2 }`.
        
    z0 : float
        Bottom sensor altitude.
        
    znew : float
        Bottom sensor new wanted altitude for the gradient.

    Returns
    -------
    F :  array
        The field to fluxgate operator in the spectral domain.

    See also
    --------
    _fluxgate_to_field, _field_to_grad, _grad_to_field, _grad_to_fluxgate, _fluxgate_to_grad

    '''

    # Radial wavenumber matrix
    k = np.sqrt(kx**2 + ky**2)

    # Field to Grad Conversion
    F_field_grad = _field_to_grad(separation, k, z0=z0, znew=znew)

    # Grad to Fluxgate Conversion
    F_grad_flux = _grad_to_fluxgate(inclination, declination, azimuth, kx, ky)

    # Field to Fluxgate Conversion
    F = F_field_grad * F_grad_flux

    return F


def _fluxgate_to_field(inclination, declination, azimuth, kx, ky, separation, z0=None, znew=None):
    ''' Fluxgate to total-field operator.

    Fluxgate (gradient of the vertical component of the field) to
    total-field operator in the spectral domain. 

    Parameters
    ----------
    inclination : float,
        Inclination in degrees positive below horizontal.

    declination : float,
        Declination in degrees positive east of geographic north.
        
    azimuth : float,
        Azimuth of x axis in degrees positive east of north.

    kx, ky : array_like
        The wavenumbers coordinates in the x and y-directions.

    separation : float
        Sensor separation.
        If **separation > 0**, the classic **bottom-top** gradient operator is returned. 
        If **separation < 0**, the **top-bottom** gradient operator is returned.

    k : array
        The radial wavenumber (:math:`k = \sqrt{k_x^2 + k_y^2 }`.
        
    z0 : float
        Bottom sensor altitude.
        
    znew : float
        Bottom sensor new wanted altitude for the gradient.

    Returns
    -------
    F :  array
        The fluxgate to field operator in the spectral domain.

    See also
    --------
    _field_to_fluxgate, _field_to_grad, _grad_to_field, _grad_to_fluxgate, _fluxgate_to_grad

    '''

    indk = k!=0  # index k valid for the RTP
    F = np.empty(k.shape, dtype=complex)
    F = ( _grad_to_fluxgate(inclination, declination, azimuth, kx, ky) )**-1
    
    F[indk] = ( _field_to_fluxgate(inclination, declination, azimuth, kx, ky, separation, z0=z0, znew=znew)[indk] )**-1

    return F


def _dircos(inclination, declination, azimuth):
    '''
    Computes the direction cosines (unit-vector components) from
    inclination, declination and azimuth.

    Parameters
    ----------
    inclination : float,
        Inclination in degrees positive below horizontal.

    declination : float,
        Declination in degrees positive east of geographic north.
        
    azimuth : float,
        Azimuth of x axis in degrees positive east of north.

    Returns
    -------
    vx, vy, vz : array_like,
        The three unit-vector components.

    Notes
    -----
    This function is a direct adaptation from the Subroutine B.9.
    "Subroutine to calculate the three direction cosines of a vector from
    its inclination and declination" in (Blakely, 96)[#]_.

    References
    ----------
    .. [#] Blakely R. J. 1996.
        Potential Theory in Gravity and Magnetic Applications. 
        Appendix B, p381. 
        Cambridge University Press.

    '''

    # Conversion of angles to radians
    incl, decl, azim = np.deg2rad([inclination, declination, azimuth])

    # Unit-vector components calculation
    vx = 1*np.cos(incl)*np.cos(decl-azim)  # x-direction cosine
    vy = 1*np.cos(incl)*np.sin(decl-azim)  # y-direction cosine
    vz = 1*np.sin(incl)  # z-direction cosine

    return np.array([vx, vy, vz])


def _freq(ix, iy, deltax, deltay, nc, nl):
   ''' Calculation of spatial frequencies u and v. '''
   nyd = 1 + nl/2
   nxd = 1 + nc/2

   if (iy < nyd):
      cv = (float(iy))/((nl-1)*deltay)
   else:
      cv = float(iy-nl)/((nl-1)*deltay)

   if (ix < nxd):
      cu = (float(ix))/((nc-1)*deltax)
   else:
      cu = float(ix-nc)/((nc-1)*deltax)

   return cu, cv


def _fillnanvalues(xi, yi, val):
   '''
   Fills the 'nan' values by interpolated values using simple spline interpolation method !
   '''
   for profile in val:                                         # for each profile,
      if (np.isnan(profile).any()):                            # if one 'nan' at least in the profile, does the completion
         nan_indexes = np.isnan(profile)
         data_indexes = np.logical_not(nan_indexes)
         valid_data = profile[data_indexes]
         interpolated = np.interp(nan_indexes.nonzero()[0], data_indexes.nonzero()[0], valid_data)   
         profile[nan_indexes] = interpolated
      
