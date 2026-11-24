"""Fit aerosol properties to measured UVIS transmittance spectra."""

import os
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d


# name_orb = "20240516_130419"
# num_spec_matlab = [65, 60, 59, 58, 57, 56, 55, 50]
# num_fit_matlab = [41, 36, 35, 34, 33, 32, 31, 26]
# do_no_ozone = True
# which_xs = "Zach"
# reference_aerosol_wavelength = 510.0
# particle_sizes = None
# dust_fraction_step = 0.025
# fit_interval_pixels = [250, 1000]
# plot_results = True
# with_error_bars = True

DATA_DIR = r"C:\Users\iant\Dropbox\NOMAD\Python\repos\Yannick_occ_trans_sim"


def get_slant_columns_from_h5(h5_basename, alts):

    with h5py.File("slant_columns.h5", "a") as f:
        if h5_basename in f.keys():
            print("Slant column found in h5 file")

            alts_l2 = f[h5_basename]["alt"][...]
            columns_l2 = f[h5_basename]["column"][...]
        else:
            print("Error not found")
            return []

    ixs_found = []
    alts_found = []
    ozone_columns_found = []
    rayleigh_columns_found = []
    for i, alt in enumerate(alts):
        ix = np.argmin(np.abs(alt - alts_l2))
        if np.abs(alt - alts_l2)[ix] < 0.1:
            ixs_found.append(i)
            alts_found.append(alt)
            ozone_columns_found.append(columns_l2[0, ix])
            rayleigh_columns_found.append(columns_l2[1, ix])
        else:
            print("%s %i: %0.3f km column not found" % (h5_basename, i, alt))

    # print(np.asarray(ozone_columns_found), np.asarray(rayleigh_columns_found))

    return np.asarray(ixs_found), np.asarray(ozone_columns_found), np.asarray(rayleigh_columns_found)


# testing
# h5_basename = "20240501_033224_0p3d_UVIS_I".replace("0p3d", "1p0a")
# alts = np.mean(np.asarray([
#     [18.48654255, 18.37963834],
#     [16.86945929, 16.76236216],
#     [15.24932599, 15.14197279],
#     [13.62545807, 13.517912  ],
#     [11.99868142, 11.89094381],
# ]), axis=1)
# columns = get_slant_columns_from_h5(h5_basename, alts)


def fit_to_spectra(h5_basename, x, spectra, spectra_error, alts, do_no_ozone=True, which_xs="Zachary", reference_aerosol_wavelength=510.0, particle_sizes=None,
                   dust_fraction_step=0.025, fit_nms=[316, 640], plot_results=True, with_error_bars=True):

    ixs_found, ozone_columns_found, rayleigh_columns_found = get_slant_columns_from_h5(h5_basename, alts)
    binning = 1024 // spectra[0].shape[0]

    wavelengths = x

    # convert nanometres into pixel indices
    fit_interval_pixels = [np.argmin(np.abs(x - fit_nms[0])), np.argmin(np.abs(x - fit_nms[1]))]

    alts_found = np.asarray([alts[i] for i in ixs_found])
    spectra_found = np.asarray([spectra[ix] for ix in ixs_found])
    spectra_error_found = np.asarray([spectra_error[ix] for ix in ixs_found])

    print("%s: indices found: %s/%i" % (h5_basename, ixs_found, len(spectra)))

    def _read_matlab_order_dataset(file: h5py.File, name: str) -> np.ndarray:
        """Read an HDF5 dataset with the dimension order used by MATLAB h5read."""
        data = np.asarray(file[name][()])
        return data.T if data.ndim > 1 else data

    def _moving_average(values: np.ndarray, span: int) -> np.ndarray:
        """Apply a centered moving average, shortening the window at each edge."""
        half_span = span // 2
        invalid = ~np.isfinite(values)
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.where(invalid, 0.0, values), dtype=float))
        )
        invalid_cumulative = np.concatenate(([0], np.cumsum(invalid)))
        indices = np.arange(values.size)
        starts = np.maximum(0, indices - half_span)
        stops = np.minimum(values.size, indices + half_span + 1)
        window_sums = cumulative[stops] - cumulative[starts]
        invalid_windows = invalid_cumulative[stops] - invalid_cumulative[starts]
        return np.where(
            invalid_windows == 0,
            window_sums / (stops - starts),
            np.nan,
        )

    def _linear_interp(
        source_x: np.ndarray,
        source_y: np.ndarray,
        target_x: np.ndarray | float,
        *,
        extrapolate: bool = False,
    ) -> np.ndarray:
        """Linear interpolation matching interp1's default or extrapolate mode."""
        interpolator = interp1d(
            source_x,
            source_y,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate" if extrapolate else np.nan,
        )
        return np.asarray(interpolator(target_x))

    def _aerosol_ratio(
        which_xs: str,
        which_aerosol: str,
        wavelengths: np.ndarray,
        particle_size: float,
        reference_wavelength: float,
        data_dir: str,
    ) -> np.ndarray:
        """Load/interpolate aerosol extinction data and normalize at the reference."""
        if which_xs == "Wolff":
            if which_aerosol == "dust":
                sizes = np.concatenate(([0.1, 0.2], np.arange(0.5, 1.51, 0.25)))
                relative_path = "Aerosol_Cext"
                prefix = "Dust_C22_"
                cross_section_column = 2
            elif which_aerosol == "cloud":
                sizes = np.array([0.1, 0.5, 1.0, 2.0])
                relative_path = "Aerosol_Cext"
                prefix = "Cloud_W19_"
                cross_section_column = 2
            else:
                raise ValueError(f"Unknown aerosol type: {which_aerosol}")
        elif which_xs == "Zachary":
            relative_path = os.path.join("Aerosol_Cext", "Dust_mix_IUVS_SO_Cloud_SO")
            if which_aerosol == "dust":
                sizes = np.arange(0.1, 1.91, 0.2)
                prefix = "dust_"
                cross_section_column = 1
            elif which_aerosol == "cloud":
                sizes = np.arange(0.1, 2.91, 0.2)
                prefix = "cloud_"
                cross_section_column = 1
            else:
                raise ValueError(f"Unknown aerosol type: {which_aerosol}")
        else:
            raise ValueError(f"Unknown cross-section set: {which_xs}")

        lower_indices = np.flatnonzero(sizes <= particle_size)
        upper_indices = np.flatnonzero(sizes >= particle_size)

        if lower_indices.size == 0:
            lower_index = upper_index = 0
        elif upper_indices.size == 0:
            lower_index = upper_index = sizes.size - 1
        else:
            lower_index = lower_indices[-1]
            upper_index = upper_indices[0]

        def read_cross_section(index: int) -> tuple[np.ndarray, np.ndarray]:
            size_name = f"{sizes[index]:4.2f}".replace(".", "p")
            filename = os.path.join(DATA_DIR, relative_path, f"{prefix}{size_name}.txt")
            data = np.loadtxt(filename, ndmin=2)
            return data[:, 0], data[:, cross_section_column]

        wavelengths_xs, lower_cross_section = read_cross_section(lower_index)
        if lower_index == upper_index:
            cross_section = lower_cross_section
        else:
            _, upper_cross_section = read_cross_section(upper_index)
            fraction_lower = (
                (sizes[upper_index] - particle_size) / (sizes[upper_index] - sizes[lower_index])
            )
            cross_section = (
                fraction_lower * np.sqrt(lower_cross_section) + (1.0 - fraction_lower) * np.sqrt(upper_cross_section)
            ) ** 2

        reference_cross_section = _linear_interp(
            wavelengths_xs,
            cross_section,
            reference_wavelength,
        )
        interpolated_cross_section = _linear_interp(
            wavelengths_xs,
            cross_section,
            wavelengths,
            extrapolate=True,
        )
        return interpolated_cross_section / reference_cross_section

    """Fit dust/cloud size and fraction for each selected spectrum.
    
    The spectrum and fit indices are supplied using MATLAB's 1-based numbering,
    so existing selections can be carried over without changing their values.
    """
    # data_dir = Path(__file__).resolve().parent
    # source_path = data_dir / f"{name_orb}_1p0a_UVIS_I.h5"
    # result_path = data_dir / f"{name_orb}_1p0a_UVIS_I_out.h5"
    # num_spec = np.asarray(num_spec_matlab, dtype=int) - 1
    # num_fit = np.asarray(num_fit_matlab, dtype=int) - 1
    # if num_spec.size != num_fit.size:
    #     raise ValueError("num_spec_matlab and num_fit_matlab must have equal lengths")
    if particle_sizes is None:
        particle_sizes = np.arange(0.10, 0.801, 0.01)

    # with h5py.File(source_path, "r") as source:
    #     y_mean = _read_matlab_order_dataset(source, "/Science/YMean")
    #     y_error_mean = _read_matlab_order_dataset(source, "/Science/YErrorMean")
    #     x_dataset = _read_matlab_order_dataset(source, "/Science/X")
    #     wavelengths = x_dataset[:, 0]
    #     binning = int(source["/Channel/HorizontalAndCombinedBinningSize"][0]) + 1
    #     altitude = np.asarray(
    #         source["/Geometry/Point0/TangentAltAreoid"][()]
    #     ).mean(axis=1)[num_spec]
    # if not os.path.exists("slant_columns.h5"):
    #     with h5py.File("slant_columns.h5", "w") as f:
    #         None

    # with h5py.File("slant_columns.h5", "a") as f:
    #     if name_orb in f.keys():
    #         print("Reading slant columns from column file")
    #         slant_columns = f[name_orb][...]
    #     else:
    #         print("Saving slant columns to file")
    #         with h5py.File(result_path, "r") as result:
    #             slant_columns = _read_matlab_order_dataset(result, "/Pass_2/SlantColumns/SlantColumnsFitted")
    #         f.create_dataset(name_orb, dtype=float, data=slant_columns, compression="gzip", shuffle=True)

    ray = np.asarray(loadmat(os.path.join(DATA_DIR, "RAY_UVIS_FitXP.mat"))["RAY"])
    ozone_xs = np.asarray(loadmat(os.path.join(DATA_DIR, "XS_UVIS_O3.mat"))["XS"])

    if binning > 1:
        if ray.shape[0] % binning:
            raise ValueError("Rayleigh cross-section rows are not divisible by binning")
        ray = ray.reshape(-1, binning, ray.shape[1]).mean(axis=1)
        if ozone_xs.shape[0] % binning:
            raise ValueError("Ozone cross-section rows are not divisible by binning")
        ozone_xs = ozone_xs.reshape(-1, binning, ozone_xs.shape[1]).mean(axis=1)

    # fit_start, fit_end = np.floor(
    #     np.asarray(fit_interval_pixels, dtype=float) / binning + 0.5
    # ).astype(int)
    # fit_slice = slice(fit_start - 1, fit_end)
    fit_slice = slice(fit_interval_pixels[0], fit_interval_pixels[1], 1)
    fractions = np.arange(
        0.0, 1.0 + dust_fraction_step / 2.0, dust_fraction_step
    )

    aerosol_properties = {
        aerosol: np.stack(
            [
                _aerosol_ratio(
                    which_xs,
                    aerosol,
                    wavelengths,
                    float(size),
                    reference_aerosol_wavelength,
                    DATA_DIR,
                )
                for size in particle_sizes
            ]
        )
        for aerosol in ("dust", "cloud")
    }

    smooth_reference_transmittance = np.array([
        _linear_interp(wavelengths, _moving_average(spectrum, 11), reference_aerosol_wavelength,).item() for spectrum in spectra_found
    ])
    tau_reference_aerosol = -np.log(smooth_reference_transmittance)

    tau_ozone = (
        ozone_columns_found[None, :] * ozone_xs[:, 1, None]
    )
    if do_no_ozone:
        tau_ozone = np.zeros_like(tau_ozone)
    tau_rayleigh = (
        rayleigh_columns_found[None, :] * ray[:, 1, None] / 0.96
    )

    fitted_transmittance = np.empty((wavelengths.size, ixs_found.shape[0]))
    legend_labels = []

    for spectrum_index in range(len(ixs_found)):
        label = f"Meas {alts_found[spectrum_index]:6.2f} km "
        legend_labels.append(label)
        print(label)
        best_x2 = 1e10
        best_dust_index = best_cloud_index = best_fraction_index = -1

        measured_fit = spectra_found[spectrum_index, fit_slice]
        error_fit = spectra_error_found[spectrum_index, fit_slice]
        denominator = measured_fit.size - 4
        base_opacity_fit = (
            tau_ozone[fit_slice, spectrum_index] + tau_rayleigh[fit_slice, spectrum_index]
        )

        start_time = time.perf_counter()
        for dust_index, dust_ratio in enumerate(aerosol_properties["dust"]):
            dust_opacity = tau_reference_aerosol[spectrum_index] * dust_ratio
            for cloud_index, cloud_ratio in enumerate(aerosol_properties["cloud"]):
                cloud_opacity = tau_reference_aerosol[spectrum_index] * cloud_ratio
                trial_transmittance = np.exp(
                    -(
                        base_opacity_fit[None, :] + dust_opacity[fit_slice][None, :] * fractions[:, None] + cloud_opacity[
                            fit_slice][None, :] * (1.0 - fractions[:, None])
                    )
                )
                x2_values = np.sum(
                    ((measured_fit[None, :] - trial_transmittance) / error_fit[None, :]) ** 2, axis=1,
                ) / denominator
                better = np.flatnonzero(x2_values < best_x2)
                if better.size:
                    fraction_index = better[np.argmin(x2_values[better])]
                    best_x2 = x2_values[fraction_index]
                    best_dust_index = dust_index
                    best_cloud_index = cloud_index
                    best_fraction_index = int(fraction_index)

        if best_fraction_index < 0:
            raise RuntimeError(f"No valid fit found for spectrum {spectrum_index + 1}")

        dust_fraction = fractions[best_fraction_index]
        dust_opacity = (
            tau_reference_aerosol[spectrum_index] * aerosol_properties["dust"][best_dust_index] * dust_fraction
        )
        cloud_opacity = (
            tau_reference_aerosol[spectrum_index] * aerosol_properties["cloud"][best_cloud_index] * (1.0 - dust_fraction)
        )
        fitted_transmittance[:, spectrum_index] = np.exp(
            -(
                tau_ozone[:, spectrum_index] + tau_rayleigh[:, spectrum_index] + dust_opacity + cloud_opacity
            )
        )

        # legend is not correct
        legend_labels.append(
            "Simu: dust= "
            f"{particle_sizes[best_dust_index]:4.2f} µm "
            f"({round(100 * dust_fraction):d}%) & cloud= "
            f"{particle_sizes[best_cloud_index]:4.2f} µm "
            f"({round(100 * (1.0 - dust_fraction)):d}%) | X2= {best_x2:6.4f}"
        )
        print(f"Fit complete in {time.perf_counter() - start_time:.2f} s")

    if plot_results:
        plt.figure(figsize=(16, 8))
        for i in range(len(ixs_found)):
            if with_error_bars:
                plt.errorbar(x, spectra_found[i, :], yerr=spectra_error_found[i, :], color="C%i" % i)
            else:
                plt.plot(x, spectra_found[i, :], color="C%i" % i)

            plt.plot(x, fitted_transmittance[:, i], "--", color="C%i" % i)
        plt.title(h5_basename)
        # plt.xlim(0, 1025)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend(legend_labels)
        plt.show()

    # return best fits and found_indices and fitted spectral indices
    return fitted_transmittance.T, ixs_found, np.r_[fit_slice]


# testing
# filepath = r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_0p3d/2024/05/01/20240501_021829_0p3d_UVIS_E.h5"
# filepath = r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_0p3d/2024/05/16/20240516_130419_0p3d_UVIS_I.h5"

# with h5py.File(filepath, 'r') as f:
#     Y = f["Science/Y"][...]
#     alts = np.mean(f["Geometry/Point0/TangentAltAreoid"][...], axis=1)
#     x = f['Science/X'][0, :]

#     YWellIllumMask = f["Science/YIllumMask"][...]
# Y_illum = Y * YWellIllumMask

# nbr_lambda_pixels = Y.shape[2]

# # first get list of non-illuminated frames with sufficient signal
# middle_col_ix = int(nbr_lambda_pixels / 2)
# middle_col_sums = np.nansum(Y_illum[:, :, middle_col_ix], axis=1)
# middle_col_trans = middle_col_sums / np.nanmax(middle_col_sums)
# toa_ix = np.where(middle_col_trans > 0.99)[0][-1]  # example index above atmosphere
# atmo_ix = np.argmin(np.abs(middle_col_trans - 0.5))

# y_toa_binned = np.sum(Y[toa_ix, :, :], axis=0)
# y_atmos_binned1 = np.sum(Y[atmo_ix, :, :], axis=0)
# y_trans_binned1 = y_atmos_binned1 / y_toa_binned
# y_atmos_binned2 = np.sum(Y[atmo_ix + 1, :, :], axis=0)
# y_trans_binned2 = y_atmos_binned2 / y_toa_binned


# h5_basename = "20240516_130419_0p3d_UVIS_I".replace("0p3d", "1p0a")
# x = x
# spectra = [y_trans_binned1, y_trans_binned2]
# spectra_error = [y_trans_binned1 / 100.0, y_trans_binned2 / 100.0]
# alts_in = [alts[atmo_ix], alts[atmo_ix + 1]]

# y_trans_fitted, _, _ = fit_to_spectra(h5_basename, x, spectra, spectra_error, alts_in, do_no_ozone=True, which_xs="Zachary",
#                                 reference_aerosol_wavelength=510.0, particle_sizes=None,
#                                 dust_fraction_step=0.025, fit_interval_pixels=[250, 1000], plot_results=False, with_error_bars=False)

# for i in range(len(spectra)):
#     plt.plot(spectra[i])
#     plt.plot(y_trans_fitted[i, :])
