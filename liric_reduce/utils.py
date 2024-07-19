import functools
import logging
import os
import warnings

import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.stats import sigma_clip, mad_std
from astropy.time import Time
from astropy.utils.exceptions import AstropyUserWarning


SHAPE_0 = 512
SHAPE_1 = 640

__all__ = [
    "split_array_equal_size_chunks",
    "get_flat_array",
    "get_first_observation_header",
    "read_fits_directory",
    "splt_by_object_filter_nudge",
    "fits_files_to_datacube",
    "sig_clip_average_combine",
    "make_sky_frame",
]

# Need to create a version of mad_std that ignores nans by default so
# that it can be used as the std func when sigma clipping correctly
nan_mad_std = functools.partial(mad_std, ignore_nan=True)


def split_array_equal_size_chunks(a, chunk_size):
    """Return a list of arrays each with length chunk_size, with potentially a remainder array."""
    return np.split(a, np.arange(chunk_size, len(a), chunk_size))


def get_flat_array(fits_filepath, min_pixel_value):
    """Return array of flat fits file."""
    flat = fits.open(fits_filepath)[0].data
    flat[flat < min_pixel_value] = np.nan
    return flat


def get_first_observation_header(df):
    """Return the header of the first (chronological) observation."""
    filepath = df.iloc[np.argmin(df["mjd"])]["filepath"]
    return fits.getheader(filepath, 0)


def read_fits_directory(directory):
    """
    Read all FITS files in a directory and return a DataFrame with the relevant header information.
    """
    fits_files = [f for f in os.listdir(directory) if f.endswith(".fits")]
    records = []
    keywords = [
        "date-obs",
        "mjd",
        "object",
        "filter1",
        "exptime",
        "coaddsec",
        "coaddnum",
        "nudgeoff",
        "nudgepos",
        "expnum",
        "exptotal",
        "ra",
        "dec",
        "ccdximsi",
        "ccdyimsi",
        "propid",
    ]

    for file in fits_files:
        with fits.open(os.path.join(directory, file)) as hdul:
            header = hdul[0].header
            if header["instrume"].strip() != "Liric":
                logging.warning(f"Skipping {file} because it is not a Liric FITS file.")

            record = {keyword: header.get(keyword) for keyword in keywords}
            record["filepath"] = os.path.join(directory, file)

            # Convert date-obs to a datetime object in UTC format
            if record["date-obs"]:
                record["date-obs"] = Time(record["date-obs"]).datetime

            # Convert ra, dec to astropy quantities
            if record["ra"]:
                record["ra"] = Angle(record["ra"], unit="hourangle").deg
            if record["dec"]:
                record["dec"] = Angle(record["dec"], unit="deg").deg

            records.append(record)

    return pd.DataFrame(records).sort_values("mjd")


def splt_by_object_filter_nudge(df):
    """
    Split a DataFrame of observations by object and filter, and return a dictionary of DataFrames.
    """
    return {
        name: group.sort_values("mjd")
        for name, group in df.groupby(["object", "filter1", "nudgeoff"])
    }


def fits_files_to_datacube(fits_files, remove_median=True, nan_border_size=0):
    """
    Read a list of FITS files and return a 3D data cube.
    """
    data = np.empty((len(fits_files), SHAPE_0, SHAPE_1))
    # median = []
    # mean = []
    # std_dev = []
    for i, file in enumerate(fits_files):
        with fits.open(file) as hdul:
            data_ = hdul[0].data
            # mean_, median_, std_dev_ = sigma_clipped_stats(data_, maxiters=2)
            # mean.append(mean)
            data[i] = data_

    if remove_median:
        data -= np.median(data, axis=(1, 2))[:, np.newaxis, np.newaxis]
    if nan_border_size > 0:
        data[:, :nan_border_size, :] = np.nan
        data[:, -nan_border_size:, :] = np.nan
        data[:, :, :nan_border_size] = np.nan
        data[:, :, -nan_border_size:] = np.nan
    return data


def sig_clip_average_combine(
    data=None, weights=None, sigma_lower=2, sigma_upper=2, maxiters=5
):
    """
    Combine a data cube into a single image by sigma-clipping in pixel-space and averaging along axis 0.
    """
    data = np.ma.array(data, mask=False)

    # Perform outlier-robust sigma clipping
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyUserWarning)
        sigma_clipped_data = sigma_clip(
            data=data,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            stdfunc=nan_mad_std,
            cenfunc="median",
            axis=0,
            maxiters=maxiters,
        )
    n_clipped = np.sum(sigma_clipped_data.mask)
    if n_clipped:
        p_clipped = 100 * n_clipped / data.size
        logging.info(f"{n_clipped} ({p_clipped:.3f}%) masked pixels due to clipping")

    combined_data = np.ma.average(
        sigma_clipped_data,
        axis=0,
        weights=weights,
    )

    return combined_data.filled(fill_value=1e-10), sigma_clipped_data.mask


def make_sky_frame(data, weights=None, sigma_lower=2, sigma_upper=2):
    """
    Combine a data cube of observations into a sky image by median-combining all exposures with
    sigma-clipping in pixel-space to remove sources.
    """
    data, _ = sig_clip_average_combine(
        data, weights=weights, sigma_lower=sigma_lower, sigma_upper=sigma_upper
    )
    return data
