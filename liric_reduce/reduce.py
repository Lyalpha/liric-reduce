import argparse
import datetime
import logging
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch, PercentileInterval, ImageNormalize
from image_registration import chi2_shift
from image_registration.fft_tools import shift
from scipy.ndimage import binary_dilation

from .utils import *

logger = logging.getLogger(__name__)

J = "Barr-J"
H = "FELH1500"

XSIZE = 640
YSIZE = 512

PIXEL_SCALE = 0.29  # arcsecond/pixel

PIX_SHIFT = {
    "SMALL": np.array(
        [
            [0, 0],
            [14.392, -13.567],
            [-17.908, 18.538],
            [-18.275, -13.621],
            [15.309, 16.882],
            [16.005, -2.723],
            [-16.038, -2.670],
            [1.487, -16.345],
            [0.240, 18.004],
        ]
    ),
    "LARGE": np.array(
        [
            [0, 0],
            [60.034, -60.075],
            [-52.884, 49.002],
            [-53.654, -60.075],
            [62.344, 47.882],
            [61.831, -1.427],
            [-52.370, -0.680],
            [0.752, -60.822],
            [1.522, 48.255],
        ]
    ),
}


def main(input_directory: str, output_directory: str, flat_j: Optional[str] = None, flat_h: Optional[str] = None,
         sigma_lower: float = 1.5, sigma_upper: float = 1.5, flat_response_min: float = 0.35, n_sky_frames: int = -1,
         nan_border_size: int = 3, save_sky_frames: bool = False, debug_output: bool = False):
    """
    Main function to process a directory of LIRAC FITS files for reduction.

    Parameters
    ----------
    input_directory : str
        The directory containing LIRIC FITS files to reduce.
    output_directory : str
        The directory to save the reduced FITS files.
    flat_j : str, optional
        The filepath of the J-band flat fits image to use.
    flat_h : str, optional
        The filepath of the H-band flat fits image to use.
    sigma_lower : float
        Lower sigma clipping threshold for sky frame creation.
    sigma_upper : float
        Upper sigma clipping threshold for sky frame creation.
    flat_response_min : float
        Minimum acceptable response level in the flat for a pixel to be used. Values below this value
        will be masked in individual frames.
    n_sky_frames : int
        Number of sky frames to create per group by equally splitting the number of observations.
        If left as -1, then it will create a sky_frame per 9 coadd single exposures to match the dither
        pattern repeat.
    nan_border_size : int
        Size of border in pixels to convert to nans to remove edge effects.
    save_sky_frames : bool
        Whether to save all created sky frames. They will be saved in the output directory.
    debug_output : bool
        Whether to output intermediate images for debugging. Will save the first 9 images with sky and flat
        subtracted, and those same first 9 images after the alignment stage.
    """
    if not os.path.isdir(input_directory):
        logger.error(f"The provided directory {input_directory} does not exist.")
        return

    if not os.path.isdir(output_directory):
        logger.warning(
            f"The provided directory {output_directory} does not exist. Creating it now."
        )
        os.makedirs(output_directory)

    flats = {}
    for filt, flat_filepath in zip([J, H], [flat_j, flat_h]):
        flats[filt] = get_flat_array(flat_filepath, flat_response_min) if flat_filepath else None

    if n_sky_frames != -1 and n_sky_frames < 1:
        msg = "Number of sky frames must be -1 or greater than 0."
        logger.error(msg)
        raise ValueError(msg)

    # Get directory name
    directory_name = os.path.basename(input_directory)

    logger.info(f"Reading Liric FITS files in {input_directory} into DataFrame.")
    df = read_fits_directory(input_directory)

    if not len(df):
        logger.error(f"No Liric FITS files found in {input_directory}.")
        return

    # Split DataFrame of observations by object and filter
    grouped_df = splt_by_object_filter_nudge(df)

    # Process each group
    for name, group in grouped_df.items():
        obj_name, filt, dither = name
        logger.info(
            f"Processing group ({obj_name}, {filt}, {dither}): {len(group)} observations."
        )
        try:
            flat = flats[filt]
        except KeyError:
            msg = f"Unknown filter {filter}"
            logger.exception(msg)
            raise ValueError(msg)
        if flat is None:
            logger.warning(f"Skipping group due to missing flat for filter {filt}.")
            continue

        output_filename_base = f"{directory_name}_{obj_name}_{filt}_{dither}"
        output_filepath_base = os.path.join(output_directory, output_filename_base)

        logger.info(f"Reading FITS files data into a data cube.")
        data = fits_files_to_datacube(
            group["filepath"], nan_border_size=nan_border_size
        )

        if n_sky_frames == -1:
            split_indices = split_array_equal_size_chunks(np.arange(len(data)), 9)
            _n_sky_frames = len(split_indices)
        else:
            _n_sky_frames = n_sky_frames
            split_indices = np.array_split(np.arange(len(data)), _n_sky_frames)

        logger.info(
            f"Subtracting sky frame from data using {_n_sky_frames} sky frames."
        )
        for i, indices in enumerate(split_indices, 1):
            logger.info(f"Creating sky frame {i} for {len(indices)} observations.")
            sky = make_sky_frame(
                data[indices], sigma_lower=sigma_lower, sigma_upper=sigma_upper
            )
            data[indices] -= sky
            if save_sky_frames:
                output_filepath_sky = f"{output_filepath_base}_sky_{i}.fits"
                logger.info(f"Writing to {output_filepath_sky}")
                fits.writeto(output_filepath_sky, sky, overwrite=True)

        logger.info("Dividing sky-subtracted data by flat field.")
        data /= flat

        if debug_output:
            for i, ((_, row), d) in enumerate(zip(group.iterrows(), data)):
                filepath = row["filepath"]
                mean, median, std = sigma_clipped_stats(d[~np.isnan(d)])
                logger.info(
                    f"{filepath} - Mean: {mean:.1f}, Median: {median:.1f}, Std: {std:.1f}"
                )
                out_filepath = os.path.join(
                    output_directory, os.path.basename(filepath)[:-5] + "_skyflat.fits"
                )
                fits.writeto(
                    out_filepath, data=d, header=fits.getheader(filepath), overwrite=True
                )
                if i == 8:
                    break

        # Remove nudging with integer pixel shifts based on nudging position. Pad arrays so that the final
        # combination includes non-overlapping regions.
        logger.info("Removing nudging with integer pixel shifts.")
        integer_shift = -np.round(PIX_SHIFT[dither] / PIXEL_SCALE).astype(int)
        # get the extrema of the shifts in each direction so that we know what to pad data with
        min_shift = np.min(integer_shift, axis=0)
        max_shift = np.max(integer_shift, axis=0)
        data = np.pad(
            data,
            ((0, 0), (-min_shift[1], max_shift[1]), (-min_shift[0], max_shift[0])),
            mode="constant",
            constant_values=np.nan,
        )
        # roll each frame in the datacube by its corresponding shift based on nudgepos in the dataframe
        logger.info("Rolling data by integer pixel shifts.")
        group["x_offset"] = np.nan
        group["y_offset"] = np.nan
        for i, (row_i, row) in enumerate(group.iterrows()):
            nudgepos = row["nudgepos"]
            data[i] = np.roll(data[i], integer_shift[nudgepos], axis=(1, 0))
            group.at[row_i, "x_offset"] = integer_shift[nudgepos][0]
            group.at[row_i, "y_offset"] = integer_shift[nudgepos][1]

        # Plot the first image with an extent showing the coverage of all frames - i.e.e where they all overlap
        # So taking into account the integer shifts
        y_low = -min_shift[1] + max_shift[1]
        y_upp = YSIZE
        x_low = -min_shift[0] + max_shift[0]
        x_upp = XSIZE
        norm = ImageNormalize(
            data[0][y_low:y_upp, x_low:x_upp],
            interval=PercentileInterval(98),
            stretch=SqrtStretch(),
        )
        while True:
            plt.close()
            fig, ax = plt.subplots(figsize=(7, 8))
            # noinspection PyTypeChecker
            ax.imshow(
                data[0][y_low:y_upp, x_low:x_upp],
                origin="lower",
                norm=norm,
                cmap="bone",
            )

            msg = (
                "Make two clicks designating opposite corners of a rectangular region to use for cross-correlation "
                "alignment. Use a right click to undo your most recent click. Press Enter/Esc to finish.  "
                "Select a small region containing one or more bright sources (galaxies are fine to use). Give a good "
                "margin around the source(s) of the order tens of pixels. "
            )
            logger.info(msg)
            # Add this message split over two lines to the plot
            ax.text(
                0.5,
                -0.1,
                msg,
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=12,
                wrap=True,
            )
            # noinspection PyTypeChecker
            points = plt.ginput(3, timeout=0, show_clicks=True, mouse_stop=None)
            if len(points) == 2:
                break
            logger.warning("You must select exactly two points to define a region.")
        plt.close()

        # Get the corners of the region to combine
        x1, y1 = points[0] + np.array([x_low, y_low])
        x2, y2 = points[1] + np.array([x_low, y_low])
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        region = (slice(y1, y2), slice(x1, x2))

        logger.info("Aligning data using cross-correlation.")

        # Our image to be aligned to
        image_zero = data[0][region]
        for i, (row_i, row) in enumerate(group.iterrows()):
            if i == 0:
                continue  # Skip the first image as it is the reference
            nan_mask = np.isnan(data[i])
            image_offset = data[i][region]
            xoff, yoff, exoff, eyoff = chi2_shift(
                image_zero, image_offset, boundary="constant"
            )
            data[i] = shift.shiftnd(data[i], (-yoff, -xoff))

            # Add these sub-pixel shifts to the integer shifts and store them in the dataframe
            group.at[row_i, "x_offset"] -= xoff
            group.at[row_i, "y_offset"] -= yoff

            # image_registration does not maintain nans, so we need to put them back
            # However, the underlying image has been shifted, so we need to roll the nan mask
            # by the same (integer) amount, then we binary_dilate it to cover any sub-pixel shifts
            nan_mask = np.roll(nan_mask, list(map(int, (-xoff, -yoff))), axis=(1, 0))
            # Note this will also dilate pixels identified as nans due to bad flat values, but this
            # is OK since there are few, and we probably want to ensure they are masked in the interpolated
            # aligned image fully anyway
            nan_mask = binary_dilation(nan_mask, iterations=1)
            data[i][nan_mask] = np.nan

        if debug_output:
            for i, ((_, row), d) in enumerate(zip(group.iterrows(), data)):
                filepath = row["filepath"]
                mean, median, std = sigma_clipped_stats(d[~np.isnan(d)])
                logger.info(
                    f"{filepath} - Mean: {mean:.1f}, Median: {median:.1f}, Std: {std:.1f}"
                )
                out_filepath = os.path.join(
                    output_directory,
                    os.path.basename(filepath)[:-5] + "_skyflat_aligned.fits",
                )
                fits.writeto(
                    out_filepath, data=d, header=fits.getheader(filepath), overwrite=True
                )
                if i == 8:
                    break

        logger.info("Combining data with sigma-clipping and averaging.")
        combined_data, mask = sig_clip_average_combine(
            data, sigma_lower=3, sigma_upper=3
        )

        logger.info("Creating stacked image header.")
        header = get_first_observation_header(group)
        header["exptime"] = (
            np.sum(group["exptime"]),
            "Total exposure time of lirac-reduce stack",
        )
        header["combnum"] = (len(group), "Number of images in lirac-reduce stack")
        header["history"] = (
            f"Stacked using lirac-reduce at {datetime.datetime.now().isoformat()}"
        )
        header["history"] = (
            f"Stacked {len(group)} images in {directory_name} for {obj_name} ({filt})"
        )
        header["history"] = f"NaN border size set to {nan_border_size} pixels"
        header["history"] = f"Flat field minimum response {flat_response_min})"
        header["history"] = f"Created {_n_sky_frames} sky frames"
        header["history"] = (
            f"Sky frames stacked with -{sigma_lower}, +{sigma_upper} sigma thresholds"
        )
        for key in [
            "nudgepos",
            "expnum",
        ]:
            del header[key]

        logger.info(f"Writing combined image to {output_filepath_base}.fits.")
        fits.writeto(
            output_filepath_base + ".fits",
            data=combined_data,
            header=header,
            overwrite=True,
        )
        group.to_csv(output_filepath_base + ".csv", index=False)

        logging.info(f"Finished processing group ({obj_name}, {filt}, {dither}).")


def run():

    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Process a directory of LIRAC FITS files for reduction."
    )
    parser.add_argument(
        "input_directory", type=str, help="The directory containing the FITS files."
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="The directory to save the reduced FITS files.",
    )
    parser.add_argument(
        "flat_j", type=str, default="", nargs="?", help="The J-band flat fits image."
    )
    parser.add_argument(
        "flat_h", type=str, default="", nargs="?", help="The H-band flat fits image."
    )
    parser.add_argument(
        "--sigma-lower",
        type=float,
        default=2,
        help="Lower sigma clipping threshold for sky frame " "creation.",
    )
    parser.add_argument(
        "--sigma-upper",
        type=float,
        default=2,
        help="Upper sigma clipping threshold for sky frame " "creation.",
    )
    parser.add_argument(
        "--flat-response-min",
        type=float,
        default=0.35,
        help="Minimum acceptable response level in "
             "the flat for a pixel to be used. Any "
             "pixels with a flat value below this "
             "value will be masked.",
    )
    parser.add_argument(
        "--n-sky-frames",
        type=int,
        default=-1,
        help="Number of sky frames to create per group by "
             "equally splitting the group. If -1 then it will "
             " create a sky_frame per 9 coadd single "
             "exposures to match the dither pattern repeat.",
    )
    parser.add_argument(
        "--nan-border-size",
        type=int,
        default=3,
        help="Size of border in pixels to convert to nans to " "remove edge effects.",
    )
    parser.add_argument(
        "--save-sky-frames", action="store_true", help="Save all created sky frames."
    )
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="Output intermediate images for debugging. Will save the first 9 images with sky and flat "
             "subtracted, and those same first 9 images after the alignment stage.",
    )

    args = parser.parse_args()

    main(
        args.input_directory,
        args.output_directory,
        args.flat_j,
        args.flat_h,
        args.sigma_lower,
        args.sigma_upper,
        args.flat_response_min,
        args.n_sky_frames,
        args.nan_border_size,
        args.save_sky_frames,
        args.debug_output,
    )


if __name__ == "__main__":
    run()
