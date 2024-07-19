# liric-reduce

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Reduce Liverpool Telescope [LIRIC](https://telescope.livjm.ac.uk/TelInst/Inst/LIRIC/) near-infrared CMOS data.

## Description

* Split a directory of raw LIRIC data into groups of individual frames.

For each group:
* Combine individual frames in pixel-space to produce one or more sky frames.
* Subtract the sky frame from the individual frames.
* Divide the individual frames by a flat field.
* Align the individual frames using cross-correlation near a bright source.
* Combine the aligned individual frames into a single stacked frame.

## Installation

If desired, create and activate a virtual environment, in which to install the package. E.g.,
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Clone the repository and install the package:
```bash
git clone https://github.com/Lyalpha/liric-reduce
cd liric-reduce
pip install .
# Or install without the use of pip:
# python setup.py install
```

## Usage

The package creates an entry point script `liric-reduce` that can be used to reduce LIRIC data (so long as your virtual 
environment is active):

```bash
liric-reduce --help
```

The data in `input_directory` will be processed in groups of observations that are split based on unique combinations
of `OBJECT`, `FILTER1`, and `NUDGEOFF` in the headers - i.e. the science object, the bandpass, and the dither pattern.

Typically one would want to organise raw data in nightly directories in this case, unless you really want to combine
observations across multiple nights.

The reduced data will be saved in the `output_directory`. An example of a typical usage (not including optional 
arguments) would be:

```bash
liric-reduce /path/to/raw/data/20240719/ /path/to/output/directory/ /path/to/flat_j.fits /path/to/flat_h.fits
```

### Interactive alignment

The package will present the first individual frame of each group of observations to the user for interactive alignment.
This requires the user to select a bounding box around a bright source in the field. The extent of this bounding box
will be used with a cross-correlation technique to align the frames. In the absence of bright sources in the individual
frames, there is currently no good alignment method implemented, and your results will be poor.

### Calling the package from within Python

There is `liric_reduce.reduce.main` function that can be called from within Python.

```python
from liric_reduce.reduce import main
main('/path/to/raw/data/20240719/', '/path/to/output/directory/', '/path/to/flat_j.fits', '/path/to/flat_h.fits')
```

## Issues

The code is a simple first pass at reducing LIRIC data in lieu of an official pipeline. Please report any issues on 
the GitHub issue tracker.