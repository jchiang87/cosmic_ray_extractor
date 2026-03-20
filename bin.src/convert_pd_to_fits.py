#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from imsim import write_cosmic_ray_catalog


parser = argparse.ArgumentParser()
parser.add_argument("pd_file_list",
                    help="file containing list of input pandas parquet files")
parser.add_argument("outfile",
                    help="filename for output FITS file")
parser.add_argument("--exptime_per_frame", type=float, default=30.0,
                    help="Exposure time per dark frame (s)")
parser.add_argument("--num_dark_frames", type=int, default=400,
                    help="Number of dark frames")
parser.add_argument("--pixels_per_ccd", type=int, default=16_000_000,
                    help="Number of pixels per CCD.")
parser.add_argument("--nsamp", type=int, default=None,
                    help="Number of pd_files to sample. "
                    "If None, then use all available files.")


args = parser.parse_args()

with open(args.pd_file_list) as fobj:
    pd_files = sorted([_.strip() for _ in fobj.readlines()])

if args.nsamp is not None and len(pd_files) < args.nsamp:
    pd_files = np.random.choice(pd_files, args.nsamp, replace=False)

exptime = len(pd_files) * args.exptime_per_frame * args.num_dark_frames

fp_id_offset = 0
fp_id = []
x0 = []
y0 = []
pixel_values = []
for pd_file in pd_files:
    if "_SW" in pd_file:
        # Skip wavefront CCDs.
        continue
    print(fp_id_offset)
    df0 = pd.read_parquet(pd_file)
    fp_id.extend(df0['index'].to_numpy() + fp_id_offset)
    x0.extend(df0['x0'])
    y0.extend(df0['y0'])
    pixel_values.extend(df0['pixel_values'])
    fp_id_offset = max(fp_id) + 1

write_cosmic_ray_catalog(fp_id, x0, y0, pixel_values, exptime,
                         args.pixels_per_ccd, outfile=args.outfile)
