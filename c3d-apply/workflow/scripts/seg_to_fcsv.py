#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np


def compute_centroid(seg, affine, fid):
    # Get slice indices matching fiducial
    x, y, z = np.where(seg == fid)

    # Compute centroid
    # Better performance than np.mean()
    centroid = [np.sum(x) / len(x), np.sum(y) / len(y), np.sum(z) / len(z)]

    # Return centroid in world coordinates
    return affine[:3, :3].dot(centroid) + affine[:3, 3]


## MAIN ##
# Grab data and affine
prob_seg = nib.load(snakemake.input.seg)
prob_seg_affine = prob_seg.affine
prob_seg_data = prob_seg.get_fdata()

# Read in fcsv template
with open(snakemake.input.fcsv_template, "r") as f:
    fcsv = [line.strip() for line in f]

# Loop over fiducials
for fid in range(1, 33):
    # Compute centroid in world coordinates
    centroid_mm = compute_centroid(prob_seg_data, prob_seg_affine, fid)

    # Update fcsv, skipping header lines
    line_idx = fid + 2
    fcsv[line_idx] = fcsv[line_idx].replace(f"afid{fid}_x", str(centroid_mm[0]))
    fcsv[line_idx] = fcsv[line_idx].replace(f"afid{fid}_y", str(centroid_mm[1]))
    fcsv[line_idx] = fcsv[line_idx].replace(f"afid{fid}_z", str(centroid_mm[2]))

# Write output fcsv
with open(snakemake.output.fcsv, "w") as f:
    f.write("\n".join(line for line in fcsv))
