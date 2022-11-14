#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import re

import nibabel as nib
import numpy as np
from skimage import measure


logger = logging.getLogger(__name__)


def sorted_nicely(lst):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    sorted_lst = sorted(lst, key=alphanum_key)

    return sorted_lst


def seg_prob(
    input_image,
    prob_map,
    prob_combined,
):
    # Load input image
    warped_img_obj = nib.load(input_image)
    img_affine = warped_img_obj.affine
    img_data = warped_img_obj.get_fdata()

    # Instantiate segmentation volume
    afid_prob_vol_out = np.empty(img_data.shape)

    afid_num = 1
    weighted_centroids = []
    for iprob in sorted_nicely(prob_map):
        logger.info("Handling afid %s", afid_num)
        # load up afid probability
        afid_prob_obj = nib.load(iprob)
        afid_prob_vol = afid_prob_obj.get_fdata().squeeze(3)

        # dynamically set using cumulative density
        hist_y, hist_x = np.histogram(afid_prob_vol.flatten(), bins=100)
        hist_x = hist_x[0:-1]
        cumHist_y = np.cumsum(hist_y.astype(float)) / np.prod(
            np.array(afid_prob_vol.shape)
        )

        # The background should contain half of the voxels
        # Currently uses 90th percentile, can be tuned
        minThreshold_byCount = hist_x[np.where(cumHist_y > 0.9)[0][0]]
        hist_diff = np.diff(hist_y)
        hist_diff_zc = np.where(np.diff(np.sign(hist_diff)) == -2)[0].flatten()
        if len(hist_diff_zc[hist_x[hist_diff_zc] > (minThreshold_byCount)]) == 0:
            minThreshold = hist_x[hist_diff_zc][-1]
        else:
            minThreshold = hist_x[
                hist_diff_zc[hist_x[hist_diff_zc] > (minThreshold_byCount)][0]
            ]
        minThreshold += 0.1

        afid_prob_vol[afid_prob_vol < minThreshold] = 0
        afid_prob_vol_binary = afid_prob_vol > 0

        labels, _ = measure.label(
            afid_prob_vol_binary.astype(int), background=0, return_num=True
        )

        # find connected components remaining weighted by probability
        regions = measure.regionprops(label_image=labels, intensity_image=afid_prob_vol)
        regions.sort(key=lambda x: x.area, reverse=True)
        areas = np.array([region.area for region in regions])
        logger.info("CC areas: %s", areas)

        area_idxs = []
        for idx, region in enumerate(regions):
            if region.area > 100:
                logger.warning(
                    "Using a region with an unusually large area (%s)", region.area
                )
            area_idxs.append(
                [
                    idx,
                    np.mean(afid_prob_vol[labels == region.label]),
                    region.area,
                    region.label,
                    region.weighted_centroid,
                ]
            )

        if not area_idxs:
            logger.warning("No appropriate region found for AFID %s", afid_num)
            raise ValueError(
                f"No appropriate region found for AFID {afid_num} in image "
                f"{iprob}.\n"
                "Areas:\n"
                f"{areas}"
            )

        # sort the candidate components based on mean probability value of area
        # take the component with highest mean
        area_idxs.sort(key=lambda x: x[1], reverse=True)
        afid_prob_vol_out[labels == area_idxs[0][-2]] = afid_num

        weighted_centroids.append(
            img_affine[:3, :3].dot(area_idxs[0][-1]) + img_affine[:3, 3]
        )

        # Move onto next fiducial
        afid_num += 1

    # write out combined map
    ni2_concat = nib.Nifti1Image(afid_prob_vol_out, warped_img_obj.affine)
    ni2_concat.to_filename(prob_combined)

    return weighted_centroids


def seg_to_fcsv(weighted_centroids, fcsv_template, fcsv_output):
    # Read in fcsv template
    with open(fcsv_template, "r") as f:
        fcsv = [line.strip() for line in f]

    # Loop over fiducials
    for fid in range(1, 33):
        # Update fcsv, skipping header
        line_idx = fid + 2
        centroid_idx = fid - 1
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_x", str(weighted_centroids[centroid_idx][0])
        )
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_y", str(weighted_centroids[centroid_idx][1])
        )
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_z", str(weighted_centroids[centroid_idx][2])
        )

    # Write output fcsv
    with open(fcsv_output, "w") as f:
        f.write("\n".join(line for line in fcsv))


if __name__ == "__main__":
    weighted_centroids = seg_prob(
        input_image=snakemake.input.warped_img,
        prob_map=snakemake.input.prob_map,
        prob_combined=snakemake.output.prob_combined,
    )

    seg_to_fcsv(
        weighted_centroids=weighted_centroids,
        fcsv_template=snakemake.params.fcsv_template,
        fcsv_output=str(snakemake.output.fcsv),
    )
