#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

import nibabel as nib
import numpy as np
from skimage import measure


def sorted_nicely(lst):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    sorted_lst = sorted(lst, key=alphanum_key)

    return sorted_lst


def seg_prob(input_image, prob_map, prob_combined, debug=False):
    # Debug mode
    if debug:
        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        input = dotdict(
            {
                "probs": "/home/greydon/Documents/data/afids-auto/data/OASIS/derivatives/20200301/c3d_rf-apply/sub-0249/*_probs.nii.gz",
                "image": "/home/greydon/Documents/data/afids-auto/data/OASIS/derivatives/20200301/reg_aladin/sub-0249/anat/sub-0249_space-MNI152NLin2009cAsym_res-1mm_T1w.nii.gz",
            }
        )
        snakemake = Namespace(input=input)


    # Load input image
    warped_img_obj = nib.load(input_image)
    img_data = warped_img_obj.get_fdata()

    # Instantiate segmentation volume
    afid_prob_vol_out = np.empty(img_data.shape)

    afid_num = 1
    weighted_centroids = []
    for iprob in sorted_nicely(prob_map):
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

        minThreshold = minThreshold + 0.1

        afid_prob_vol[afid_prob_vol < minThreshold] = 0
        afid_prob_vol_binary = afid_prob_vol > 0

        labels, _ = measure.label(
            afid_prob_vol_binary.astype(int), background=0, return_num=True
        )

        # find connected components remaining weighted by probability
        properties = measure.regionprops(
            label_image=labels,
            intensity_image=afid_prob_vol
        )
        properties.sort(key=lambda x: x.area, reverse=True)
        areas = np.array([prop.area for prop in properties])

        # extract any component with an area less than 100
        areaIdxs = []
        for icomp in range(len(areas)):
            if properties[icomp].area < 100:
                areaIdxs.append(
                    [
                        icomp,
                        np.mean(afid_prob_vol[labels == properties[icomp].label]),
                        properties[icomp].area,
                        properties[icomp].label,
                        properties[icomp].weighted_centroid,
                    ]
                )

        # sort the candidate components based on mean probability value of area
        # take the component with highest mean
        if areaIdxs:
            areaIdxs.sort(key=lambda x: x[1], reverse=True)
            afid_prob_vol_out[labels == areaIdxs[0][-2]] = afid_num

        weighted_centroids.append(areaIdxs[0][-1])

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
        fcsv[line_idx] = fcsv[line_idx].replace(f"afid{fid}_x", str(weighted_centroids[centroid_idx][0]))
        fcsv[line_idx] = fcsv[line_idx].replace(f"afid{fid}_y", str(weighted_centroids[centroid_idx][1]))
        fcsv[line_idx] = fcsv[line_idx].replace(f"afid{fid}_z", str(weighted_centroids[centroid_idx][2]))

    # Write output fcsv
    with open(str(fcsv_output), "w") as f:
        f.write("\n".join(line for line in fcsv))


if __name__ == '__main__':
    weighted_centroids = seg_prob(
        input_image=snakemake.input.warped_img,
        prob_map=snakemake.input.prob_map,
        prob_combined=snakemake.output.prob_combined,
    )

    seg_to_fcsv(
        weighted_centroids=weighted_centroids,
        fcsv_template=snakemake.params.fcsv_template,
        fcsv_output=snakemake.output.fcsv,
    )
