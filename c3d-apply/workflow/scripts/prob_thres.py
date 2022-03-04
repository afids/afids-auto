#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import re

import nibabel as nib
import numpy as np
from skimage import measure


def sorted_nicely(lst):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    sorted_lst = sorted(lst, key=alphanum_key)

    return sorted_lst


debug = False

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


option = 1


warped_img_obj = nib.load(snakemake.input.warped_img)
img_data = warped_img_obj.get_fdata()
afid_prob_vol_out = np.empty(img_data.shape)


afid_num = 1

for iprob in sorted_nicely(snakemake.input.prob_map):

    # load up afid probability
    afid_prob_obj = nib.load(iprob)
    afid_prob_vol = afid_prob_obj.get_fdata().squeeze(3)

    if option == 1:  # just use max prob minus .15
        max_prob = afid_prob_vol.max()
        minThreshold = max_prob - 0.1
    elif option == 2:  # dynamically set using cumulative density
        hist_y, hist_x = np.histogram(afid_prob_vol.flatten(), bins=100)
        hist_x = hist_x[0:-1]
        cumHist_y = np.cumsum(hist_y.astype(float)) / np.prod(
            np.array(afid_prob_vol.shape)
        )

        # The background should contain half of the voxels
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

    labels, n_labels = measure.label(
        afid_prob_vol_binary.astype(int), background=0, return_num=True
    )

    # find connected components remaining
    properties = measure.regionprops(labels)
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
                ]
            )

    # sort the candidate components based on mean probability value of area
    # take the component with highest mean
    if areaIdxs:
        areaIdxs.sort(key=lambda x: x[1], reverse=True)
        afid_prob_vol_out[labels == areaIdxs[0][-1]] = afid_num

    afid_num += 1

# write out combined map
ni2_concat = nib.Nifti1Image(afid_prob_vol_out, warped_img_obj.affine)
ni2_concat.to_filename(snakemake.output.prob_combined)
