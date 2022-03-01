#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 23:22:24 2022

@author: greydon
"""

import glob
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, axes3d
from skimage import measure, morphology


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
            "probs": f"/home/greydon/Documents/data/autoafids/clinical/derivatives/run_2022-02-22/rf-apply/c3d_rf-apply/sub-120/*_probs.nii.gz"
        }
    )
    snakemake = Namespace(input=input)

voxelCoords = []
afid_num = 1
for iprob in sorted_nicely(glob.glob(snakemake.input.probs)):

    # load up afid probability
    afid_prob_vol = nib.load(iprob).get_fdata().squeeze(3)
    voxel_dims = (nib.load(iprob).header["dim"])[1:4]

    hist_y, hist_x = np.histogram(afid_prob_vol.flatten(), bins=100)
    hist_x = hist_x[0:-1]
    cumHist_y = np.cumsum(hist_y.astype(float)) / np.prod(np.array(afid_prob_vol.shape))

    # The background should contain half of the voxels
    minThreshold_byCount = hist_x[np.where(cumHist_y > 0.90)[0][0]]
    hist_diff = np.diff(hist_y)
    hist_diff_zc = np.where(np.diff(np.sign(hist_diff)) == -2)[0].flatten()
    if len(hist_diff_zc[hist_x[hist_diff_zc] > (minThreshold_byCount)]) == 0:
        minThreshold = hist_x[hist_diff_zc][-1]
    else:
        minThreshold = hist_x[
            hist_diff_zc[hist_x[hist_diff_zc] > (minThreshold_byCount)][0]
        ]

    afid_prob_vol[afid_prob_vol < minThreshold] = 0
    afid_prob_vol_binary = afid_prob_vol > 0

    labels, n_labels = measure.label(
        afid_prob_vol_binary.astype(int), background=0, return_num=True
    )

    properties = measure.regionprops(labels)
    properties.sort(key=lambda x: x.area, reverse=True)
    areas = np.array([prop.area for prop in properties])
    if len(areas) > 10:
        areas = areas[:10]

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

    if areaIdxs:
        areaIdxs.sort(key=lambda x: x[1], reverse=True)
        voxelCoords.append(
            np.c_[
                properties[areaIdxs[0][0]].coords,
                np.repeat(afid_num, len(properties[areaIdxs[0][0]].coords)),
            ]
        )

    afid_num += 1


voxelCoords = np.vstack(voxelCoords)

fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    voxelCoords[:, 0], voxelCoords[:, 1], voxelCoords[:, 2], c=voxelCoords[:, 3], s=2
)
ax.tick_params(axis="x", labelrotation=45)
ax.tick_params(axis="y", labelrotation=45)
ax.set_xlim(0, voxel_dims[0])
ax.set_ylim(0, voxel_dims[1])
ax.set_zlim(0, voxel_dims[2])
ax.set_xlabel("X axis: M-L (mm)", fontsize=14, fontweight="bold", labelpad=14)
ax.set_ylabel("Y axis: A-P (mm)", fontsize=14, fontweight="bold", labelpad=14)
ax.set_zlabel("Z axis: I-S (mm)", fontsize=14, fontweight="bold", labelpad=14)
ax.xaxis._axinfo["label"]["space_factor"] = 3.8
ax.view_init(elev=20, azim=50)
ax.figure.canvas.draw()
fig.tight_layout()
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=[str(x) for x in np.unique(voxelCoords[:, 3])],
)
