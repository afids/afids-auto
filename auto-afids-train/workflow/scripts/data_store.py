#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:26:50 2020

@author: greydon
"""
import itertools
import os

import hickle as hkl
import numpy as np
import nibabel as nib

from imresize import imresize
from utils import read_fcsv, process_testerarr


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train_model(
    nii_filename, fcsv_filename, afid_idx, train_level, model_params
):
    # Loading image
    niimeta = nib.load(nii_filename)
    hdr = niimeta.header
    img_source = np.transpose(niimeta.get_fdata(), (2, 0, 1))

    finalpredarr = []

    # Load and process .fcsv file.
    arr = read_fcsv(fcsv_filename, hdr)
    print(f"Read fcsv. Shape: {arr.shape}")
    arr = arr[int(afid_idx) - 1:int(afid_idx), ...]

    img = np.single(img_source)
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    inner, outer, patch, skip = setup_by_level(train_level, arr, img)

    if skip:
        return {}

    J = patch.cumsum(0).cumsum(1).cumsum(2)

    inner = np.array(inner)
    outer = np.array(outer)

    full = np.unique(np.concatenate((inner, outer)), axis=0)

    # Loads offset file that specifies where to extract features.
    file = np.load(model_params["feature_offsets"])
    smin = file["arr_0"]
    smax = file["arr_1"]

    perm = [2, 0, 1]
    full = full[:, perm]
    smin = smin[:, perm]
    smax = smax[:, perm]

    diff = process_testerarr(full, smin, smax, J, True)
    dist = full - 60
    p = np.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2 + dist[:, 2] ** 2)

    finalpred = []
    for index in range(p.shape[0]):
        finalpred.append(np.hstack((diff[index], p[index])))

    # Concatenate to array of feature vectors.
    finalpredarr.append(np.asarray(finalpred, dtype=np.float32))

    return {
        "name": os.path.basename(nii_filename).split("_")[0],
        "data_arr": np.asarray(finalpredarr, dtype=np.float32),
    }


def setup_by_level(train_level, arr, img):
    skip = False
    if train_level == "fine":
        arr = np.rint(arr).astype(int)[:, [2, 0, 1]]
        patch = imresize(
            img[
                arr[0, 0] - 30 : arr[0, 0] + 31,
                arr[0, 1] - 30 : arr[0, 1] + 31,
                arr[0, 2] - 30 : arr[0, 2] + 31,
            ],
            2,
        )

        if arr[0, 0] < 30 or arr[0, 1] < 30 or arr[0, 2] < 30:
            print("skip")
            skip = True

        x_vals = [60, 60, 60]

    elif train_level == "medium":
        # Image at normal resolution.
        arr = (np.rint(arr) + 50).astype(int)[:, [2, 0, 1]]
        patch = np.pad(imresize(img, 1), 50, mode="constant")[
            arr[0, 0] - 60 : arr[0, 0] + 61,
            arr[0, 1] - 60 : arr[0, 1] + 61,
            arr[0, 2] - 60 : arr[0, 2] + 61,
        ]
        patch = (patch - np.amin(patch)) / (np.amax(patch) - np.amin(patch))
        x_vals = [60, 60, 60]

    elif train_level == "coarse":
        # Downsampled image.
        patch = np.pad(imresize(img, 0.25), 50, mode="constant")
        arr = (np.rint(arr / 4) + 50).astype(int)
        x_vals = [arr[0][0], arr[0][1], arr[0][2]]

    inner_ranges = [range(x - 5, x + 6) for x in x_vals]
    outer_ranges = [range(x - 10, x + 11, 2) for x in x_vals]

    inner = list(itertools.product(*inner_ranges))
    outer = list(itertools.product(*outer_ranges))
    return inner, outer, patch, skip


def train_all(args):
    space = (
        "space-"
        + os.path.basename(args.output_dir).split("space-")[0].split("_")[0]
    )
    finalpredarr_all = [
        train_model(
            nii, fcsv, args.afid_idx, args.train_level, args.model_params
        )
        for nii, fcsv in zip(args.nii_files, args.fcsv_files)
    ]
    finalpredarr_all = [
        finalpredarr
        for finalpredarr in finalpredarr_all
        if len(finalpredarr) > 0
    ]

    # Save to file
    data = {
        "name": "_".join([args.afid_idx, space, args.train_level]),
        "data_arr": finalpredarr_all,
    }

    # Dump data to file
    with open(args.output_dir, "wb") as out_dir:
        hkl.dump(data, out_dir)


if __name__ == "__main__":
    train_all(
        Namespace(
            output_dir=snakemake.output[0],
            afid_idx=snakemake.params[0],
            model_params=snakemake.params[1],
            nii_files=snakemake.input["nii_files"],
            fcsv_files=snakemake.input["fcsv_files"],
            train_level=snakemake.params[2],
        )
    )
