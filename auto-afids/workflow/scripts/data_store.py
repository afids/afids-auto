#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:26:50 2020

@author: greydon
"""
import csv
import itertools
import os

import hickle as hkl
import numpy as np
import nibabel as nib

from imresize import imresize


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
    with open(fcsv_filename, encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for _ in range(3):
            next(csv_reader)

        arr = np.empty((0, 3))
        for row in csv_reader:
            arr = np.vstack([arr, np.asarray(row[1:4], dtype="float64")])

    arr = load_arr(hdr, afid_idx, arr)

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

    mincornerlist = np.zeros((4000 * full.shape[0], 3)).astype("uint8")
    maxcornerlist = np.zeros((4000 * full.shape[0], 3)).astype("uint8")

    for index in range(full.shape[0]):
        mincorner = full[index] + smin
        maxcorner = full[index] + smax
        mincornerlist[index * 4000 : (index + 1) * 4000] = mincorner
        maxcornerlist[index * 4000 : (index + 1) * 4000] = maxcorner

    cornerlist = np.hstack((mincornerlist, maxcornerlist)).astype(int)

    Jnew = np.zeros((J.shape[0] + 1, J.shape[1] + 1, J.shape[2] + 1))
    Jnew[1:, 1:, 1:] = J

    # Generation of features (random blocks of intensity around fiducial)
    testerarr = (
        Jnew[cornerlist[:, 3] + 1, cornerlist[:, 4] + 1, cornerlist[:, 5] + 1]
        - Jnew[cornerlist[:, 3] + 1, cornerlist[:, 4] + 1, cornerlist[:, 2]]
        - Jnew[cornerlist[:, 3] + 1, cornerlist[:, 1], cornerlist[:, 5] + 1]
        - Jnew[cornerlist[:, 0], cornerlist[:, 4] + 1, cornerlist[:, 5] + 1]
        + Jnew[cornerlist[:, 0], cornerlist[:, 1], cornerlist[:, 5] + 1]
        + Jnew[cornerlist[:, 0], cornerlist[:, 4] + 1, cornerlist[:, 2]]
        + Jnew[cornerlist[:, 3] + 1, cornerlist[:, 1], cornerlist[:, 2]]
        - Jnew[cornerlist[:, 0], cornerlist[:, 1], cornerlist[:, 2]]
    ) / (
        (cornerlist[:, 3] - cornerlist[:, 0] + 1)
        * (cornerlist[:, 4] - cornerlist[:, 1] + 1)
        * (cornerlist[:, 5] - cornerlist[:, 2] + 1)
    )

    vector1arr = np.zeros((4000 * full.shape[0]))
    vector2arr = np.zeros((4000 * full.shape[0]))

    for index in range(full.shape[0]):
        vector = range(index * 4000, index * 4000 + 2000)
        vector1arr[index * 4000 : (index + 1) * 4000 - 2000] = vector

    for index in range(full.shape[0]):
        vector = range(index * 4000 + 2000, index * 4000 + 4000)
        vector2arr[index * 4000 + 2000 : (index + 1) * 4000] = vector

    vector1arr[0] = 1
    vector1arr = vector1arr[vector1arr != 0]
    vector1arr[0] = 0
    vector2arr = vector2arr[vector2arr != 0]
    vector1arr = vector1arr.astype(int)
    vector2arr = vector2arr.astype(int)

    diff = testerarr[vector1arr] - testerarr[vector2arr]
    diff = np.reshape(diff, (full.shape[0], 2000))
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


def load_arr(hdr, afid_idx, arr):
    if hdr["qform_code"] > 0 and hdr["sform_code"] == 0:
        newarr = []
        B = hdr["quatern_b"]
        C = hdr["quatern_c"]
        D = hdr["quatern_d"]
        A = np.sqrt(1 - B ** 2 - C ** 2 - D ** 2)

        R = (
            [
                A ** 2 + B ** 2 - C ** 2 - D ** 2,
                2 * (B * C - A * D),
                2 * (B * D + A * C),
            ],
            [
                2 * (B * C + A * D),
                A ** 2 + C ** 2 - B ** 2 - D ** 2,
                2 * (C * D + A * B),
            ],
            [
                2 * (B * D - A * C),
                2 * (C * D + A * B),
                A ** 2 + D ** 2 - C ** 2 - B ** 2,
            ],
        )
        R = np.array(R)

        ijk = arr[int(afid_idx) - 1].reshape(-1, 1)
        ijk[2] = ijk[2] * hdr["pixdim"][0]
        pixdim = hdr["pixdim"][1], hdr["pixdim"][2], hdr["pixdim"][3]
        pixdim = np.array(pixdim).reshape(-1, 1)
        fill = np.matmul(R, ijk) * pixdim + np.vstack(
            [hdr["qoffset_x"], hdr["qoffset_y"], hdr["qoffset_z"]]
        )
        fill = fill.reshape(3)
        newarr.append(fill)

        arr = np.array(newarr)

        arr = arr - 1

    elif hdr["sform_code"] > 0:

        newarr = []
        four = np.vstack(
            [hdr["srow_x"], hdr["srow_y"], hdr["srow_z"], [0, 0, 0, 1]]
        )
        four = np.linalg.inv(four)
        trying = np.hstack([arr, np.ones((32, 1))])
        fill = np.matmul(four, trying[int(afid_idx) - 1].reshape(-1, 1))
        fill = fill.reshape(4)
        newarr.append(fill)

        arr = np.array(newarr)
        arr = arr - 1

    else:
        print("Error in sform_code or qform_code, cannot obtain coordinates.")

    return arr


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
