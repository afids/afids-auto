# -*- coding: utf-8 -*-
"""
Created on Fri May 22 02:18:14 2020
@author: danie
"""

import sys
import csv
import itertools
import os
import time

import joblib
from nilearn.image import resample_img
import pandas as pd
import numpy as np
import nibabel as nib

from imresize import imresize


def main(
    file_img,
    file_fcsv,
    initial_features,
    fiducial_number,
    file_feature_offsets,
    coarse_model,
    med_model,
    fine_model,
):
    """Autofid main script

    Use on new mri T1w image to find predicted fiducial location.
    """

    # Loading the image in, as well as some initial variables.
    testingarr = np.empty((0, 3))
    distarr = np.empty((0, 1))

    niimeta = nib.load(file_img)
    niimeta = resample_img(niimeta, target_affine=np.eye(3))
    hdr = niimeta.header
    img = niimeta.get_fdata()
    img = np.transpose(img, (2, 0, 1))

    # Starting at a coarse resolution level of the image (downsampled by a factor of 4).
    img_new = imresize(img, 0.25)

    img_pad = np.pad(img_new, 50, mode="constant")

    if os.path.exists(initial_features):
        # Load an 'initial features' file if it exists.
        # Will speed up processing.
        print("Loading initial features...")
        diff_coarse = np.load(initial_features)
        iterables = [
            range(50, img_pad.shape[1] - 50, 2),
            range(50, img_pad.shape[2] - 50, 2),
            range(50, img_pad.shape[0] - 50, 2),
        ]

        full = list(itertools.product(*iterables))
        full = np.asarray(full)
        full = np.unique(full, axis=0)
        perm = [2, 0, 1]
        full = full[:, perm]
        Jstored = img_pad.cumsum(0).cumsum(1).cumsum(2)
        print("Starting autofid...")
    else:
        # Generate an 'initial features' file by scanning through the
        # image and extracting intensity information.
        print("Creating initial features...")
        iterables = [
            range(50, img_pad.shape[1] - 50, 2),
            range(50, img_pad.shape[2] - 50, 2),
            range(50, img_pad.shape[0] - 50, 2),
        ]
        full = list(itertools.product(*iterables))
        full = np.asarray(full)
        full = np.unique(full, axis=0)
        Jstored = img_pad.cumsum(0).cumsum(1).cumsum(2)

        file = np.load(file_feature_offsets)
        smin = file["arr_0"]
        smax = file["arr_1"]
        perm = [2, 0, 1]
        full = full[:, perm]
        smin = smin[:, perm]
        smax = smax[:, perm]

        diff_coarse = process_testerarr(
            full, smin, smax, Jstored, False
        )

        np.save(initial_features, diff_coarse)

    # Load offsets file.
    file = np.load(file_feature_offsets)
    smin = file["arr_0"]
    smax = file["arr_1"]
    perm = [2, 0, 1]
    smin = smin[:, perm]
    smax = smax[:, perm]
    os.chdir("/project/6050199/dcao6/autofid/models/new/")

    start = time.time()

    # Load regression forest trained on downsampled resolution and test the
    # current image.
    with open(coarse_model, "rb") as f:
        model = joblib.load(f)

    answer = model.predict(diff_coarse)
    df = pd.DataFrame(answer)
    diff, full2, _, _ = check_prediction_generic(
        df, full, Jstored, smin, smax, False
    )

    answer = model.predict(diff)
    df = pd.DataFrame(answer)
    diff, fullmed, Jstored2, testing = check_prediction_generic(
        df, full2, img, smin, smax, True
    )

    with open(med_model, "rb") as f:
        model = joblib.load(f)

    answer = model.predict(diff)
    df = pd.DataFrame(answer)
    diff, fullmed, _, _ = check_prediction_generic(
        df, fullmed, Jstored2, smin, smax, False
    )

    answer = model.predict(diff)
    df = pd.DataFrame(answer)
    diff, fullfine, _, _ = check_prediction_generic(
        df, fullmed, img, smin, smax, True, testing=testing
    )

    with open(fine_model, "rb") as f:
        model = joblib.load(f)

    answer = model.predict(diff)
    df = pd.DataFrame(answer)
    idx = df[0].idxmin()
    testing = (
        testing - 30 + (fullfine[idx] * (30 / 60.5) + (0.5 * (30 / 60.5)))
    )
    testingarr = np.vstack([testingarr, testing])

    if file_fcsv is not None:
        arr = read_fcsv(file_fcsv, hdr)
        dist = np.sqrt(
            (arr[fiducial_number - 1][0] - testing[1]) ** 2
            + (arr[fiducial_number - 1][1] - testing[2]) ** 2
            + (arr[fiducial_number - 1][2] - testing[0]) ** 2
        )
        distarr = np.vstack([distarr, dist])
        print("AFLE = " + str(distarr[:]))
    else:
        print(
            "No ground truth fiducial file specified. Continuing with execution."
        )

    print(f"Fid = {fiducial_number}")
    print(f"Fid coordinates = {str(testingarr[:])}")

    end = time.time()
    elapsed = end - start
    print(f"Time to locate fiducial = {str(elapsed)}")


def read_fcsv(fcsv_path, hdr):
    # Obtain fiducial coordinates given a Slicer .fcsv file.
    with open(fcsv_path, encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)
        next(csv_reader)
        next(csv_reader)
        arr = np.empty((0, 3))
        for row in csv_reader:
            x = row[1:4]
            x = np.asarray(x, dtype="float64")
            arr = np.vstack([arr, x])

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

        for i in range(32):
            ijk = arr[i].reshape(-1, 1)
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

        print(arr)

    elif hdr["sform_code"] > 0:

        newarr = []
        four = np.vstack(
            [hdr["srow_x"], hdr["srow_y"], hdr["srow_z"], [0, 0, 0, 1]]
        )
        four = np.linalg.inv(four)
        trying = np.hstack([arr, np.ones((32, 1))])
        for i in range(32):
            fill = np.matmul(four, trying[i].reshape(-1, 1))
            fill = fill.reshape(4)
            newarr.append(fill)

        arr = np.array(newarr)

        arr = arr - 1
        print(arr)

    else:
        print("Error in sform_code or qform_code, cannot obtain coordinates.")
    return arr


def check_prediction_generic(
    df, full, Jstored, smin, smax, resize_Jstored, testing=None
):
    idx = df[0].idxmin()
    int_cornerlist = False

    if resize_Jstored:
        if testing is not None:
            int_cornerlist = True
            testing = np.rint(full[idx] - 60 + testing - 50).astype("int")
            patch = imresize(
                Jstored[
                    testing[0] - 30 : testing[0] + 31,
                    testing[1] - 30 : testing[1] + 31,
                    testing[2] - 30 : testing[2] + 31,
                ],
                2,
            )
            Jstored = patch.cumsum(0).cumsum(1).cumsum(2)
            iterables = [
                range(60 - 7, 60 + 8),
                range(60 + 7, 60 + 8),
                range(60 + 7, 60 + 8),
            ]
        else:
            testing = (full[idx] - 50) * 4
            img_pad = np.pad(Jstored, 50, mode="constant")
            testing = testing + 50
            patch = img_pad[
                testing[0] - 60 : testing[0] + 61,
                testing[1] - 60 : testing[1] + 61,
                testing[2] - 60 : testing[2] + 61,
            ]
            patch = (patch - np.amin(patch)) / (
                np.amax(patch) - np.amin(patch)
            )
            Jstored = patch.cumsum(0).cumsum(1).cumsum(2)
            iterables = [
                range(60 - 7, 60 + 8, 2),
                range(60 - 7, 60 + 8, 2),
                range(60 - 7, 60 + 8, 2),
            ]
    else:
        testing = None
        iterables = [
            range(full[idx][0] - 3, full[idx][0] + 4),
            range(full[idx][1] - 3, full[idx][1] + 4),
            range(full[idx][2] - 3, full[idx][2] + 4),
        ]

    full2 = np.unique(np.asarray(list(itertools.product(*iterables))), axis=0)

    if resize_Jstored:
        full2 = full2[:, [2, 0, 1]]

    diff = process_testerarr(full2, smin, smax, Jstored, int_cornerlist)

    return diff, full2, Jstored, testing


def process_testerarr(full, smin, smax, Jstored, int_cornerlist):
    mincornerlist = np.zeros((4000 * full.shape[0], 3)).astype("uint8")
    maxcornerlist = np.zeros((4000 * full.shape[0], 3)).astype("uint8")

    for index in range(full.shape[0]):
        mincorner = full[index] + smin
        maxcorner = full[index] + smax
        mincornerlist[index * 4000 : (index + 1) * 4000] = mincorner
        maxcornerlist[index * 4000 : (index + 1) * 4000] = maxcorner

    cornerlist = np.hstack((mincornerlist, maxcornerlist))
    if int_cornerlist:
        cornerlist = cornerlist.astype(int)

    Jnew = np.zeros(
        (Jstored.shape[0] + 1, Jstored.shape[1] + 1, Jstored.shape[2] + 1)
    )
    Jnew[1:, 1:, 1:] = Jstored
    Jcoarse = Jnew

    testerarr = (
        Jcoarse[
            cornerlist[:, 3] + 1,
            cornerlist[:, 4] + 1,
            cornerlist[:, 5] + 1,
        ]
        - Jcoarse[cornerlist[:, 0], cornerlist[:, 4] + 1, cornerlist[:, 5] + 1]
        - Jcoarse[cornerlist[:, 3] + 1, cornerlist[:, 4] + 1, cornerlist[:, 2]]
        - Jcoarse[cornerlist[:, 3] + 1, cornerlist[:, 1], cornerlist[:, 5] + 1]
        + Jcoarse[cornerlist[:, 3] + 1, cornerlist[:, 1], cornerlist[:, 2]]
        + Jcoarse[cornerlist[:, 0], cornerlist[:, 1], cornerlist[:, 5] + 1]
        + Jcoarse[cornerlist[:, 0], cornerlist[:, 4] + 1, cornerlist[:, 2]]
        - Jcoarse[cornerlist[:, 0], cornerlist[:, 1], cornerlist[:, 2]]
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

    return diff
