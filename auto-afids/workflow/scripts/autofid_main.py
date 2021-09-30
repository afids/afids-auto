# -*- coding: utf-8 -*-
"""
Created on Fri May 22 02:18:14 2020
@author: danie
"""

import itertools
import time

import joblib
from nilearn.image import resample_img
import pandas as pd
import numpy as np
import nibabel as nib

from imresize import imresize
from utils import process_testerarr, read_fcsv


def main(
    file_img,
    file_fcsv,
    initial_features,
    fiducial_number,
    file_feature_offsets,
    coarse_model,
    med_model,
    fine_model,
    out_file,
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

    # Load offsets file.
    file = np.load(file_feature_offsets)
    smin = file["arr_0"]
    smax = file["arr_1"]
    perm = [2, 0, 1]
    smin = smin[:, perm]
    smax = smax[:, perm]

    start = time.time()

    # Load regression forest trained on downsampled resolution and test the
    # current image.
    with open(coarse_model, "rb") as f:
        model = joblib.load(f)

    answer = model.predict(diff_coarse)
    df = pd.DataFrame(answer)
    diff, full2, _, _ = check_prediction(df, full, Jstored, smin, smax, False)

    answer = model.predict(diff)
    df = pd.DataFrame(answer)
    diff, fullmed, Jstored2, testing = check_prediction(
        df, full2, img, smin, smax, True
    )

    with open(med_model, "rb") as f:
        model = joblib.load(f)

    answer = model.predict(diff)
    df = pd.DataFrame(answer)
    diff, fullmed, _, _ = check_prediction(
        df, fullmed, Jstored2, smin, smax, False
    )

    answer = model.predict(diff)
    df = pd.DataFrame(answer)
    diff, fullfine, _, _ = check_prediction(
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

    with open(out_file, "w", encoding="utf-8") as out:
        out.write(f"{str(testingarr[:])}\n")

    end = time.time()
    elapsed = end - start
    print(f"Time to locate fiducial = {str(elapsed)}")


def check_prediction(
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


if __name__ == "__main__":
    main(
        snakemake.input["nii_file"],
        None,
        snakemake.input["initial_features"],
        snakemake.params["afid_num"],
        snakemake.input["feature_offsets"],
        snakemake.input["models"][0],
        snakemake.input["models"][1],
        snakemake.input["models"][2],
        snakemake.output,
    )
