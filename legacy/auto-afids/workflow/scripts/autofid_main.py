# -*- coding: utf-8 -*-
"""
Created on Fri May 22 02:18:14 2020
@author: danie
"""

import itertools
import time

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_img
from scipy.ndimage import zoom

# from imresize import imresize
from utils import process_testerarr, read_fcsv


def imresize(arr, size):
    return zoom(arr, (size for shape in arr.shape))


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
    niimeta = nib.load(file_img)
    niimeta = resample_img(niimeta, target_affine=np.eye(3))
    hdr = niimeta.header
    img = niimeta.get_fdata().copy()
    # (x, y, z) -> (z, x, y)
    # Not sure why we do this
    img = np.transpose(img, (2, 0, 1))

    # Cast array data to single (Not sure why we do this)
    img = np.single(img)
    # Normalize img on scale from 0-1
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    # Coarse resolution level of the image (downsampled by a factor of 4).
    img_coarse = imresize(img, 0.25)

    # (Zero-)Pad the image by 50 (Add 50 zeroes on both sides of each axis)
    img_coarse_pad = np.pad(img_coarse, 50, mode="constant")
    print(f"img_coarse_pad size: {img_coarse_pad.shape}")

    # Load an 'initial features' file
    # Will speed up processing.
    print("Loading initial features...")
    diff_coarse = np.load(initial_features)
    # full is a list of coordinates to check from the model(?)
    test_voxels_coarse_full = np.unique(
        np.asarray(
            list(
                itertools.product(
                    *[
                        range(50, img_coarse_pad.shape[1] - 50, 2),
                        range(50, img_coarse_pad.shape[2] - 50, 2),
                        range(50, img_coarse_pad.shape[0] - 50, 2),
                    ]
                )
            )
        ),
        axis=0,
    )

    perm = [2, 0, 1]
    test_voxels_coarse_full = test_voxels_coarse_full[:, perm]
    print(f"Test voxels coarse full size: {test_voxels_coarse_full.shape}")
    img_coarse_pad_cumsum = img_coarse_pad.cumsum(0).cumsum(1).cumsum(2)
    print(f"img_coarse_pad_cumsum size: {img_coarse_pad_cumsum.shape}")
    print("Starting autofid...")

    # Load offsets file.
    file = np.load(file_feature_offsets)
    smin = file["arr_0"]
    smax = file["arr_1"]
    smin = smin[:, perm]
    smax = smax[:, perm]

    start = time.time()

    # Load regression forest trained on downsampled resolution and test the
    # current image.
    with open(coarse_model, "rb") as f:
        model = joblib.load(f)
    idx = pd.DataFrame(model.predict(diff_coarse))[0].idxmin()
    print(f"first prediction idx: {idx}")
    diff, test_voxels_coarse_patch = check_prediction_cumsum(
        idx, test_voxels_coarse_full, img_coarse_pad_cumsum, smin, smax
    )
    print(f"test_voxels_coarse_patch size: {test_voxels_coarse_patch.shape}")

    idx = pd.DataFrame(model.predict(diff))[0].idxmin()
    # Problem line
    print(f"Full img size: {img.shape}")
    (
        diff,
        test_voxels_med_patch,
        patch_med_pad_cumsum,
        testing,
    ) = check_prediction_coarse(idx, test_voxels_coarse_patch, img, smin, smax)
    print(f"first test_voxels_med_patch size: {test_voxels_med_patch.shape}")

    with open(med_model, "rb") as f:
        model = joblib.load(f)
    idx = pd.DataFrame(model.predict(diff))[0].idxmin()
    diff, test_voxels_med_patch = check_prediction_cumsum(
        idx, test_voxels_med_patch, patch_med_pad_cumsum, smin, smax
    )

    idx = pd.DataFrame(model.predict(diff))[0].idxmin()
    # Problem line
    print(f"second test_voxels_med_patch size: {test_voxels_med_patch.shape}")
    diff, test_voxels_fine_patch = check_prediction_med(
        idx, test_voxels_med_patch, img, smin, smax, testing
    )

    with open(fine_model, "rb") as f:
        model = joblib.load(f)
    idx = pd.DataFrame(model.predict(diff))[0].idxmin()
    testing = (
        testing - 30 + (test_voxels_fine_patch[idx] * (30 / 60.5) + (0.5 * (30 / 60.5)))
    )
    testingarr = np.vstack([np.empty((0, 3)), testing])

    if file_fcsv is not None:
        arr = read_fcsv(file_fcsv, hdr)
        dist = np.sqrt(
            (arr[fiducial_number - 1][0] - testing[1]) ** 2
            + (arr[fiducial_number - 1][1] - testing[2]) ** 2
            + (arr[fiducial_number - 1][2] - testing[0]) ** 2
        )

        distarr = np.vstack([np.empty((0, 1)), dist])
        print("AFLE = " + str(distarr[:]))
    else:
        print("No ground truth fiducial file specified. Continuing with execution.")

    with open(out_file, "w", encoding="utf-8") as out:
        out.write(f"{str(testingarr[:])}\n")

    end = time.time()
    elapsed = end - start
    print(f"Time to locate fiducial = {str(elapsed)}")


def check_prediction_cumsum(idx, test_voxels, img_cumsum, smin, smax):
    # Define a 7*7*7 patch around the minimum voxel
    iterables = [
        range(test_voxels[idx][0] - 3, test_voxels[idx][0] + 4),
        range(test_voxels[idx][1] - 3, test_voxels[idx][1] + 4),
        range(test_voxels[idx][2] - 3, test_voxels[idx][2] + 4),
    ]
    test_voxels_new = np.unique(np.asarray(list(itertools.product(*iterables))), axis=0)

    diff = process_testerarr(test_voxels_new, smin, smax, img_cumsum, False)

    return diff, test_voxels_new


def check_prediction_coarse(idx, test_voxels, img, smin, smax):
    # Remove padding from minimum idx and bring back to med resolution
    testing = (test_voxels[idx] - 50) * 4
    # Pad nominal image
    img_pad = np.pad(img, 50, mode="constant")
    # Add padding to minimum index at med resolution
    testing = testing + 50
    # Define a 121*121*121 patch around testing
    patch = img_pad[
        testing[0] - 60 : testing[0] + 61,
        testing[1] - 60 : testing[1] + 61,
        testing[2] - 60 : testing[2] + 61,
    ]
    print(f"First patch size: {patch.shape}")
    # Normalize patch to 0-1
    print(
        f"check_prediction_coarse patch -- amin: {np.amin(patch)}, amax: {np.amax(patch)}"
    )
    patch = (patch - np.amin(patch)) / (np.amax(patch) - np.amin(patch))
    patch_cumsum = patch.cumsum(0).cumsum(1).cumsum(2)
    # Define an 8*8*8 patch around testing in the patch, using every other
    # voxel.
    iterables = [
        range(60 - 7, 60 + 8, 2),
        range(60 - 7, 60 + 8, 2),
        range(60 - 7, 60 + 8, 2),
    ]

    test_voxels_new = np.unique(
        np.asarray(list(itertools.product(*iterables))), axis=0
    )[:, [2, 0, 1]]

    diff = process_testerarr(test_voxels_new, smin, smax, patch_cumsum, False)

    return diff, test_voxels_new, patch_cumsum, testing


def check_prediction_med(idx, test_voxels, img, smin, smax, testing):
    print(f"first testing: {testing}")
    # Adjust minimum test_voxels using testing and a fixed bias
    # -60 to remove patch(?), -50 to ignore padding
    print(f"med idxmin test voxel: {test_voxels[idx]}")
    testing = np.rint((test_voxels[idx] - 60) + (testing - 50)).astype("int")
    print(f"second testing (testing is not None): {testing}")
    print(f"first img size (testing is not None): {img.shape}")
    # if any element of testing is less than 30 this breaks
    # Define a 61*61*61 patch around testing and move to fine resolution
    patch = imresize(
        img[
            testing[0] - 30 : testing[0] + 31,
            testing[1] - 30 : testing[1] + 31,
            testing[2] - 30 : testing[2] + 31,
        ],
        2,
    )
    print(f"patch size (testing is not None): {patch.shape}")
    patch_cumsum = patch.cumsum(0).cumsum(1).cumsum(2)
    # Define a set of voxels in a 15*15*15 cube in the middle of the patch
    iterables = [range(60 - 7, 60 + 8), range(60 + 7, 60 + 8), range(60 + 7, 60 + 8)]

    test_voxels_new = np.unique(
        np.asarray(list(itertools.product(*iterables))), axis=0
    )[:, [2, 0, 1]]

    diff = process_testerarr(test_voxels_new, smin, smax, patch_cumsum, True)

    return diff, test_voxels_new


if __name__ == "__main__":
    main(
        snakemake.input["nii_file"],
        None,
        snakemake.input["initial_features"],
        snakemake.params["afid_num"],
        snakemake.params["feature_offsets"],
        snakemake.input["models"][0],
        snakemake.input["models"][1],
        snakemake.input["models"][2],
        snakemake.output[0],
    )
