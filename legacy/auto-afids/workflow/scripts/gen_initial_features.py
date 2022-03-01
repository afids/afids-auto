import itertools

import nibabel as nib
import numpy as np
from imresize import imresize
from nilearn.image import resample_img
from utils import process_testerarr


def gen_initial_features(path_img, path_feature_offsets, path_out):
    """Generate an 'initial features' file.

    This works by scanning through the image and extracting intensity
    information.
    """
    niimeta = nib.load(path_img)
    # Does this do anything?
    niimeta = resample_img(niimeta, target_affine=np.eye(3))
    img = niimeta.get_fdata()
    # (x, y, z) -> (z, x, y)
    img = np.transpose(img, (2, 0, 1))
    # Downsample image to "coarse" size (by a factor of 4), then zero-pad by
    # 50 on every axis.
    img_pad = np.pad(imresize(img, 0.25), 50, mode="constant")
    # Generate an array of every other unique voxel within the main image (i.e.
    # skipping padding)
    voxels_to_test = np.unique(
        np.asarray(
            list(
                itertools.product(
                    *[
                        range(50, img_pad.shape[1] - 50, 2),
                        range(50, img_pad.shape[2] - 50, 2),
                        range(50, img_pad.shape[0] - 50, 2),
                    ]
                )
            )
        ),
        axis=0,
    )
    # Take the cumulative sum of the image along all three axes.
    img_cumsum = img_pad.cumsum(0).cumsum(1).cumsum(2)

    file = np.load(path_feature_offsets)
    smin = file["arr_0"]
    smax = file["arr_1"]
    perm = [2, 0, 1]
    # (z, x, y) -> (y, z, x)
    voxels_to_test = voxels_to_test[:, perm]
    smin = smin[:, perm]
    smax = smax[:, perm]

    # Produce an n * 2000 array to process further.
    diff_coarse = process_testerarr(voxels_to_test, smin, smax, img_cumsum, False)

    np.save(path_out, diff_coarse)


if __name__ == "__main__":
    gen_initial_features(
        snakemake.input["nii_file"],
        snakemake.params["feature_offsets"],
        snakemake.output[0],
    )
