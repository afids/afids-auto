import itertools

import numpy as np
import nibabel as nib
from nilearn.image import resample_img

from imresize import imresize
from utils import process_testerarr


def gen_initial_features(path_img, path_feature_offsets, path_out):
    """Generate an 'initial features' file.

    This works by scanning through the image and extracting intensity
    information.
    """
    niimeta = nib.load(path_img)
    niimeta = resample_img(niimeta, target_affine=np.eye(3))
    img = niimeta.get_fdata()
    img = np.transpose(img, (2, 0, 1))
    img_pad = np.pad(imresize(img, 0.25), 50, mode="constant")
    iterables = [
        range(50, img_pad.shape[1] - 50, 2),
        range(50, img_pad.shape[2] - 50, 2),
        range(50, img_pad.shape[0] - 50, 2),
    ]
    full = list(itertools.product(*iterables))
    full = np.asarray(full)
    full = np.unique(full, axis=0)
    Jstored = img_pad.cumsum(0).cumsum(1).cumsum(2)

    file = np.load(path_feature_offsets)
    smin = file["arr_0"]
    smax = file["arr_1"]
    perm = [2, 0, 1]
    full = full[:, perm]
    smin = smin[:, perm]
    smax = smax[:, perm]

    diff_coarse = process_testerarr(full, smin, smax, Jstored, False)

    np.save(path_out, diff_coarse)


if __name__ == "__main__":
    gen_initial_features(
        snakemake.input["nii_file"],
        snakemake.input["feature_offsets"],
        snakemake.output[0],
    )
