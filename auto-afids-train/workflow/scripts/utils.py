import csv

import numpy as np


def process_testerarr(voxels_to_test, smin, smax, img_cumsum, int_cornerlist):
    """Compute differences between regions of an image.

    Parameters
    ----------
    voxels_to_test : array_like
        An n * 3 array, where each 3-vector is the index of a voxel in
        img_cumsum.
    smin : int
        A number that gets added to every 3-vector in full.
    smax : int
        A number that gets added to every 3-vector in float.
    img_cumsum : array_like
        A 3D array that's related to the image under investigation.
    int_cornerlist : bool
        cornerlist gets converted to int if this is True.
    """
    # Initialize corner lists with 4000 times the number of voxels provided
    mincornerlist = np.zeros((4000 * voxels_to_test.shape[0], 3)).astype(
        "uint8"
    )
    maxcornerlist = np.zeros((4000 * voxels_to_test.shape[0], 3)).astype(
        "uint8"
    )

    # Fill the corner lists with each voxel plus smin/smax
    for index in range(voxels_to_test.shape[0]):
        mincornerlist[index * 4000 : (index + 1) * 4000] = (
            voxels_to_test[index] + smin
        )
        maxcornerlist[index * 4000 : (index + 1) * 4000] = (
            voxels_to_test[index] + smax
        )

    # Put the corner lists side-by-side, resulting in an (n * 4000) * 6 array
    # Each row of this defines a cubic patch of img_cumsum to look at
    cornerlist = np.hstack((mincornerlist, maxcornerlist))
    # Convert cornerlist to int from "uint8"
    if int_cornerlist:
        cornerlist = cornerlist.astype(int)

    # Construct an array that's one bigger than Jstored along all axes, with
    # the extra space filled by zeros
    img_cumsum_expanded = np.zeros(
        (
            img_cumsum.shape[0] + 1,
            img_cumsum.shape[1] + 1,
            img_cumsum.shape[2] + 1,
        )
    )
    img_cumsum_expanded[1:, 1:, 1:] = img_cumsum

    # Construct a tester array from various parts of Jcoarse.
    # Each cornerlist[:, idx] is an array of length (n * 4000) with vals from
    # one column of cornerlist. So, The first element here is all the values at
    # the maximum corners defined by the corner list.
    # This ends up being an array of length (n * 4000)
    testerarr = (
        img_cumsum_expanded[
            cornerlist[:, 3] + 1,
            cornerlist[:, 4] + 1,
            cornerlist[:, 5] + 1,
        ]
        - img_cumsum_expanded[
            cornerlist[:, 0], cornerlist[:, 4] + 1, cornerlist[:, 5] + 1
        ]
        - img_cumsum_expanded[
            cornerlist[:, 3] + 1, cornerlist[:, 4] + 1, cornerlist[:, 2]
        ]
        - img_cumsum_expanded[
            cornerlist[:, 3] + 1, cornerlist[:, 1], cornerlist[:, 5] + 1
        ]
        + img_cumsum_expanded[
            cornerlist[:, 3] + 1, cornerlist[:, 1], cornerlist[:, 2]
        ]
        + img_cumsum_expanded[
            cornerlist[:, 0], cornerlist[:, 1], cornerlist[:, 5] + 1
        ]
        + img_cumsum_expanded[
            cornerlist[:, 0], cornerlist[:, 4] + 1, cornerlist[:, 2]
        ]
        - img_cumsum_expanded[
            cornerlist[:, 0], cornerlist[:, 1], cornerlist[:, 2]
        ]
    ) / (
        (cornerlist[:, 3] - cornerlist[:, 0] + 1)
        * (cornerlist[:, 4] - cornerlist[:, 1] + 1)
        * (cornerlist[:, 5] - cornerlist[:, 2] + 1)
    )

    vector1arr = np.zeros((4000 * voxels_to_test.shape[0]))
    vector2arr = np.zeros((4000 * voxels_to_test.shape[0]))

    # Set up two arrays of indices, each with values equal to their indices or
    # zero. These will each be length 4000 arrays, with half their values being
    # zero.
    # e.g. vector1arr[0:2000] = 0 - 1999, vector2arr[2000:4000] = 2000:3999
    # e.g. vector2arr[0:2000] = 0       , vector1arr[2000:4000] = 0
    for index in range(voxels_to_test.shape[0]):
        vector1arr[index * 4000 : (index + 1) * 4000 - 2000] = range(
            index * 4000, index * 4000 + 2000
        )
        vector2arr[index * 4000 + 2000 : (index + 1) * 4000] = range(
            index * 4000 + 2000, index * 4000 + 4000
        )

    # Keep only the nonzero elements of each vector array
    vector1arr[0] = 1
    vector1arr = vector1arr[vector1arr != 0].astype(int)
    vector1arr[0] = 0
    vector2arr = vector2arr[vector2arr != 0].astype(int)

    # Subtract the elements in vector1arr from the elements in vector2arr
    # So we're subtracting every other 2000 elements starting at index 2000
    # from every other 2000 elements starting at index 0
    diff = testerarr[vector1arr] - testerarr[vector2arr]
    # And we end up with an n * 2000 array
    diff = np.reshape(diff, (voxels_to_test.shape[0], 2000))

    return diff


def read_fcsv(fcsv_path, hdr):
    """Obtain fiducial coordinates given a Slicer .fcsv file."""
    with open(fcsv_path, encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for _ in range(3):
            next(csv_reader)

        arr = np.empty((0, 3))
        for row in csv_reader:
            arr = np.vstack([arr, np.asarray(row[1:4], dtype="float64")])

    if hdr["qform_code"] > 0 and hdr["sform_code"] == 0:
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

        newarr = []
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

    else:
        print("Error in sform_code or qform_code, cannot obtain coordinates.")
    return arr
