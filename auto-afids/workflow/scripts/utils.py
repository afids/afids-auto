import csv

import numpy as np


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
