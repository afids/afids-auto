#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np

fcsv_source = snakemake.input["groundtruth"]
xfm_txt = snakemake.input["xfm_new"]
template = snakemake.params["template"]
fcsv_new = snakemake.output["fcsv_new"]

# load transform from subj to template
sub2template= np.loadtxt(xfm_txt)
fcsv_df = pd.read_table(fcsv_input, sep=",", header=2)

coords = fcsv_df[['x','y','z']].to_numpy()

# to plot in mni space, need to transform coords
tcoords = np.zeros(coords.shape)
for i in range(len(coords)):
    vec = np.hstack([coords[i,:],1])
    tvec = np.linalg.inv(sub2template) @ vec.T
    tcoords[i,:] = tvec[:3]

with open(template, "r", encoding="utf-8") as file:
    list_of_lists = []
    reader = csv.reader(file)
    for i in range(3):
        list_of_lists.append(next(reader))
    for idx, val in enumerate(reader):
        val[1:4] = tcoords[idx][:3]
        list_of_lists.append(val)

with open(fcsv_new, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(list_of_lists)
