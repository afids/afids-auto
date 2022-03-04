import csv
import re

import numpy as np
import pandas as pd

fcsv_input = snakemake.input["fcsv_new"]
txt_output = snakemake.output["landmarks"]

fcsv_df = pd.read_csv(fcsv_input, sep=",", header=2)

with open(txt_output, 'w') as fid:
    for afid_num in range(1, 33):
        coords = fcsv_df.loc[afid_num - 1, ['x','y','z']].to_numpy()
        fid.write(' '.join(str(i) for i in coords) + f" {afid_num}\n")
