import csv
import re

import numpy as np
import pandas as pd

fcsv_input = snakemake.input["fcsv_new"]
txt_output = snakemake.output["afids_txt"]

fcsv_df = pd.read_csv(fcsv_input, sep=",", header=2)
afid_pattern = r"desc-afid(\d{2})"

for output_file in txt_output:
    afid_num = int(re.search(afid_pattern, output_file).group(1))
    coords = fcsv_df.loc[afid_num - 1, ['x','y','z']].to_numpy()
    with open (output_file, 'w') as fid:
        fid.write(' '.join(str(i) for i in coords) + " 1")
