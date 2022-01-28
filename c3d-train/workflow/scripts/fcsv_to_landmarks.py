import csv

import numpy as np
import pandas as pd

fcsv_input = snakemake.input["fcsv_new"]
txt_output = snakemake.output["afids_txt"]
afid_num = int(snakemake.wildcards["afid_num"])

fcsv_df = pd.read_csv(fcsv_input, sep=",", header=2)

coords = fcsv_df.loc[afid_num, ['x','y','z']].to_numpy()
with open (landmarks_txt, 'w') as fid:
    fid.write(' '.join(str(i) for i in coords) + " 1")
