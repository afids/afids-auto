import pandas as pd


def write_coords(fcsv_input, txt_output):
    fcsv_df = pd.read_csv(fcsv_input, sep=",", header=2)

    with open(txt_output, 'w') as fid:
        for afid_num in range(1, 33):
            coords = fcsv_df.loc[afid_num - 1, ['x','y','z']].to_numpy()
            fid.write(' '.join(str(i) for i in coords) + f" {afid_num}\n")

if __name__ == '__main__':
    write_coords(
        fcsv_input=snakemake.input['fcsv_new'],
        txt_output=snakemake.output['landmarks']
        )
