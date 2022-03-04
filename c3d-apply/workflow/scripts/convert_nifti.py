import nibabel as nb
import numpy as np

img = nb.load(snakemake.input.ct)
data = np.round(img.get_fdata()).astype(np.float64)
data = nb.Nifti1Image(data, header=img.header, affine=img.affine)
data.header.set_data_dtype(np.float64)
nb.save(data, str(snakemake.output.ct_out))
