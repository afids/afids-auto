#!/usr/bin/python

import tempfile
import shutil
import pathlib
import subprocess

image = snakemake.input["image"]

resampled = snakemake.output["resampled"]
warped = snakemake.output["warped"]
xfm = snakemake.output["xfm"]

fixed = snakemake.params["fixed"]
dof = str(snakemake.params["dof"])
coarse = str(snakemake.params["coarse"])
fine = str(snakemake.params["fine"])
cost = snakemake.params["cost"]
interp = snakemake.params["interp"]

with tempfile.TemporaryDirectory() as tmpdir:
    fixed_tmp = pathlib.Path(tmpdir) / (pathlib.Path(fixed).name + ".nii.gz")
    shutil.copyfile(fixed, fixed_tmp)
    

    subprocess.run(
        [
            "flirt",
            "-in",
            image,
            "-ref",
            fixed_tmp,
            "-out",
            warped,
            "-omat",
            xfm,
            "-dof",
            dof,
            "-coarsesearch",
            coarse,
            "-finesearch",
            fine,
            "-cost",
            cost,
            "-interp",
            interp
        ],
        check=True
    )

    subprocess.run(
        [
            "flirt",
            "-in",
            warped,
            "-ref",
            fixed_tmp,
            "-out",
            resampled,
            "-dof",
            dof,
            "-applyisoxfm",
            "1",
        ],
        check=True
    )
