# Auto-afids
Auto-afids uses random forest models to automatically locate 32 anatomical fiducials, which were originally described in the Afids project.

Auto-afids consists of two workflows: `auto-afids-train`, which trains a set of random forest models from an input BIDS dataset (which is first registered to MNI space), and `auto-afids`, which automatically locates the 32 fiducials for each T1 weighted image in the input dataset.

## Installation

A python environment with the packages in `requirements.txt` installed is required for both workflows, and FSL and ANTS must be installed to use `auto-afids-train`.

## auto-afids-train

`auto-afids-train` is run with Snakebids through `auto-afids-train/run.py`. It is formatted as a BIDS app, so it can be run with:

`python3 auto-afids-train/run.py <input BIDS dataset> <output directory> <participant or group> <snakemake arguments>`

The output models will be available in `<output dir>/derivatives/models`.

## auto-afids

`auto-afids` is also run with Snakebids through `auto-afids/run.py`. The CLI is:

`python3 auto-afids/run.py <input BIDS dataset> <output directory> <participant or group> --model_dir <root model directory> <snakemake arguments>`

The output should be one text file for each of the 32 AFIDs. This workflow is a WIP, and properly formatted FCSV or JSON files should be produced for each subject in the future.
