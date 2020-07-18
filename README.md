---
tags: [Import-7a2f]
title: Auto-afids
created: '2020-07-18T04:27:30.292Z'
modified: '2020-07-18T07:34:12.470Z'
---

# Auto-afids
Auto-afids uses random forest models to automatically locate 32 anatomical fiducials, which were originally described in the Afids project.

## Preprocessing
This is the initial step to align the T1w brain images to MNI space to create the training set. The currently available datasets are:
* **OASIS (25 subjects)**: /scratch/dcao6/autofid/data/OASIS
* **lhsc_dbs (40 subjects)**: /scratch/dcao6/autofid/data/lhsc_dbs
* **HCP (30 subjects)**: /scratch/dcao6/autofid/data/HCP

This step revolves around the `workflow/Snakefile` (see Snakefile for more details). Specifically, from the listed BIDS datasets it takes the following data as input:
* ***/bids** which stores the nifti files in BIDS format
* ***/deriv/afids** which stores the `.fcsv` files with the associated coordinates of 32 anatomical fiducial points

Using FSL `Flirt`, the imaging data is rigidly registered to MNI space (). The transform is applied to the imaging data and the associated  `.fcsv` files. 

To run, ssh login and cd to the folder with the snakefile. First, make sure the appropriate modules are loaded. FSL and ANTS are the two modules being used. First input:

`module load fsl`

`module load ants`

Then, input:

`snakemake -j [N]`

[N] denotes the number of cores to use.

The outputs will be created in a 'OAS1_aligned' folder.

# Main pipeline:
In the autofid_main folder, there are three subdirectories: training, testing, and results. 

## Training
In the training folder, the coarse_train, med_train, and fine_train python scripts are used to build the random forest models. Training inputs (images and .fcsv files) are taken from folders on Graham (see code for more info). When new models are created via these scripts, they will be stored in a new 'models' folder.

To train, go to the terminal and run: `python coarse_train.py`, or `python med_train.py`, or `python fine_train.py` to specify model training at each of 3 resolutions (downsampled by 4, normal, upsampled by 2). All 3 layers are needed to obtain an accurate final prediction.

In some cases, some python packages may be missing. To circumvent this, make sure python is loaded already, and then for the missing package that gets prompted on the screen, input:

`pip install --user [package]`

[package] denotes the name of the python package that is missing.

To skip the hassle of training from scratch, models have been trained already and are stored on Graham.

## Testing
In the testing folder, there is a main script called autofid_main.py. In it is a function that accepts a new image, a ground_truth fiducial .fcsv file (optional), and a fiducial number. Running the function will output the predicted fiducial location.

To run autofid_main using examples that come with this package (see OAS1 folder), use the following line of code on the terminal after moving to this directory.

`arg1='OAS1/sub-0109_T1w.nii'; arg2='OAS1/OAS1_0109_MR1_T1_MEAN.fcsv' ;arg3=1; python autofid_main.py $arg1 $arg2 $arg3`

arg1: image file

arg2: .fcsv ground_truth file (optional)

arg3: fiducial number

Arguments can be modified to encompass custom image files and fiducial files outside of the samples available here.

To use autofid_main without a .fcsv file, the code input to the terminal can look like this:

`arg1='OAS1/sub-0109_T1w.nii'; arg3=1; python autofid_main.py $arg1 $arg3`

## Results
The results folder gives the predicted fiducial coordinates for 5 test subjects in .fcsv files. It also gives distance errors compared to the ground_truth labels in .npy files. A summary of the results can be found in pandas_fid_errors.pdf.
