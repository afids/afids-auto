# Auto-afids
Auto-afids uses random forest models to automatically locate 32 anatomical fiducials, which were originally described in the Afids project.

# Preprocessing:
This is the initial step to align 25 OASIS-1 T1w brain images to MNI space to create the training set. Involves all files and folders except the folder, "autofid_main."
This step revolves around the Snakefile (see Snakefile for more details). Specifically, it takes images from a bids directory and their corresponding .fcsv files in the OAS1_unaligned folder as inputs. Afterwards, it uses FSL Flirt to rigidly register the image to an MNI space template. The .fcsv files are transformed as well. 

To run, ssh login and cd to the folder with the snakefile. Then, input:

`snakemake -j [N]`

[N] denotes the number of cores to use.

The outputs are currently found in the OAS1_aligned folder.

# Main pipeline:
In the autofid_main folder, there are three subdirectories: training, testing, and results. 

## Training
In the training folder, the coarse_train, med_train, and fine_train python scripts are used to build the random forest models. Training inputs (images and .fcsv files) are taken from the pythonimg and pythonfcsv folders, respectively, while the resultant models are stored in the models folder. Models have already been trained and are stored in the models folder currently.

To train, go to the terminal and run: `python coarse_train.py`, or `python med_train.py`, or `python fine_train.py` to specify model training at each of 3 resolutions (downsampled by 4, normal, upsampled by 2). All 3 layers are needed to obtain an accurate final prediction.

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
