# Auto-afids
Auto-afids uses random forest models to automatically locate 32 anatomical fiducials, as originally described in the Afids project.

# Preprocessing:
This is the initial step to align 25 OASIS-1 T1w brain images to MNI space (training set). Involves all files and folders except the folder, "autofid_main."
This step revolves around the Snakefile (see Snakefile for more details). Specifically, it takes images from a bids directory and their corresponding .fcsv files in the OAS1_unaligned folder as inputs. Afterwards, it uses FSL Flirt to rigidly register the image to an MNI space template. The .fcsv files are transformed as well. The outputs are found in the OAS1_aligned folder.

# Main pipeline:
In the autofid_main folder, there are three subdirectories: training, testing, and results. In the training folder, the coarse_train, med_train, and fine_train python scripts are used to build the random forest models. Training inputs (images and .fcsv files) are taken from the pythonimg and pythonfcsv folders, respectively, while the resultant models are stored in the models folder.
In the testing folder, there is a main script called autofid_main.py. In it is a function that accepts a new image, a ground_truth fiducial .fcsv file (optional), and a fiducial number. Running the function will output the predicted fiducial location.
The results folder gives the predicted fiducial coordinates for 5 test subjects in .fcsv files. It also gives distance errors compared to the ground_truth labels in .npy files. A summary of the results can be found in pandas_fid_errors.pdf.
