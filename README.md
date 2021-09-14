# Auto-afids
Auto-afids uses random forest models to automatically locate 32 anatomical fiducials, which were originally described in the Afids project.

## Preface
All data related to Auto-afids are stored on Graham. Specifically, everything found in this Github repository is stored in: '/scratch/dcao6/autofid_final/Auto-afids'. 
Auto-afids uses data from MRI images. All relevant image and transform files are stored in: '/scratch/dcao6/autofid_final/data'. The currently available datasets are:
* **OASIS (25 subjects)**: '/scratch/dcao6/autofid_final/data/OASIS'
* **lhsc_dbs (40 subjects)**: '/scratch/dcao6/autofid_final/data/clinical'
* **HCP (30 subjects)**: '/scratch/dcao6/autofid_final/data/HCP'

## Preprocessing
This is the initial step to align the T1w brain images to MNI space to create the training set using FLIRT (rigid registration to MNI space).
This step is a registration procedure and pertains to the 'registration' folder. Specifically, running the Snakefile in '/registration/workflow' will run the procedure. In this case, snakemake is configured to work on the OASIS dataset.

To run the Snakefile on Graham, ssh login and cd to the folder with the snakefile: '/scratch/dcao6/autofid_final/Auto-afids/registration/workflow'.

Then, make sure the appropriate modules are loaded. FSL and ANTS are the two modules being used. First input:

`module load fsl`

`module load ants`

Then, input:

`snakemake -j [N]`

[N] denotes the number of cores to use.

The outputs will be the newly aligned images and their corresponding fiducials (.fcsv files).


To skip the preprocessing and just examine the inputs and results of this step, look at:
'/scratch/dcao6/autofid_final/data/OASIS/bids' for the input images and fiducials; and
'/scratch/dcao6/autofid_final/data/OASIS/deriv/afids_mni' for the output images and fiducials.

# Main pipeline:
In the autofid_main folder, there are three subdirectories: training, testing, and results. 

## Training
In the 'training folder', the coarse_train.py, med_train.py, and fine_train.py python scripts are used to build the random forest models. Training inputs (images and .fcsv files) are taken from the outputs of the preprocessing step, located here: '/scratch/dcao6/autofid_final/data/OASIS/deriv/afids_mni'.
When new models are created via these scripts, they will be stored in a new 'models' subfolder that resides in the 'training' folder.

To train, go to the terminal and run: `python coarse_train.py`, or `python med_train.py`, or `python fine_train.py` to specify model training at each of 3 resolutions (downsampled by 4, normal, upsampled by 2). All 3 layers are needed to obtain an accurate final prediction.

In some cases, some python packages may be missing. To circumvent this, make sure python is loaded already, and then for the missing package that gets prompted on the screen, input:

`pip install --user [package]`

[package] denotes the name of the python package that is missing.

To skip the hassle of training from scratch, models have been trained already and are stored on Graham at a secondary location:
'/project/6050199/dcao6/autofid/models/new/'.

## Testing
In the 'testing' folder, there is a main script called autofid_main.py. In it is a function that accepts a new image, a ground_truth fiducial .fcsv file (optional), and a fiducial number. Running the function will output the predicted fiducial location.

To run autofid_main.py using examples that come with this package (see 'HCP_testing' folder), use the following line of code on the terminal after moving to this directory.

`arg1='HCP_testing/sub-103111_space-MNI152NLin2009cAsym_T1w.nii.gz'; arg2='HCP_testing/sub-103111_space-MNI152NLin2009cAsym_desc-ras_afids_ground_truth.fcsv'; arg3=1; python autofid_main.py $arg1 $arg2 $arg3`

arg1: image file

arg2: .fcsv ground_truth file (optional)

arg3: fiducial number

Arguments can be modified to encompass custom image files and fiducial files outside of the samples available here.

To use autofid_main without a .fcsv file, the code input to the terminal can look like this:

`arg1='HCP_testing/sub-103111_space-MNI152NLin2009cAsym_T1w.nii.gz'; arg3=1; python autofid_main.py $arg1 $arg3`

The autofid_main.py script uses models stored in the secondary location, '/project/6050199/dcao6/autofid/models/new/', to run testing. 

## Results
After testing, you should be able to see the anatomical fiducial localization error (AFLE) printed on Graham for the chosen fiducial.

**Notes**

The 'modelling' folder contains a snakemake process for model training (work-in-progress).

In '/registration/workflow', registration_decoupling.py is a file that is run separately in python. What it does is it computes the linear and nonlinear transformations that come from running fmriprep's linear and nonlinear registration procedures on the original HCP images.
Essentially, we use fmriprep's registration to determine registration localization error of manually labelled fiducials, and registration_decoupling.py helps apply and save the transformations for replicability.
This is a standalone script and is only used after the main pipeline of Auto-afids is performed (FLIRT registration, training, and testing).
