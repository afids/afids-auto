from os.path import join
from glob import glob

configfile: "config.yml"


# bids_dir  set in json file.
# can also override at command-line with e.g.:  --config bids_dir='path/to/dir'  or --configfile ...
bids_dir = config['bids_dir']
subjects = config['subjects']


rule all:
    input:
        mni_rigid = expand('OAS1_aligned/sub-{subject}/sub-{subject}_T1w_rigid.nii.gz',subject=subjects),
	tform_new = expand('OAS1_aligned/sub-{subject}/sub-{subject}_T1w_rigid_xfm_slicer.txt',subject=subjects),
        fid_mni_rigid = expand('OAS1_aligned/sub-{subject}/OAS1-{subject}_MR1_T1_MEAN_mni_rigid.fcsv',subject=subjects)

rule align_mni_rigid:
    input:
        moving = lambda wildcards: join(bids_dir,'PHASE2/OAS1_bids_MR1/sub-{subject}/anat/sub-{subject}_T1w.nii.gz'.format(subject=wildcards.subject)),
        fixed = 'template_align_mni_rigid.nii.gz'
    output:
        warped = 'OAS1_aligned/sub-{subject}/sub-{subject}_T1w_rigid.nii.gz',
        xfm = 'OAS1_aligned/sub-{subject}/sub-{subject}_T1w_rigid_xfm.mat' 
    envmodules: 'fsl'
    log: 'logs/align_mni_rigid/sub-{subject}_T1w.log'
    shell:
        'flirt -in {input.moving} -ref {input.fixed} -out {output.warped} -omat {output.xfm} -dof 6 -coarsesearch 60 -finesearch 15 &> {log}'

rule fsl_to_ras:
    input:
        moving = lambda wildcards: join(bids_dir,'PHASE2/OAS1_bids_MR1/sub-{subject}/anat/sub-{subject}_T1w.nii.gz'.format(subject=wildcards.subject)),
        reference = 'OAS1_aligned/sub-{subject}/sub-{subject}_T1w_rigid.nii.gz',
        xfm = 'OAS1_aligned/sub-{subject}/sub-{subject}_T1w_rigid_xfm.mat'
    output:
        xfm_new = 'OAS1_aligned/sub-{subject}/sub-{subject}_T1w_rigid_xfm_slicer.txt'
    envmodules: 'fsl'
    log: 'logs/fsl_to_ras/sub-{subject}_T1w.log'
    shell:
        './c3d_affine_tool -ref {input.reference} -src {input.moving} {input.xfm} -fsl2ras -o {output.xfm_new}'

rule fid_tform_mni_rigid:
    input:
        orig = lambda wildcards: '/project/6050199/dcao6/autofid/training/OAS1_orig/OAS1_{subject}_MR1_T1_MEAN.fcsv'.format(subject=wildcards.subject),
        xfm_new = 'OAS1_aligned/sub-{subject}/sub-{subject}_T1w_rigid_xfm_slicer.txt', 
        template = 'dummy.fcsv'
    output:
        fid_tform = 'OAS1_aligned/sub-{subject}/OAS1-{subject}_MR1_T1_MEAN_mni_rigid.fcsv'
    envmodules: 'fsl'
    log: 'logs/fid_tform_mni_rigid/sub-{subject}_T1w.log'
    shell:
        'arg1={input.orig}; arg2={input.xfm_new}; arg3={input.template}; arg4={output.fid_tform}; python tform_script.py $arg1 $arg2 $arg3 $arg4 &> {log}'
