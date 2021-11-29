from snakebids import bids
from os.path import join
import os


rule align_mni_rigid:
    input:
        image=config["input_path"]["t1w"]
    output:
        resampled=bids(
            root=join(config["output_dir"], "flirt"),
            datatype="anat",
            suffix="T1w.nii.gz",
            space="MNI152NLin2009cAsym",
            res="iso1",
            **config["input_wildcards"]["t1w"],
        ),
        warped=bids(
            root=join(config["output_dir"], "flirt"),
            datatype="anat",
            suffix="T1w.nii.gz",
            space="MNI152NLin2009cAsym",
            **config["input_wildcards"]["t1w"],
        ),
        xfm=bids(
            root=join(config["output_dir"], "flirt"),
            datatype="anat",
            suffix="xfm.mat",
            space="MNI152NLin2009cAsym",
            res="iso1",
            **config["input_wildcards"]["t1w"],
        ), 
    params:
        fixed = workflow.source_path(config['template']),
        dof = config['flirt']['dof'],
        coarse = config['flirt']['coarsesearch'],
        fine = config['flirt']['finesearch'],
        cost = config['flirt']['cost'],
        interp = config['flirt']['interp'],
    envmodules: 'fsl'
    #log: 'logs/align_mni_rigid/sub-{subject}_T1w.log'
    script:
        "../scripts/run_flirt.py"

rule fsl_to_ras:
    input:
        warped=rules.align_mni_rigid.output.resampled,
        xfm=rules.align_mni_rigid.output.xfm,
        moving_vol=config["input_path"]["t1w"],
    output:
        xfm_new=bids(
            root=join(config["output_dir"], "c3d_affine_tool"),
            datatype="anat",
            suffix="xfm.mat",
            space="MNI152NLin2009cAsym",
            desc="ras",
            res="iso1",
            **config["input_wildcards"]["t1w"],
        ),
        tfm_new=bids(
            root=join(config["output_dir"], "c3d_affine_tool"),
            datatype="anat",
            suffix="xfm.tfm",
            space="MNI152NLin2009cAsym",
            desc="ras",
            res="iso1",
            **config["input_wildcards"]["t1w"],
        ),
    params:
        c3d_affine_tool = workflow.source_path("../../resources/c3d_affine_tool")
    shell:
        'c3d_affine_tool -ref {input.warped} -src {input.moving_vol} {input.xfm} -fsl2ras -o {output.xfm_new} && \
        c3d_affine_tool -ref {input.warped} -src {input.moving_vol} {input.xfm} -fsl2ras -oitk {output.tfm_new}'

rule fid_tform_mni_rigid:
    input:
        xfm_new=rules.fsl_to_ras.output.xfm_new,
        groundtruth=bids(
            root=join(config["bids_dir"], 'derivatives', 'afids_groundtruth'),
            space="T1w",
            desc="groundtruth",
            suffix="afids.fcsv",
            **config["input_wildcards"]["t1w"],
        ),
    params:
        template = workflow.source_path('../../resources/dummy.fcsv'),
    output:
        fcsv_new=bids(
            root=join(config['output_dir'], 'auto-afids-train'),
            suffix="afids.fcsv",
            space='MNI152NLin2009cAsym',
            desc='ras',
            res="iso1",
            **config["input_wildcards"]["t1w"],
        ),
        reg_done = touch(
            bids(
                root=join(config['output_dir'], 'auto-afids-train'),
                suffix='registration.done',
                **config["input_wildcards"]["t1w"],
            )
        ),
    #log: 'logs/fid_tform_mni_rigid/sub-{subject}_T1w.log'
    script:
        '../scripts/tform_script.py'
