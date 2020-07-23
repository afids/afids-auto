rule align_mni_rigid:
    input:
        moving = join(input_dir, 'bids',subj_sess_dir,datatype,subj_sess_prefix + suffix),
        fixed = config['template']
    output:
        warped = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype,subj_sess_prefix + '_space-MNI152NLin2009cAsym' + suffix ),
        xfm = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype,subj_sess_prefix + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.mat' ),
    envmodules: 'fsl'
    #log: 'logs/align_mni_rigid/sub-{subject}_T1w.log'
    shell:
        'flirt -in {input.moving} -ref {input.fixed} -out {output.warped} -omat {output.xfm} -dof 6 -coarsesearch 60 -finesearch 15'

rule fsl_to_ras:
    input:
        moving = join(input_dir, 'bids',subj_sess_dir,datatype,subj_sess_prefix +  suffix ),
        reference = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype,subj_sess_prefix + '_space-MNI152NLin2009cAsym' + suffix ),
        xfm = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype,subj_sess_prefix + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.mat' ),
    output:
        xfm_new = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype,subj_sess_prefix + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_desc-ras_xfm.txt' ),
        tfm_new = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype,subj_sess_prefix + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_desc-ras_xfm.tfm' )
    envmodules: 'fsl'
    #log: 'logs/fsl_to_ras/sub-{subject}_T1w.log'
    shell:
        'resources/c3d_affine_tool -ref {input.reference} -src {input.moving} {input.xfm} -fsl2ras -o {output.xfm_new} && \
        resources/c3d_affine_tool -ref {input.reference} -src {input.moving} {input.xfm} -fsl2ras -oitk {output.tfm_new}'

rule fid_tform_mni_rigid:
    input:
        orig = join(input_dir, 'deriv','afids','sub-{subject}',datatype_afids,'sub-{subject}' + '_space-' + entities['suffix'] + '_desc-average' + suffix_afids ),
        xfm_new = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype,subj_sess_prefix + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_desc-ras_xfm.txt' ),
        tfm_new = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype,subj_sess_prefix + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_desc-ras_xfm.tfm' ),
        template = 'resources/dummy.fcsv'
    output:
        fid_tform = join(output_dir, 'deriv','mni_space',subj_sess_dir,datatype_afids,subj_sess_prefix + '_space-MNI152NLin2009cAsym' + suffix_afids)
    envmodules: 'fsl'
    #log: 'logs/fid_tform_mni_rigid/sub-{subject}_T1w.log'
    script:
        '../scripts/tform_script.py'