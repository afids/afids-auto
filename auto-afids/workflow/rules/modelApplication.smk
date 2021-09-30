rule applyModel:
    input:
        nii_files=expand(
            bids(
                root=config["input_dir"], 
                datatype="anat",
                suffix="T1w.nii.gz",
                space="MNI152Nlin2009cAsym",
                **config["input_wildcards"]["t1w"],
            )
            zip,
            **config["input_zip_lists"]["t1w"]
        ),
        models=expand(
            bids(
                root=join(config["model_dir"], "derivatives", "models")
                prefix="afid-{afid_num}",
                suffix="model",
                desc="{train_level}"
                space="MNI152NLin2009cAsym"
            ),
            train_level=["coarse", "med", "fine"],
        ),
        feature_offsets="resources/feature_offsets.npz"
