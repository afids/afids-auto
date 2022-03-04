rule initialFeatures:
    input:
        nii_file=config["input_path"]["t1w"],
    params:
        feature_offsets=workflow.source_path(
            "../../resources/feature_offsets.npz"
        ),
    output:
        initial_features=bids(
            join(config["output_dir"], "auto-afids"),
            suffix="initial_features.npy",
            **config["input_wildcards"]["t1w"],
        )
    script:
        "../scripts/gen_initial_features.py"

rule applyModel:
    input:
        nii_file=config["input_path"]["t1w"],
        models=expand(
            bids(
                root=join(config["model_dir"], "auto-afids-train"),
                prefix="afid-{afid_num}",
                suffix="model",
                desc="{train_level}",
                space="MNI152NLin2009cAsym",
                res="iso1",
            ),
            train_level=["coarse", "medium", "fine"],
            allow_missing=True,
        ),
        initial_features=rules.initialFeatures.output.initial_features,
    params:
        afid_num="{afid_num}",
        feature_offsets=workflow.source_path(
            "../../resources/feature_offsets.npz"
        ),
    output:
        afid=bids(
            root=join(config["output_dir"], "derivatives", "auto-afids"),
            prefix="afid-{afid_num}",
            suffix="afid.txt",
            space="MNI152NLin2009cAsym",
            **config["input_wildcards"]["t1w"]
        )
    script:
        "../scripts/autofid_main.py"
