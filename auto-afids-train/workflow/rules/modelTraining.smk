rule featureExtract:
    input:
        nii_files=expand(
            bids(
                root=join(config["output_dir"], "flirt"),
                datatype="anat",
                suffix="T1w.nii.gz",
                space="MNI152NLin2009cAsym",
                **config["input_wildcards"]["t1w"],
            ),
            zip,
            **config["input_zip_lists"]["t1w"],
        ),
        fcsv_files=expand(
            bids(
                root=join(config["output_dir"], "auto-afids-train"),
                suffix="afids.fcsv",
                space="MNI152NLin2009cAsym",
                desc="ras",
                **config["input_wildcards"]["t1w"],
            ),
            zip,
            **config["input_zip_lists"]["t1w"],
        ),
    params:
        afid_num='{afid_num}',
        model_params=config['model_params'],
        feature_offsets=workflow.source_path(config["model_params"]["feature_offsets"]),
        train_level='{train_level}',
    output:
        bids(
            root=join(config["output_dir"], "auto-afids-train"),
            prefix="afid-{afid_num}",
            suffix="features.hkl",
            desc="{train_level}",
            space="MNI152NLin2009cAsym",
        )
    script:
        "../scripts/data_store.py"

rule modelTrain:
    input:
        rules.featureExtract.output
    params:
        model_params = config['model_params'],
    threads: workflow.cores * config['model_params']['num_threads']
    resources:
        mem_mb=config['model_params']['max_memory']
    output:
        bids(
            root=join(config["output_dir"], "auto-afids-train"),
            prefix="afid-{afid_num}",
            suffix="model",
            desc="{train_level}",
            space="MNI152NLin2009cAsym",
        ),
    script:
        "../scripts/train.py"
