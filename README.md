# Auto-afids
Auto-afids uses random forest models to automatically locate 32 anatomical fiducials, which were originally described in the Afids project.

Auto-afids consists of two workflows: `auto-afids-train`, which trains a set of random forest models from an input BIDS dataset (which is first registered to MNI space), and `auto-afids`, which automatically locates the 32 fiducials for each T1 weighted image in the input dataset.

## Installation / Contributing

Clone the git repository. `auto-afids` dependencies are managed with Poetry, which will need to be installed. You can find the instructions on the [poetry website](https://python-poetry.org/docs/master/). Once installed, the development environment can setup with the following commands:

```
poetry install
poetry run poe setup
```

[poethepoet](https://github.com/nat-n/poethepoet) is used as a task runner. You can see what commands are available by running 

```
poetry run poe
```

Additionally, pre-commit hooks (installed via the `poe setup` command) is used to lint and format code (we use [black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort), [pylint](https://pylint.org/), and [flake8](https://flake8.pycqa.org/en/latest/)).

## auto-afids-train

`auto-afids-train` is run with Snakebids through `auto-afids-train/run.py`. It is formatted as a BIDS app, so it can be run with:

`python3 auto-afids-train/run.py <input BIDS dataset> <output directory> <participant or group> <snakemake arguments>`

The output models will be available in `<output dir>/derivatives/models`.

## auto-afids

`auto-afids` is also run with Snakebids through `auto-afids/run.py`. The CLI is:

`python3 auto-afids/run.py <input BIDS dataset> <output directory> <participant or group> --model_dir <root model directory> <snakemake arguments>`

The output should be one text file for each of the 32 AFIDs. This workflow is a WIP, and properly formatted FCSV or JSON files should be produced for each subject in the future.
