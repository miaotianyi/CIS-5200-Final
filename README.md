# CIS-5200-Final
CIS-5200 Machine Learning final project

## About models
Models are put under the ``models`` directory.

## About experiments
Experiment scripts are under the ``scripts`` directory,
separate from models. They can import from models.

## About datasets
Datasets are under the ``datasets`` directory
on your local machine, using symbolic link when necessary.

Please do not actually upload the datasets to GitHub.
By default, anything under ``datasets`` will be ignored.

## About dataloaders
API to interact with datasets should be put under ``dataloaders``.
They will be tracked by git and shared across computers.
They will be imported in scripts as abstract interfaces.
