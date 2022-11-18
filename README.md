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

### About contact dataset
Each dataset is organized under a directory whose path you must pass into the dataset class. Each index of the dataset corresponds to a temporally paired color and depth image which is captured at 10 Hz. The other proprioceptive items, end-effector pose and force-torque wrench, are captured at a much higher frequency, 500 Hz and for each image sample, the nearest temporal neighbor is returned.

The `ContactDataset` class takes as init arguments the dataset path, the desired history window size T, and some other parameters that aren't really worth playing with lol. The history size is to return some past history of measurements at each index, such as returning not just a single frame but some T frames in the past. But for this project we should just stick with T = 1. It returns a dictionary of items that include the above but also return various other quantities that probably aren't relevant for this project. When batched, the dictionary values will usually return a tensor of dimension B x T x N where N is some item specific dimension (e.g. for images N = H x W).    

See `contact_dset_test.ipynb` file for example of loading samples from a dataloader which includes comments to explain use. 

## About dataloaders
API to interact with datasets should be put under ``dataloaders``.
They will be tracked by git and shared across computers.
They will be imported in scripts as abstract interfaces.
