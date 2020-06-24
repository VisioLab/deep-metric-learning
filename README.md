# Deep Metric Learning

This repository contains code for my MSc Thesis and work at VisioLab. It focuses on the development of a deep metric learning model for incremental learning in the classification regime. For a deep overview of the theory and design decisions behind the architecture refer to my MSc thesis which will be added to the repository.

## Three Stage Architecture

![network](./images/network.png)

## How to Run Basics

Due to relative imports, and the general directory design, this model is currently only meant to be used through `main.py`.  The simplest way to start is to import the architecture, instantiate it and then run train with your preferred hyperparameters on your dataset.

```python
from model.model import ThreeStageNetwork

# from scratch, no need to load weights or indices
model = ThreeStageNetwork()
model.setup_data(path="my_training_data.hdf5",
                 batch_size=128)
model.train(n_epochs=10,
            loss_ratios=[1,1,1,3])

# load pretrained and finetune
model = ThreeStageNetwork()
model.load_weights("models/models")
model.setup_data(path="my_training_data.hdf5",
                 batch_size=128,
                 load_indices=True,
                 indices_path="data_indices.npz")
model.train(n_epochs=10,
            loss_ratios=[1,1,1,3])
```

In the first case we first instantiate the network architecture with the default values. We then setup our data by providing the path to our training data file and the batch size to use. This will prepare our dataloaders, select the indices to train, validate and holdout on. Finally we call the train method with the number of epochs and the loss ratios to pass onto our four loss functions.

When training is finished the model will automatically be saved and the entire log files will be zipped up.

## Configuration, Dependencies and Directory Structure

### Dependencies

This is a list of the non-common dependencies used in the project, a full list will be provided in requirements.txt:

* The core dependencies are the [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) library by Kevin from which we utilize the pair based losses and the MS miner. 
* We also use [efficientnet_pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) for the pretrained EfficientNet weights.
*  [UMAP](https://umap-learn.readthedocs.io/en/latest/) is used for visualization purposes.
* [tqdm](https://github.com/tqdm/tqdm) is used for loading bars
* [pynvml](https://pypi.org/project/pynvml/) is used for GPU monitoring

### Directory Structure

```
> data
	> data
		> # training files
	> autoaugment.py
	> hdf5_loader.py
> images
> model
	> losses.py
	> model.py
	> utils.py
README.md
```

 `data` contains the augmentation policy and everything we need to construct the datasets and dataloaders from our hdf5 dataset. `model` contains the actual architecture and loss func implementations as well as some utilities for directory management and S3 uploads/downloads.
