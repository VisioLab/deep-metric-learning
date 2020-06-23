
import torch
import numpy as np
import umap # used to visualize embeddings
from PIL import Image
import matplotlib.pyplot as plt
from cycler import cycler # some plotting thing
import torch
import subprocess
import warnings
from zipfile import ZipFile
import os


def umap_plot(umap_embeddings, labels, save_path=None):
    """
    This is primarily from Kevins hook example in pytorch-metric-learning library

    Inputs:
        umap_embeddings np.array - 2D embedding output from umap.UMAP()
        labels np.array - 1D labels associated with embeddings, used for colors
        save_path None or Str - where to save plots. See example Useage. If None
                                then plot is not saved.

    Example Useage:
        >>> e, l = get_embeddings([trunk, embedder], train_dataset)
        >>> lo, la = get_logits(classifier, e, l)
        >>> embeds = umap.UMAP().fit_transform(e)
        >>> umap_plot(embeds, l, save_path=None) # do not save image

        Or an example to save the plot, where epoch is our current epoch val
        >>> umap_plot(embeds, l, save_path=f"images/umap_train_epoch{epoch}")
    """
    label_set = np.unique(labels)
    # all this is for figure
    fig = plt.figure(figsize=(12,8))
    plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, len(label_set))]))
    for i in range(len(label_set)):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    if save_plot:
        plt.savefig(save_path, dpi=300)
    plt.show()



def calc_accuracy(logits, labels):
    """
    func to compute accuracy from input logits and labels, meant to be used
    with tensor inputs from Pytorch. This func essentially just implements
    np.sum(np.argmax(logits, axis=0) == labels)/len(labels).
    Make sure if logits is in cuda mode labels need to be in cuda mode.

    Inputs:
        logits tensor - logits of dim num classes x N
        labels tensor - associated labels vector size N
    Outputs:
        train_acc float - float of accuracy between [0, 1]
    
    Example Useage:
        >>> inputs, labels = data # get inputs and labels for current batch
        >>> logits = neural_network(inputs) # forward pass over inputs
        >>> calc_accuracy(logits, labels.cuda())
    """
    max_vals, max_indices = torch.max(logits, 1)
    train_acc = (max_indices == labels).sum().item()/max_indices.size()[0]
    return train_acc


def prepare_experiment():

    directory_list = ["logs", "models"]
    for directory in directory_list:
        if os.path.exists(directory) is False:
            os.mkdir(directory)


def zip_files(directory_list=["logs", "models"], experiment_id=0, zip_train_data=False):
    """
    utility to zip files at the end of experiment. Looks for all files in
    directories "logs" and "models" to zip. saves zipped file in local path
    under name "experiment_x.zip" where x is user given experiment_id.
    if file name already exists, warning is prompted and experiment_id is
    incremented until it an unused name is found.

    Inputs:
        experiment_id int - id to put in place of x in "experiment_x.zip"
    Warnings:
        UserWarning if file "experiment_x.zip" exists.
    """

    while True:
        file_name = f"experiment_{experiment_id}.zip"
        if os.path.exists(file_name):
            warnings.warn(f"File {file_name} exists, incrementing id", UserWarning)
            experiment_id += 1
        else:
            break

    print("Zipping files!")
    directory_list = directory_list
    with ZipFile(file_name, "w") as zipObj:
        for directory in directory_list:
            for folderName, _, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(folderName, filename)
                    if "hdf5" not in file_path:
                        print(f"Zipping {file_path}")
                        # Add file to zip
                        zipObj.write(file_path, os.path.basename(file_path))

    return experiment_id


def download_from_s3(file_name, destination, bucket="msc-thesis"):
    """
    util to download an s3 file, currently uses subprocess instead of AWS CLI or Boto3.

    Inputs:
        file_name str - relative or full path to file to download locally
        destination str - path without file name, relative or full to save file to
        bucket str - AWS S3 bucket to downlad file from

    Example Useage:
        >>> download_from_s3(file_name="data_128_v2.hdf5",
                             destination="data",
                             bucket="msc-thesis")
        This is equivalent to terminal command:
        >>> aws s3 cp s3://msc-thesis/data_224_v1.hdf5 data/data_224_v1.hdf5  
    """

    if destination is not None:
        file_path = f"{destination}/{file_name}"
    else:
        file_path = file_name
    
    subprocess.run(["aws", "s3", "cp", f"s3://{bucket}/{file_name}", file_path])


def upload_to_s3(file_name, destination, bucket="msc-thesis"):
    """
    util to upload a file to s3, currently uses subprocess instead of AWS CLI or Boto3

    Inputs:
        file_name str - relative or full path to file to copy to s3 bucket
        destination str - path without file name, relative or full
        bucket str - AWS S3 bucket to upload file to

    Example Useage:
        >>> upload_to_s3(file_name="batch_history.csv",
                         destination="logs",
                         bucket="msc-thesis")
        This is equivalent to terminal command:
        >>> aws s3 cp logs/batch_history.csv s3://msc-thesis/batch_history.csv
    """

    if destination is not None:
        file_path = f"{destination}/{file_name}"
    else:
        file_path = file_name

    subprocess.run(["aws", "s3", "cp", file_path, f"s3://{bucket}/{file_name}"])
