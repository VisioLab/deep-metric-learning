import h5py
import collections
import numpy as np
from .autoaugment import ImageNetPolicy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image


class FoodData(torch.utils.data.Dataset):
    def __init__(self, h5_path, transform=None):
        """
        Inputs:
            h5_path (Str): specifying path of HDF5 file to load
            transform (torch transforms): if None is skipped, otherwise torch
                                          applies transforms
        """
        self.h5_path = h5_path
        self.transform = transform

    def __getitem__(self, index):
        """
        Method for pulling images and labels from the initialized HDF5 file
        """
        with h5py.File(self.h5_path, "r") as f:
            X = f["images"][index]
            y = f["labels"][index]

        if self.transform is not None:
            X = Image.fromarray(X)
            X = self.transform(X)
        return X, y

    def __len__(self):
        with h5py.File(self.h5_path, "r") as f:
            return f["labels"].shape[0]



class CustomSampler(torch.utils.data.Sampler):

    def __init__(self, M, indices, labels):
        """
        this is a custom sampler to try to extend Kevins sampler.
        Change from: https://github.com/KevinMusgrave/pytorch-metric-learning
        """
        self.indices = indices
        self.labels = labels
        self.M = M
        self.unique_labels = np.unique(labels)
        self.label_indices = self.get_label_indices(labels)

        self.list_size = len(indices)
        if self.M*self.unique_labels.shape[0] < self.list_size:
            self.list_size -= (self.list_size) % (self.M*self.unique_labels.shape[0])
        
        self.valid_labels = np.unique(labels[indices])

    def get_label_indices(self, labels):
        """
        creates a label indices dictionary, where labels are keys and the values are
        a list of indices that are associated with that label.
        For example if the labels are [1, 1, 2, 3, 4, 5] then this func returns:
            >>> defaultdict(list,
                            {1: array([0, 1]),
                            2: array([2]),
                            3: array([3]),
                            4: array([4]),
                            5: array([5])})

        Inputs:
            labels - the array of labels to build into a dictionary
        Returns:
            label_indices - a dictionary as outlined above
        """
        # init an empty dict
        label_indices = collections.defaultdict(list)
        # fill our dict by going over the given indices and finding associated labels
        for i in self.indices:
            label_indices[labels[i]].append(i)

        # change them to np arrays!
        for k, v in label_indices.items():
            label_indices[k] = np.array(v, dtype=np.int)

        return label_indices

    def __iter__(self):

        iter_list = [0]*self.list_size
        
        print("Running Sampler")
        for i in range(0, self.list_size, self.M):
            # pick a random label, for now use a while loop in case
            # the label has 0 entries.=
            label = np.random.choice(self.valid_labels)
           
            # now pick M random indices from that label
            iter_list[i:i+self.M] = np.random.choice(self.label_indices[label],
                                                     size=self.M,
                                                     replace=len(self.label_indices[label]) < self.M)
        return iter(iter_list)

    def __len__(self):
        return self.list_size


def data_augmentation(hflip=True,
                      crop=False,
                      colorjitter=True,
                      rotations=False,
                      affine=False,
                      imagenet=False):
    """
    TO DO, need to decide how best to implement augmentations!!!
    This function is very experimental
    """

    augments = []
    if hflip:
        augments.append(transforms.RandomHorizontalFlip(p=0.5))
    if crop:
        augments.append(transforms.RandomResizedCrop(scale=(0.1, 1),
                                                    ratio=(0.75, 1.33),
                                                    size=128))
    if colorjitter:
        augments.append(transforms.ColorJitter(brightness=0.3,
                                        contrast=0.2,
                                        saturation=0.2,
                                        hue=0.3))
    if rotations:
        augments.append(transforms.RandomRotation(degrees=(-5, 5),
                                            expand=True))
        augments.append(transforms.Resize(128))
    if affine:
        augments.append(transforms.RandomAffine(degrees=5))

    if imagenet:
        augments.append(ImageNetPolicy())

    return augments


class CustomDataLoader():

    def __init__(self,
                 h5_path,
                 train_labels=np.arange(101),
                 batch_size=128, 
                 num_workers=3,
                 augmentations=None,
                 train_split=0.8,
                 M=3):
        """
        Inputs:
            h5_path str: the path of our dataset to pass to h5py.File
            train_labels np.array: the labels to train on
            batch_size int: the batch size of our dataloaders
            num_workers int: number of workers for DataLoaders
            augmentations transforms or None: transformations to apply to train set
            train_split float: train/val split, 0.8 -> 80%/20%
            M int: samples per class to return in trainloader CustomSampler
        """
        
        self.labels = h5py.File(h5_path, "r")["labels"][:]
        self.augmentations = augmentations
        self.train_labels = train_labels
        self.batch_size = batch_size
        self.M = M

        # build the transforms now
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformations = [transforms.ToTensor(),
                           normalize]
        if augmentations is not None:
            transformations = augmentations + transformations

        self.train_transform = transforms.Compose(transformations)
        self.val_transform = transforms.Compose([transforms.ToTensor(),
                                                 normalize])

        self.train_dataset = FoodData(h5_path, transform=train_transform)
        self.val_dataset = FoodData(h5_path, transform=val_transform)

        # get our train and validation indices!
        # first we get the indices associated with our train/val split
        t_label_ids = []
        for i in train_labels:
            t_label_ids.append(np.where(i == labels))
        t_label_ids = np.concatenate(t_label_ids, axis=1).squeeze()
        
        # now we make our train and val indices by splitting 80/20
        self.train_indices = np.random.choice(t_label_ids,
                                              int(len(t_label_ids)*train_split),
                                              replace=False)
        self.val_indices = np.setdiff1d(the_label_ids,
                                        self.train_indices)
        # finally the leftover indices are our holdout set
        self.holdout_indices = np.setdiff1d(np.arange(labels.shape[0]),
                                            t_label_ids)



def get_dataloaders(h5_path="data.h5", batch_size=128, num_workers=3, augmentations=None, M=3):
    """
    This function loads the h5 file and instantiates a FoodData object, it then builds dataloaders
    for the train set and returns necessary information.
    """

    transformations = [transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])]
    if augmentations is not None:
        transformations = augmentations + transformations

    train_transform = transforms.Compose(transformations)
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    # we create our two datasets
    labels = h5py.File(h5_path, "r")["labels"][:]
    
    # build our datasets using the train indices
    train_dataset = FoodData(h5_path, transform=train_transform)
    val_dataset = FoodData(h5_path, transform=val_transform)

    # get our train and validation indices!
    # first we get the indices associated with our train/val split
    train_labels = np.arange(101)
    t_label_ids = []
    for i in train_labels:
        t_label_ids.append(np.where(i == labels))
    t_label_ids = np.concatenate(t_label_ids, axis=1).squeeze()
    
    # now we make our train and val indices by splitting 80/20
    train_indices = np.random.choice(t_label_ids,
                                     int(len(t_label_ids)*0.8),
                                     replace=False)
    val_indices = np.setdiff1d(t_label_ids,
                               train_indices)
    # finally the leftover indices are our holdout set
    holdout_indices = np.setdiff1d(np.arange(labels.shape[0]),
                                   t_label_ids)
    # initialize our samplers with the selected indices
    sampler = CustomSampler(M=M,
                            indices=train_indices,
                            labels=labels)

    # now build the dataloaders with the selected samplers
    trainloader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=batch_size, num_workers=num_workers)

    return trainloader, val_dataset, train_indices, val_indices, holdout_indices, labels
