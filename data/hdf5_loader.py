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

    def __init__(self, M, indices, labels, batch_size, class_weights=None):
        """
        this is a custom sampler to try to extend Kevins sampler.
        Change from: https://github.com/KevinMusgrave/pytorch-metric-learning
        """
        self.indices = indices
        self.labels = labels
        self.batch_size = batch_size
        self.M = M
        self.unique_labels = np.unique(labels)
        self.label_indices = self.get_label_indices(labels)
        self.class_weights = class_weights

        self.list_size = len(indices)
        if self.M*self.unique_labels.shape[0] < self.list_size:
            self.list_size -= (self.list_size) % (self.M*self.unique_labels.shape[0])

        # valid classes (classes that have items in them)
        self.valid_labels = np.unique(labels[indices])
        # whether or not to replace labels when picking classes
        self.replace_labels = len(self.valid_labels) < int(np.ceil(self.list_size/self.M))

        # if no weights given start uniform
        if self.class_weights is None:
            self.class_weights = np.ones(len(self.valid_labels)) / len(self.valid_labels)
        else:
            # normalize to equal to 1
            self.class_weights /= np.sum(self.class_weights)

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

        # pick labels we will use (we need batch_size / M classes)
        # labels = np.random.choice(self.valid_labels, 
        #                           size=int(np.ceil(self.list_size/self.M))+1,
        #                           replace=self.replace_labels,
        #                           p=self.class_weights)

        print(self.class_weights)

        num_batches = int(np.ceil(self.list_size/self.batch_size))
        labels = []
        for _ in range(num_batches):
            labels.append(np.random.choice(self.valid_labels,
                                           size=int(np.ceil(self.batch_size/self.M)),
                                           replace=len(self.valid_labels) < int(np.ceil(self.batch_size/self.M)),
                                           p=self.class_weights))
        labels = np.hstack(labels)
        
        for i in range(0, self.list_size, self.M):
            # now pick M random indices from that label and add to iter_list
            iter_list[i:i+self.M] = np.random.choice(self.label_indices[labels[i//self.M]],
                                                     size=self.M,
                                                     replace=len(self.label_indices[labels[i//self.M]]) < self.M)
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

    augments = [transforms.Resize(224)]
    if hflip:
        augments.append(transforms.RandomHorizontalFlip(p=0.5))
    if crop:
        augments.append(transforms.RandomResizedCrop(scale=(0.6, 1),
                                                    size=224))
    if colorjitter:
        augments.append(transforms.ColorJitter(brightness=(0.8, 1.3),
                                        contrast=(0.8, 1.2),
                                        saturation=(0.9, 1.2),
                                        hue=(-0.05, 0.05)))
    if rotations:
        augments.append(transforms.RandomRotation(degrees=(-5, 5),
                                            expand=True))
        augments.append(transforms.CenterCrop(224))
    if affine:
        augments.append(transforms.RandomAffine(degrees=5))

    if imagenet:
        augments.append(ImageNetPolicy())
    else:
        augments.append(transforms.CenterCrop(224))

    return augments


def get_train_val_holdout_indices(labels, train_labels=None, train_split=0.8):

    # labels to train on, rest will be in holdout set, if none given select all classes
    if train_labels is None:
        train_labels = np.arange(len(np.unique(labels)))

    t_label_ids = []
    for i in train_labels:
        t_label_ids.append(np.where(i == labels))
    t_label_ids = np.concatenate(t_label_ids, axis=1).squeeze()

    train_indices = np.random.choice(t_label_ids,
                                     int(len(t_label_ids)*train_split),
                                     replace=False)
    val_indices = np.setdiff1d(t_label_ids,
                               train_indices)
    # finally the leftover indices are our holdout set
    holdout_indices = np.setdiff1d(np.arange(labels.shape[0]),
                                   t_label_ids)

    return train_indices, val_indices, holdout_indices

 
def get_dataloaders(dataset,
                    h5_path="data.h5",
                    batch_size=128,
                    num_workers=3, 
                    augmentations=None, 
                    M=3,
                    labels=None,
                    train_labels=None,
                    train_indices=None,
                    class_weights=None):
    """
    This function loads the h5 file and instantiates a FoodData object, it then builds dataloaders
    for the train set and returns necessary information.
    """

    transformations = [transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])]
    if augmentations is not None:
        transformations = augmentations + transformations

    train_transform = transforms.Compose(transformations)
    val_transform = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    # build our datasets using the train indices
    if h5_path is None:
        train_dataset = dataset(transform=train_transform)
        val_dataset = dataset(transform=val_transform)
    else:
        train_dataset = dataset(h5_path, transform=train_transform)
        val_dataset = dataset(h5_path, transform=val_transform)


    # initialize our samplers with the selected indices
    sampler = CustomSampler(M=M,
                            indices=train_indices,
                            labels=labels,
                            batch_size=batch_size,
                            class_weights=class_weights)

    # now build the dataloaders with the selected samplers
    trainloader = DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=batch_size, num_workers=num_workers)

    return trainloader, val_dataset
