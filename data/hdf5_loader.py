import h5py
import collections
import numpy as np
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
        Inspired by: https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/samplers/m_per_class_sampler.py
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