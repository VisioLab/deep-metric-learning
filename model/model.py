"""
This contains the entire code to build and train the deep metric learning models with the Food-101 dataset.

Third party libraries:
    pytorch + torchvision - quite obvious why
    umap - purely for visualization
    efficientnet_pytorch - using it as the base model, in future can save weights rather than using library
    tqdm - loading bars
    PIL - for image transforms
    matplotlib + cycler - visualizations
    h5py - storage and access datastructure for FoodDataset class
    pytorch_metric_learning - Kevins library, currently only using it for the losses, miner and sampler
                              This can be replaced with custom code once final choices have been made.

"""

import argparse
import collections
import numpy as np
from sklearn.metrics import classification_report
import functools

# Import pytorch metric learning stuff
from pytorch_metric_learning import losses, miners, samplers
from pytorch_metric_learning.utils import common_functions

# import pytorch stuff
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.optim import Adam, RMSprop, AdamW, SGD
from torchvision import datasets, transforms
import torchvision.models as models
import torchvision # to get access to the pretrained models
# plotting libraries
from tqdm.auto import tqdm
from PIL import Image

# file management
import h5py

# logging
import os
import subprocess
import warnings
from pynvml import *
import csv
import os
import json
from time import time
from datetime import datetime

# my modules
from .utils import calc_accuracy
from .losses import *
from data.hdf5_loader import *
from data.autoaugment import ImageNetPolicy
from .knn import knn_sim, get_weights, impostor_weights
from efficientnet import EfficientNet

class Network(nn.Module):

    def __init__(self, layer_sizes, neuron_fc=2048, activate_last=False, num_layers=2):
        super().__init__()

        layers = []

        if activate_last:
            layers += [nn.ReLU(True)]

        if num_layers == 1:
            layers += [nn.Linear(layer_sizes[0], layer_sizes[1])]

        elif num_layers == 2:
            layers += [nn.Linear(layer_sizes[0], neuron_fc),
                       nn.BatchNorm1d(neuron_fc, momentum=0.9),
                       nn.ReLU(True),
                       nn.Dropout(0.2),
                       nn.Linear(neuron_fc, layer_sizes[1])]
    
        self.classifier = nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        return self.classifier(x)


class ThreeStageNetwork():

    def __init__(self,
                 num_classes=101,
                 embedding_size=512,
                 trunk_architecture="efficientnet-b0",
                 trunk_optim="RMSprop",
                 embedder_optim="RMSprop",
                 classifier_optim="RMSprop",
                 trunk_lr=1e-4,
                 embedder_lr=1e-3, 
                 classifier_lr=1e-3,
                 weight_decay=1.5e-6,
                 trunk_decay=0.98,
                 embedder_decay=0.93,
                 classifier_decay=0.93,
                 log_train=True,
                 gpu_id=0):
        """
        Inputs:
            num_classes int: Number of Classes (for Classifier purely)
            embedding_size int: The size of embedding space output from Embedder
            trunk_architecture str: To pass to self.get_trunk() either efficientnet-b{i} or resnet-18/50 or mobilenet
            trunk_optim optim: Which optimizer to use, such as adamW
            embedder_optim optim: Which optimizer to use, such as adamW
            classifier_optim optim: Which optimizer to use, such as adamW
            trunk_lr float: The learning rate for the Trunk Optimizer
            embedder_lr float: The learning rate for the Embedder Optimizer
            classifier_lr float: The learning rate for the Classifier Optimizer
            weight_decay float: The weight decay for all 3 optimizers
            trunk_decay float: The multiplier for the Scheduler y_{t+1} <- trunk_decay * y_{t}
            embedder_decay float: The multiplier for the Scheduler y_{t+1} <- embedder_decay * y_{t}
            classifier_decay float: The multiplier for the Scheduler y_{t+1} <- classifier_decay * y_{t}
            log_train Bool: whether or not to save training logs
            gpu_id Int: Only currently used to track the GPU useage
        """

        self.gpu_id = gpu_id
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(f"cuda")
        self.pretrained = False # this is used to load the indices for train/val data for now
        self.log_train = log_train

        # build three stage network
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.MLP_neurons = 2048 # output size of neural network + size used inside embedder/classifier MLP

        self.get_trunk(trunk_architecture)
        self.trunk = nn.DataParallel(self.trunk.to(self.device))
        self.embedder = nn.DataParallel(Network(layer_sizes=[self.MLP_neurons, self.embedding_size], 
                                                neuron_fc=self.MLP_neurons).to(self.device))
        self.classifier = nn.DataParallel(Network(layer_sizes=[self.embedding_size, self.num_classes], 
                                                  neuron_fc=self.MLP_neurons).to(self.device))
        
        # build optimizers
        self.trunk_optimizer = self.get_optimizer(trunk_optim, 
                                                  self.trunk.parameters(), 
                                                  lr=trunk_lr, 
                                                  weight_decay=weight_decay)
        self.embedder_optimizer = self.get_optimizer(embedder_optim, 
                                                     self.embedder.parameters(), 
                                                     lr=embedder_lr, 
                                                     weight_decay=weight_decay)
        self.classifier_optimizer = self.get_optimizer(classifier_optim, 
                                                       self.classifier.parameters(), 
                                                       lr=classifier_lr, 
                                                       weight_decay=weight_decay)

        # build schedulers
        self.trunk_scheduler = ExponentialLR(self.trunk_optimizer, gamma=trunk_decay)
        self.embedder_scheduler = ExponentialLR(self.embedder_optimizer, gamma=embedder_decay)
        self.classifier_scheduler = ExponentialLR(self.classifier_optimizer,  gamma=classifier_decay)

        # build pair based losses and the miner
        self.triplet = losses.TripletMarginLoss(margin=0.2).to(self.device)
        self.multisimilarity = losses.MultiSimilarityLoss(alpha = 2, beta = 50, base = 1).to(self.device)
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        # build proxy anchor loss
        self.proxy_anchor = Proxy_Anchor(nb_classes = num_classes, sz_embed = embedding_size, mrg = 0.2, alpha = 32).to(self.device)
        self.proxy_optimizer = AdamW(self.proxy_anchor.parameters(), lr=trunk_lr*10, weight_decay=1.5E-6)
        self.proxy_scheduler = ExponentialLR(self.proxy_optimizer, gamma=0.8)
        # finally crossentropy loss
        self.crossentropy = torch.nn.CrossEntropyLoss().to(self.device)

        # log some of this information
        self.model_params = {"Trunk_Model":trunk_architecture,
                             "Optimizers":[str(self.trunk_optimizer),
                                           str(self.embedder_optimizer),
                                           str(self.classifier_optimizer)],
                             "Embedder":str(self.embedder),
                             "Embedding_Dimension":str(embedding_size),
                             "Weight_Decay":weight_decay,
                             "Scheduler_Decays":[trunk_decay, embedder_decay, classifier_decay],
                             "Embedding_Size":embedding_size,
                             "Learning_Rates":[trunk_lr, embedder_lr, classifier_lr],
                             "Miner":str(self.miner)}


    def get_optimizer(self, optim, params, lr, weight_decay):

        if optim == "adamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optim == "SGD":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif optim == "RMSprop":
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            return None

    def get_trunk(self, architecture):

        if "efficientnet" in architecture.lower():
            self.trunk = EfficientNet.from_pretrained(architecture, num_classes=self.MLP_neurons)

        elif "resnet" in architecture.lower():
            if "18" in architecture.lower():
                self.trunk = models.resnet18(pretrained=True)
                self.trunk.fc = nn.Linear(512, self.MLP_neurons)

            elif "50" in architecture.lower():
                self.trunk = models.resnext50_32x4d(pretrained=True)
                self.trunk.fc = nn.Linear(2048, self.MLP_neurons)

        elif "mobilenet" in architecture.lower():
            self.trunk = models.mobilenet_v2(pretrained=True)
            self.trunk.classifier[1] = torch.nn.Linear(1280, self.MLP_neurons)


    def get_embeddings_logits(self, dataset, indices, batch_size=128, num_workers=16, return_collisions=False):
        """
        This can be used for inference but is not super appropriate since
        it requires the dataset/indices
        """
        
        # build a temporary dataloader
        temp_sampler = SubsetRandomSampler(indices)
        temp_loader = DataLoader(dataset=dataset,
                                 sampler=temp_sampler,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        tot_embeds = []
        tot_logits = []
        tot_labels = []
        accuracies = []

        if return_collisions:
            # initialize weights to count # of collisions
            class_weights = np.zeros(self.num_classes) + 0.2
            label_count = torch.ones(self.num_classes).cuda()

        # turn all models into eval mode
        self.trunk.eval()
        self.embedder.eval()
        self.classifier.eval()

        n_iter = int(temp_loader.sampler.__len__()/batch_size)
        # turn grad off for evaluation
        with torch.no_grad():
            print("Getting Embeddings")
            with tqdm(total=n_iter) as t:
                for i, data in enumerate(temp_loader):
                    im, labels = data

                    # forward pass for each model
                    fc_out = self.trunk(im)
                    embeds = self.embedder(fc_out)
                    logits = self.classifier(embeds)

                    if return_collisions:
                        preds = knn_sim(embeds, labels, 
                                        k=self.M,
                                        distance_weighted=False,
                                        local_normalization=False, 
                                        num_classes=self.num_classes)
                        weights = impostor_weights(preds, labels, k=self.M, num_classes=self.num_classes)
                        class_weights += weights.cpu().numpy()
                        label_count = label_count.scatter_add(0, labels, torch.ones(len(labels)).cuda())

                    # embeds -> to cpu and then to array
                    accuracies.append(calc_accuracy(logits, labels.to(self.device)))
                    tot_embeds.append(embeds.cpu().numpy())
                    tot_logits.append(logits.cpu().numpy())
                    tot_labels.append(labels.cpu().numpy())

                    t.update()

        # return the np arrays
        tot_embeds = np.concatenate(tot_embeds, axis=0)
        tot_logits = np.concatenate(tot_logits, axis=0)
        tot_labels = np.concatenate(tot_labels, axis=0)

        print("logits shape", tot_logits.shape)
        print("Accuracy is", np.mean(accuracies))

        del temp_loader, temp_sampler

        if return_collisions:
            return  tot_embeds, tot_logits, tot_labels, np.mean(accuracies), class_weights/label_count.cpu().numpy()
        else:
            return tot_embeds, tot_logits, tot_labels, np.mean(accuracies)


    def save_all_logits_embeds(self, path):
        """
        This is usually run at the end, it will save all logits/embeddings of the train
        and validation datasets to disk. Warning: Holdout not currently included.
        """

        tembeds, tlogits, tlabels, _ = self.get_embeddings_logits(self.val_dataset, 
                                                                  self.train_indices, 
                                                                  batch_size=self.batch_size*4,
                                                                  num_workers=self.num_workers)
        vembeds, vlogits, vlabels, _ = self.get_embeddings_logits(self.val_dataset, 
                                                                  self.val_indices, 
                                                                  batch_size=self.batch_size*4,
                                                                  num_workers=self.num_workers)

        np.savez(path, tembeds=tembeds, tlogits=tlogits, tlabels=tlabels,
                       vembeds=vembeds, vlogits=vlogits, vlabels=vlabels)


    def image_inference(self, image):
        # image should be an nparray

        self.trunk.eval()
        self.embedder.eval()
        self.classifier.eval()

        with torch.no_grad():
            fc_out = self.trunk(image.to(self.device))
            embeds = self.embedder(fc_out)
            logits = self.classifier(embeds)

        return embeds, logits


    def save_model(self, path):
        """
        This function is used to save the state dictionaries of
        all three models and their corresponding classifiers to
        the input path provided under the name "models.h5"
        """
        
        print("Saving model to", path)
        torch.save({
                "trunk_state_dict": self.trunk.state_dict(),
                "embedder_state_dict": self.embedder.state_dict(),
                "classifier_state_dict": self.classifier.state_dict(),
                "trunk_optimizer_state_dict": self.trunk_optimizer.state_dict(),
                "embedder_optimizer_state_dict": self.embedder_optimizer.state_dict(),
                "classifier_optimizer_state_dict": self.classifier_optimizer.state_dict(),
                }, path + "/models.h5")


    def load_weights(self, 
                     path, 
                     load_trunk=True, 
                     load_embedder=True, 
                     load_classifier=True, 
                     partial_classifier=False, 
                     load_optimizers=True):
        """
        This function is to continue training or to use a pretrained model,
        it will load a file saved from the save_model() method above at the
        given path. It will also set the pretrained flag to True.
        """

        weights = torch.load(path)

        self.pretrained = True
        loaded = []

        if load_trunk:
            self.trunk.load_state_dict(weights["trunk_state_dict"])
            if load_optimizers:
              self.trunk_optimizer.load_state_dict(weights["trunk_optimizer_state_dict"])
            loaded.append("Trunk")

        if load_embedder:
            self.embedder.load_state_dict(weights["embedder_state_dict"])
            if load_optimizers:
              self.embedder_optimizer.load_state_dict(weights["embedder_optimizer_state_dict"])
            loaded.append("Embedder")

        if load_classifier:
            self.classifier.load_state_dict(weights["classifier_state_dict"])
            if load_optimizers:
              self.classifier_optimizer.load_state_dict(weights["classifier_optimizer_state_dict"])
            loaded.append("Classifier")

        if partial_classifier:
            print("Partial Loading of classifier")
            # load overall network
            self.classifier = nn.DataParallel(Network(layer_sizes=[self.embedding_size, 693], 
                                                      neuron_fc=self.MLP_neurons).to(self.device))
            self.classifier.load_state_dict(weights["classifier_state_dict"])
            # replace last layer with number of classes
            self.classifier.module.classifier[4] = nn.Linear(self.MLP_neurons, self.num_classes).to(self.device)

        print("Loaded pretrained weights for", path, "for", loaded)


    def augmented_embeds(self, image, N):
        # TEMPORARY METHOD WILL BE REMOVED AS IT ISN'T USEFUL RIGHT NOW
        """
        Pass PIL Image and get back N augmented versions of the image,
        as well as the original
        """

        # setup the augmentation, this code should change, pretty ugly!
        transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ColorJitter(brightness=(0.8, 1.3),
                                        contrast=(0.8, 1.2),
                                        saturation=(0.9, 1.2),
                                        hue=(-0.05, 0.05)),
                                transforms.RandomRotation(degrees=(-5, 5),
                                            expand=True),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])

        val_transform = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

        # augment the image N times and return
        self.trunk.eval()
        self.embedder.eval()

        tot_embeds = []
        with torch.no_grad():

            # get original image with val transform
            fc_out = self.trunk(val_transform(image).to(self.device).reshape(1, 3, 224, 224))
            embeds = self.embedder(fc_out)
            tot_embeds.append(embeds.squeeze())

            # now get N augmented images
            for _ in range(N):
                fc_out = self.trunk(transform(image).to(self.device).reshape(1, 3, 224, 224))
                embeds = self.embedder(fc_out)
                tot_embeds.append(embeds.squeeze())

        return torch.stack(tot_embeds)
        

    def setup_data(self,
                   dataset,
                   h5_path=None,
                   batch_size=128,
                   num_workers=16,
                   M=3,
                   train_split=0.8,
                   labels=None,
                   repeat_indices=1,
                   train_labels=None, 
                   load_indices=False, 
                   indices_path=None,
                   max_batches=None,
                   log_save_path="logs"):
        """
        This method is meant to be used prior to training in order
        to get the appropriate dataloaders, datasets, indices etc...
        I'm not sure if the need for this method suggests bad design?

        Inputs:
            path str: the hdf5 dataset  path for training
            batch_size int: the batch size used for training later
            num_workers int: the number of workers to pass to training dataloader
            labels None or array: the array of all labels
            train_labels None or array: the labels to train on, if none will train on all
            load_indices Bool: whether to load train/val/holdout indices, usually used if
                               you are continuing to train a pretrained model. Careful as
                               incorrect loading of indices can result in test set leakage.
            indices_path str: Where the indices you wish to load are located
            log_save_path str: which directory to save training logs to, such as batch_history.csv
        """

        if labels is None and h5_path is not None:
            self.labels = h5py.File(h5_path, "r")["labels"][:]
        else:
            self.labels = labels

        if load_indices:
            arr = np.load(indices_path)
            train_indices, val_indices, holdout_indices = arr["train"], arr["val"], arr["holdout"]
        else:
            if self.pretrained is True:
                warnings.warn("Picking random indices, dangerous if you are continuing training!")
            train_indices, val_indices, holdout_indices = get_train_val_holdout_indices(labels=self.labels,
                                                                                        train_labels=train_labels,
                                                                                        train_split=train_split)
            if self.log_train:
                np.savez(log_save_path + "/data_indices.npz", train=train_indices, val=val_indices, holdout=holdout_indices)

        self.augmentations = data_augmentation(hflip=True,
                                               crop=False,
                                               colorjitter=True,
                                               rotations=False,
                                               affine=False,
                                               imagenet=True)

        trainloader, val_dataset = get_dataloaders(dataset=dataset,
                                                   h5_path=h5_path,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   augmentations=self.augmentations,
                                                   M=M,
                                                   labels=self.labels,
                                                   train_indices=np.repeat(train_indices, repeat_indices),
                                                   max_batches=max_batches)

        self.max_batches = max_batches
        self.dataset = dataset
        self.trainloader = trainloader
        self.val_dataset = val_dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.holdout_indices = holdout_indices
        self.batch_size = batch_size
        self.log_save_path = log_save_path
        self.num_workers = num_workers
        self.M = M
        self.repeat_indices = repeat_indices


    def train(self,
              n_epochs,
              loss_ratios=[1,1,1,3],
              class_weighting=False,
              model_save_path="models",
              model_name="models.h5",
              epoch_train=False,
              epoch_val=True,
              epoch_save=False,
              save_trunk=True,
              save_embedder=True,
              save_classifier=True,
              train_trunk=True):
        """
        This method is used for actually training the model, it is meant
        to be called after the setup_data() method and won't function
        properly without it as it will not have access to some attributes.

        Inputs:
            n_epochs int: the amount of epochs to train for
            loss_ratios array: the ratios to pass to triplet, multisimilarity,
                               proxy anchor and crossentropy in that order
            model_save_path str: where to save models at checkpoints and at end
            log_train Bool: whether to log the train files
        """

        self.model_params["Loss_Ratios"] = loss_ratios

        # Set up the GPU handle to report useage/vram useage
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(self.gpu_id)

        # Set up the logging
        self.batch_history = {"Iteration":[],
                         "Loss":[],
                         "Losses":[],
                         "Accuracy":[],
                         "GPU_useage":[],
                         "GPU_mem":[],
                         "Time":[]}
        self.epoch_history = {"Epoch":[],
                         "Train_Accuracy":[],
                         "Val_Accuracy":[],
                         "Learning_Rates":[],
                         "Time":[]}
        batch_log_path = self.log_save_path + "/batch_history.csv"
        epoch_log_path = self.log_save_path + "/epoch_history.csv"

        if self.log_train is True and os.path.exists(batch_log_path) is False:
            with open(batch_log_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(list(self.batch_history.keys()))
            with open(epoch_log_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(list(self.epoch_history.keys()))

        # This is purely used for model checkpoints to save the best epoch model
        best_val_accuracy = 0

        # setup AMP GradScaler
        scaler = torch.cuda.amp.GradScaler()

        print(f"Starting training with {n_epochs} Epochs.")
        n_iters = np.int(self.trainloader.sampler.__len__() / self.batch_size)
        for epoch in range(n_epochs):
            # set our models to train mode
            if train_trunk:
                self.trunk.train()
            else:
                self.trunk.eval()
            self.embedder.train()
            self.classifier.train()

            # initialize our batch accuracy and loss parameters that are later used
            # to compute a rolling mean.
            batch_acc = 0
            batch_loss = 0
            batch_acc_queue = []
            batch_loss_queue = []

            performance_dict = {"Load_Data":0,
                                "Forward_Pass":0,
                                "Mining":0,
                                "Compute_Loss":0,
                                "Optim_Step":0,
                                "Logging":0,
                                "Total":0}

            # initialize class weights to be uniform distribution if no class_weighting
            if class_weighting:
                # start in a smoothed fashion with 5 collisions each
                class_weights = np.zeros(self.num_classes) + 0.2
            else:
                class_weights = np.zeros(self.num_classes) + 0.2 #/ self.num_classes
            label_count = torch.ones(self.num_classes).cuda()

            with tqdm(total=int(n_iters)) as t:
                start_t = time()
                for i, data in enumerate(self.trainloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # zero the parameter gradients
                    if train_trunk:
                        self.trunk_optimizer.zero_grad()
                        self.trunk.zero_grad()

                    self.embedder_optimizer.zero_grad()
                    self.embedder.zero_grad()
                    self.classifier_optimizer.zero_grad()
                    self.classifier.zero_grad()
                    self.proxy_optimizer.zero_grad()

                    # forward pass
                    with autocast():
                        time_check = time()
                        fc_out = self.trunk(inputs)
                        embeddings = self.embedder(fc_out)
                        logits = self.classifier(embeddings)
                        performance_dict["Forward_Pass"] += time() - time_check

                        # mine interesting pairs
                        time_check = time()
                        if loss_ratios[0] + loss_ratios[1] != 0:
                            hard_pairs = self.miner(embeddings, labels)
                        performance_dict["Mining"] += time() - time_check

                        # compute loss, the conditionals are to speed up compute if a loss
                        # has been switched off.
                        time_check = time()
                        loss = 0
                        curr_losses = []
                        if loss_ratios[0] != 0:
                            triplet_loss_curr = self.triplet(embeddings, labels, hard_pairs)
                            curr_losses.append(triplet_loss_curr.item() * loss_ratios[0])
                            loss += triplet_loss_curr * loss_ratios[0]

                        if loss_ratios[1] != 0:
                            ms_loss_curr = self.multisimilarity(embeddings, labels, hard_pairs)
                            curr_losses.append(ms_loss_curr.item() * loss_ratios[1])
                            loss += ms_loss_curr * loss_ratios[1]

                        if loss_ratios[2] != 0:
                            proxy_loss_curr = self.proxy_anchor(embeddings, labels)
                            curr_losses.append(proxy_loss_curr.item() * loss_ratios[2])
                            loss += proxy_loss_curr * loss_ratios[2]

                        if loss_ratios[3] != 0:
                            cse_loss_curr = self.crossentropy(logits, labels.to(self.device).long())
                            curr_losses.append(cse_loss_curr.item() * loss_ratios[3])
                            loss += cse_loss_curr * loss_ratios[3]

                    scaler.scale(loss).backward()
                    performance_dict["Compute_Loss"] += time() - time_check

                    # now take a step
                    time_check = time()
                    if train_trunk:
                        scaler.step(self.trunk_optimizer)
                    scaler.step(self.embedder_optimizer)
                    scaler.step(self.classifier_optimizer)
                    scaler.step(self.proxy_optimizer)

                    scaler.update()

                    #if class_weighting:
                    # compute batch label weightings
                    k = 1#self.M
                    preds = knn_sim(embeddings, labels,
                                    k=k,
                                    distance_weighted=False,
                                    local_normalization=False,
                                    num_classes=self.num_classes)
                    weights = impostor_weights(preds, labels, k=k, num_classes=self.num_classes)
                    #weights, associated_labels = get_weights(preds, labels)
                    # moving average weight calculation (x[l] = x[l] + (new_data - x[l])/(i+1))
                    class_weights += weights.cpu().numpy()
                    label_count = label_count.scatter_add(0, labels, torch.ones(len(labels)).cuda())
                    #class_weights += (weights.cpu().numpy() - class_weights)/(i+1)
                    #class_weights[associated_labels] += (weights - class_weights[associated_labels])/(i+1)

                    performance_dict["Optim_Step"] += time() - time_check

                    time_check = time()
                    # compute mean using queue datastructure of length 2048//batch_size.
                    batch_acc_queue.append(calc_accuracy(logits, labels.to(self.device)))
                    batch_loss_queue.append(loss.item())
                    if len(batch_acc_queue) >= 2048//self.batch_size:
                        batch_acc_queue.pop(0)
                        batch_loss_queue.pop(0)
                    batch_acc = np.mean(batch_acc_queue)
                    batch_loss = np.mean(batch_loss_queue)

                    res = nvmlDeviceGetUtilizationRates(handle)
                    # log the current batch information
                    if self.log_train:
                        self.batch_history["Iteration"].append(epoch*n_iters+i)
                        self.batch_history["Loss"].append(batch_loss)
                        self.batch_history["Accuracy"].append(batch_acc)
                        self.batch_history["Time"].append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                        self.batch_history["GPU_useage"].append(res.gpu)
                        self.batch_history["GPU_mem"].append(res.memory)

                        # write to CSV file, should change this to dict writer soon
                        with open(batch_log_path, "a") as f:
                            writer = csv.writer(f)
                            writer.writerow([epoch*n_iters+i,
                                             batch_loss,
                                             batch_acc,
                                             res.gpu,
                                             res.memory,
                                             datetime.now().strftime("%d/%m/%Y %H:%M:%S")])

                    # now update our loading bar with new values of batch loss and accuracy
                    t.set_description('Epoch %i' % int(epoch))
                    t.set_postfix(loss=batch_loss, 
                                  acc=batch_acc, 
                                  gpu=res.gpu, 
                                  gpuram=res.memory,
                                  losses= [np.round(i,2) for i in curr_losses])
                    t.update()
                    performance_dict["Logging"] += time() - time_check

            # save class weights after each epoch
            #if class_weighting:
            class_weights /= label_count.cpu().numpy() # normalize by amount of times a certain label has occured
            np.save(f"logs/class_weights_{epoch}.npy", class_weights)

            # build and save performance dictionary, keep in mind Load_Data is inaccurate due to prefetching
            performance_dict["Load_Data"] = np.sum(performance_dict[key] for key in performance_dict)
            performance_dict["Total"] = time() - start_t
            performance_dict["Load_Data"] = performance_dict["Total"] - performance_dict["Load_Data"]
            performance_dict = {key:np.round(performance_dict[key], 2) for key in performance_dict}

            if self.log_train:
                with open(self.log_save_path + "/performance.json", "w") as json_file:
                    json.dump(performance_dict, json_file)
            print(performance_dict)

            if class_weighting is True:
                # get the next dataloader based on class weights
                del self.trainloader
                self.trainloader, self.val_dataset = get_dataloaders(dataset=self.dataset,
                                                           batch_size=self.batch_size,
                                                           num_workers=self.num_workers,
                                                           augmentations=self.augmentations,
                                                           M=self.M,
                                                           labels=self.labels,
                                                           train_indices=np.repeat(self.train_indices, self.repeat_indices),
                                                           class_weights=class_weights,
                                                           max_batches=self.max_batches)

            if epoch_train is True:
                # Train accuracy, embeddings and potential UMAP
                print("Training")
                em, lo, la, train_accuracy, collisions = self.get_embeddings_logits(self.val_dataset,
                                                                                    self.train_indices,
                                                                                    self.batch_size*2,
                                                                                    self.num_workers,
                                                                                    True)
                np.save(f"logs/class_weights_{epoch}_val.npy", collision)

            if epoch_val is True:
                # Validation accuracy, loss, embeddings and potential UMAP
                print("Validation")
                em, lo, la, val_accuracy = self.get_embeddings_logits(self.val_dataset,
                                                                      self.val_indices, 
                                                                      self.batch_size*2, 
                                                                      self.num_workers)
            else:
                # sadly the way its written we have to increment this or it won't
                # save the model since it won't be greater than last epochs.
                val_accuracy = 0.001

            # finally we log the batch metrics
            if self.log_train:
                self.epoch_history["Learning_Rates"].append([self.trunk_scheduler.get_last_lr(),
                                                             self.embedder_scheduler.get_last_lr(),
                                                             self.classifier_scheduler.get_last_lr()])
                self.epoch_history["Epoch"].append(epoch)
                if epoch_train:
                    self.epoch_history["Train_Accuracy"].append(train_accuracy)
                else:
                    train_accuracy = 0
                if epoch_val:
                    self.epoch_history["Val_Accuracy"].append(val_accuracy)

                self.epoch_history["Time"].append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

                # write CSV file
                with open(epoch_log_path, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch,
                                     train_accuracy,
                                     val_accuracy,
                                     [self.trunk_scheduler.get_last_lr(),
                                      self.embedder_scheduler.get_last_lr(),
                                      self.classifier_scheduler.get_last_lr()],
                                      datetime.now().strftime("%d/%m/%Y %H:%M:%S")])

            # check the learning rate schedulers
            if train_trunk:
                self.trunk_scheduler.step()
            self.embedder_scheduler.step()
            self.classifier_scheduler.step()
            self.proxy_scheduler.step()

            # save best model (based on validation accuracy)
            if val_accuracy >= best_val_accuracy or epoch_save is True:
                best_val_accuracy = val_accuracy
                # WARNING!!!! This MIGHT not work if parallel GPUs are used, then would
                # need to use model.module.state_dict() I believe? Not sure!
                save_dict = {}
                if save_trunk:
                    save_dict["trunk_state_dict"] = self.trunk.state_dict()
                    save_dict["trunk_optimizer_state_dict"] = self.trunk_optimizer.state_dict()
                if save_embedder:
                    save_dict["embedder_state_dict"] = self.embedder.state_dict()
                    save_dict["embedder_optimizer_state_dict"] = self.embedder_optimizer.state_dict()
                if save_classifier:
                    save_dict["classifier_state_dict"] = self.classifier.state_dict()
                    save_dict["classifier_optimizer_state_dict"] = self.classifier_optimizer.state_dict()

                torch.save(save_dict, model_save_path + "/" + model_name)

                # save the JSON including model details, this can be improved to take the mean
                self.model_params["final_val_accuracy"] = best_val_accuracy
                self.model_params["Time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                self.model_params = {k:str(self.model_params[k]) for k in self.model_params}
                if self.log_train:
                    with open(self.log_save_path + "/model_dict.json", "w") as json_file:
                        json.dump(self.model_params, json_file)
