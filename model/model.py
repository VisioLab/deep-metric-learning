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

# Import pytorch metric learning stuff
import pytorch_metric_learning
from pytorch_metric_learning import losses, miners, samplers

# import pytorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.optim import Adam, RMSprop, AdamW, SGD
from torchvision import datasets, transforms
import torchvision # to get access to the pretrained models
from efficientnet_pytorch import EfficientNet

# plotting libraries
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from cycler import cycler # some plotting thing

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


class Network(nn.Module):

    def __init__(self, layer_sizes, neuron_fc=2048):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(layer_sizes[0], neuron_fc),
            nn.BatchNorm1d(neuron_fc),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(neuron_fc, layer_sizes[1]),
            #nn.Linear(layer_sizes[0], layer_sizes[1]),
        )

    def forward(self, x):
        return self.classifier(x)


class ThreeStageNetwork():

    def __init__(self,
                 num_classes=101,
                 embedding_size=512,
                 efficientnet_version="efficientnet-b0",
                 trunk_optim=RMSprop,
                 embedder_optim=RMSprop,
                 classifier_optim=RMSprop,
                 trunk_lr=1e-4,
                 embedder_lr=1e-3, 
                 classifier_lr=1e-3,
                 weight_decay=1.5e-6,
                 trunk_decay=0.98,
                 embedder_decay=0.93,
                 classifier_decay=0.93):
        """
        Inputs:
            num_classes int: Number of Classes (for Classifier purely)
            embedding_size int: The size of embedding space output from Embedder
            efficientnet_version str: Which efficientnet model to use for Trunk
            trunk_optim optim: Which optimizer to use, such as AdamW
            embedder_optim optim: Which optimizer to use, such as AdamW
            classifier_optim optim: Which optimizer to use, such as AdamW
            trunk_lr float: The learning rate for the Trunk Optimizer
            embedder_lr float: The learning rate for the Embedder Optimizer
            classifier_lr float: The learning rate for the Classifier Optimizer
            weight_decay float: The weight decay for all 3 optimizers
            trunk_decay float: The multiplier for the Scheduler y_{t+1} <- trunk_decay * y_{t}
            embedder_decay float: The multiplier for the Scheduler y_{t+1} <- embedder_decay * y_{t}
            classifier_decay float: The multiplier for the Scheduler y_{t+1} <- classifier_decay * y_{t}
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained = False # this is used to load the indices for train/val data for now

        # build three stage network
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.efficientnet_version = efficientnet_version
        self.trunk = EfficientNet.from_pretrained(efficientnet_version, num_classes=2048)
        self.model_output_size = self.trunk._fc.in_features
        self.trunk._fc = torch.nn.Identity()
        self.trunk = nn.DataParallel(self.trunk.to(self.device))
        self.embedder = nn.DataParallel(Network([self.model_output_size, self.embedding_size]).to(self.device))
        self.classifier = nn.DataParallel(Network([self.embedding_size, self.num_classes]).to(self.device))

        # build optimizers
        self.trunk_optimizer = trunk_optim(self.trunk.parameters(), lr=trunk_lr, weight_decay=weight_decay)
        self.embedder_optimizer = embedder_optim(self.embedder.parameters(), lr=embedder_lr, weight_decay=weight_decay)
        self.classifier_optimizer = classifier_optim(self.classifier.parameters(), lr=classifier_lr, weight_decay=weight_decay)

        # build schedulers
        self.trunk_scheduler = ExponentialLR(self.trunk_optimizer, gamma=trunk_decay)
        self.embedder_scheduler = ExponentialLR(self.embedder_optimizer, gamma=embedder_decay)
        self.classifier_scheduler = ExponentialLR(self.classifier_optimizer,  gamma=classifier_decay)

        # build pair based losses and the miner
        self.triplet = losses.TripletMarginLoss(margin=5).cuda()
        self.multisimilarity = losses.MultiSimilarityLoss(alpha = 2, beta = 50, base = 1).cuda()
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        # build proxy anchor loss
        self.proxy_anchor = Proxy_Anchor(nb_classes = num_classes, sz_embed = embedding_size, mrg = 0.2, alpha = 32).cuda()
        self.proxy_optimizer = AdamW(self.proxy_anchor.parameters(), lr=trunk_lr*100, weight_decay=1.5E-6)
        self.proxy_scheduler = ExponentialLR(self.proxy_optimizer, gamma=0.8)
        # finally crossentropy loss
        self.crossentropy = torch.nn.CrossEntropyLoss().cuda()

        # log some of this information
        self.model_params = {"Trunk_Model":self.efficientnet_version,
                             "Optimizers":[str(self.trunk_optimizer),
                                           str(self.embedder_optimizer),
                                           str(self.classifier_optimizer)],
                             "Embedder":str(self.embedder),
                             "Weight_Decay":weight_decay,
                             "Scheduler_Decays":[trunk_decay, embedder_decay, classifier_decay],
                             "Embedding_Size":embedding_size,
                             "Learning_Rates":[trunk_lr, embedder_lr, classifier_lr],
                             "Miner":str(self.miner)}


    def get_embeddings_logits(self, dataset, indices, batch_size=256):
        """
        This can be used for inference but is not super appropriate since
        it requires the dataset/indices
        """
        
        # build a temporary dataloader
        temp_sampler = SubsetRandomSampler(indices)
        temp_loader = DataLoader(dataset=dataset,
                                 sampler=temp_sampler,
                                 batch_size=batch_size,
                                 num_workers=16)
        tot_embeds = []
        tot_logits = []
        tot_labels = []
        accuracies = []

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

                    # embeds -> to cpu and then to array
                    accuracies.append(calc_accuracy(logits, labels.cuda()))
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

        return tot_embeds, tot_logits, tot_labels, np.mean(accuracies)


    def save_all_logits_embeds(self, path):

        tembeds, tlogits, tlabels, _ = self.get_embeddings_logits(self.val_dataset, self.train_indices, batch_size=256)
        vembeds, vlogits, vlabels, _ = self.get_embeddings_logits(self.val_dataset, self.val_indices, batch_size=256)

        np.savez(path, tembeds=tembeds, tlogits=tlogits, tlabels=tlabels,
                       vembeds=vembeds, vlogits=vlogits, vlabels=vlabels)


    def image_inference(self, image):
        # image should be an nparray

        self.trunk.eval()
        self.embedder.eval()
        self.classifier.eval()

        with torch.no_grad():
            fc_out = self.trunk(torch.from_numpy(image).cuda())
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


    def load_weights(self, path):
        """
        This function is to continue training or to use a pretrained model,
        it will load a file saved from the save_model() method above at the
        given path. It will also set the pretrained flag to True.
        """

        weights = torch.load(path)

        self.pretrained = True
        self.trunk.load_state_dict(weights["trunk_state_dict"])
        self.embedder.load_state_dict(weights["embedder_state_dict"])
        self.classifier.load_state_dict(weights["classifier_state_dict"])
        self.trunk_optimizer.load_state_dict(weights["trunk_optimizer_state_dict"])
        self.embedder_optimizer.load_state_dict(weights["embedder_optimizer_state_dict"])
        self.classifier_optimizer.load_state_dict(weights["classifier_optimizer_state_dict"])


    def setup_data(self,
                   path,
                   batch_size=128, 
                   num_workers=16, 
                   train_labels=None, 
                   load_indices=False, 
                   indices_path=None,
                   log_save_path="logs"):
        """
        This method is meant to be used prior to training in order
        to get the appropriate dataloaders, datasets, indices etc...
        I'm not sure if the need for this method suggests bad design?

        Inputs:
            path str: the hdf5 dataset  path for training
            batch_size int: the batch size used for training later
            num_workers int: the number of workers to pass to training dataloader
            train_labels None or array: the labels to train on, if none will train on all
            load_indices Bool: whether to load train/val/holdout indices, usually used if
                               you are continuing to train a pretrained model. Careful as
                               incorrect loading of indices can result in test set leakage.
            indices_path str: Where the indices you wish to load are located
            log_save_path str: which directory to save training logs to, such as batch_history.csv
        """

        self.labels = h5py.File(path, "r")["labels"][:]
        
        if load_indices:
            arr = np.load(indices_path)
            train_indices, val_indices, holdout_indices = arr["train"], arr["val"], arr["holdout"]

        else:
            if self.pretrained is True:
                warnings.warn("Picking random indices, dangerous if you are continuing training!")
            train_indices, val_indices, holdout_indices = get_train_val_holdout_indices(labels=self.labels,
                                                                                        train_labels=train_labels,
                                                                                        train_split=0.8)
            np.savez(log_save_path + "/data_indices.npz", train=train_indices, val=val_indices, holdout=holdout_indices)

        augmentations = data_augmentation(hflip=True,
                                          crop=False,
                                          colorjitter=False,
                                          rotations=False,
                                          affine=False,
                                          imagenet=True)

        trainloader, val_dataset = get_dataloaders(h5_path=path,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   augmentations=augmentations,
                                                   labels=self.labels,
                                                   train_indices=train_indices)

        self.trainloader = trainloader
        self.val_dataset = val_dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.holdout_indices = holdout_indices
        self.batch_size = batch_size
        self.log_save_path = log_save_path


    def train(self,
              n_epochs,
              loss_ratios=[1,1,1,3],
              model_save_path="models",
              log_train=True):
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

        self.log_train = log_train

        # Set up the GPU handle to report useage/vram useage
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        # Set up the logging
        batch_history = {"Iteration":[],
                         "Loss":[],
                         "Losses":[],
                         "Accuracy":[],
                         "GPU_useage":[],
                         "GPU_mem":[],
                         "Time":[]}
        epoch_history = {"Epoch":[],
                         "Train_Accuracy":[],
                         "Val_Accuracy":[],
                         "Learning_Rates":[],
                         "Time":[]}
        batch_log_path = self.log_save_path + "/batch_history.csv"
        epoch_log_path = self.log_save_path + "/epoch_history.csv"

        if self.log_train is True and os.path.exists(batch_log_path) is False:
            with open(batch_log_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(list(batch_history.keys()))
            with open(epoch_log_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(list(epoch_history.keys()))

        # This is purely used for model checkpoints to save the best epoch model
        best_val_accuracy = 0

        print(f"Starting training with {n_epochs} Epochs.")
        n_iters = np.int(self.trainloader.sampler.__len__() / self.batch_size)
        for epoch in range(n_epochs):
            # set our models to train mode
            self.trunk.train()
            self.embedder.train()
            self.classifier.train()
            self.proxy_optimizer.train()

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

            with tqdm(total=int(n_iters)) as t:
                start_t = time()
                for i, data in enumerate(self.trainloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # zero the parameter gradients
                    self.trunk_optimizer.zero_grad()
                    self.embedder_optimizer.zero_grad()
                    self.classifier_optimizer.zero_grad()
                    self.proxy_optimizer.zero_grad()
                    self.trunk.zero_grad()
                    self.embedder.zero_grad()
                    self.classifier.zero_grad()

                    # forward pass
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
                    if loss_ratios[0] != 0:
                        triplet_loss_curr = self.triplet(embeddings, labels, hard_pairs)
                        loss += triplet_loss_curr * loss_ratios[0]
                    if loss_ratios[1] != 0:
                        ms_loss_curr = self.multisimilarity(embeddings, labels, hard_pairs)
                        loss += ms_loss_curr * loss_ratios[1]
                    if loss_ratios[2] != 0:
                        proxy_loss_curr = self.proxy_anchor(embeddings, labels)
                        loss += proxy_loss_curr * loss_ratios[2]
                    if loss_ratios[3] != 0:
                        cse_loss_curr = self.crossentropy(logits, labels.cuda().long())
                        loss += cse_loss_curr * loss_ratios[3]
                    loss.backward()
                    performance_dict["Compute_Loss"] += time() - time_check

                    # now take a step
                    time_check = time()
                    self.trunk_optimizer.step()
                    self.embedder_optimizer.step()
                    self.classifier_optimizer.step()
                    self.proxy_optimizer.step()
                    performance_dict["Optim_Step"] += time() - time_check

                    time_check = time()
                    # compute mean using queue datastructure of length 2048//batch_size.
                    batch_acc_queue.append(calc_accuracy(logits, labels.cuda()))
                    batch_loss_queue.append(loss.item())
                    if len(batch_acc_queue) >= 2048//self.batch_size:
                        batch_acc_queue.pop(0)
                        batch_loss_queue.pop(0)
                    batch_acc = np.mean(batch_acc_queue)
                    batch_loss = np.mean(batch_loss_queue)

                    res = nvmlDeviceGetUtilizationRates(handle)
                    # log the current batch information
                    if self.log_train:
                        batch_history["Iteration"].append(epoch*n_iters+i)
                        batch_history["Loss"].append(batch_loss)
                        batch_history["Accuracy"].append(batch_acc)
                        batch_history["Time"].append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                        batch_history["GPU_useage"].append(res.gpu)
                        batch_history["GPU_mem"].append(res.memory)

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
                                  gpuram=res.memory)
                    t.update()
                    performance_dict["Logging"] += time() - time_check

            # build and save performance dictionary, keep in mind Load_Data is inaccurate due to prefetching
            performance_dict["Load_Data"] = np.sum(performance_dict[key] for key in performance_dict)
            performance_dict["Total"] = time() - start_t
            performance_dict["Load_Data"] = performance_dict["Total"] - performance_dict["Load_Data"]
            performance_dict = {key:np.round(performance_dict[key], 2) for key in performance_dict}

            with open(self.log_save_path + "/performance.json", "w") as json_file:
                json.dump(performance_dict, json_file)
            print(performance_dict)

            # Train accuracy, embeddings and potential UMAP
            print("Training")
            em, lo, la, train_accuracy = self.get_embeddings_logits(self.val_dataset, self.train_indices)

            # Validation accuracy, loss, embeddings and potential UMAP
            print("Validation")
            em, lo, la, val_accuracy = self.get_embeddings_logits(self.val_dataset, self.val_indices)

            # finally we log the batch metrics
            if self.log_train:
                epoch_history["Learning_Rates"].append([self.trunk_scheduler.get_last_lr(),
                                                        self.embedder_scheduler.get_last_lr(),
                                                        self.classifier_scheduler.get_last_lr()])
                epoch_history["Epoch"].append(epoch)
                epoch_history["Train_Accuracy"].append(train_accuracy)
                epoch_history["Val_Accuracy"].append(val_accuracy)
                epoch_history["Time"].append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

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
            self.trunk_scheduler.step()
            self.embedder_scheduler.step()
            self.classifier_scheduler.step()
            self.proxy_scheduler.step()

            # save best model (based on validation accuracy)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # WARNING!!!! This MIGHT not work if parallel GPUs are used, then would
                # need to use model.module.state_dict() I believe? Not sure!
                torch.save({
                    "trunk_state_dict": self.trunk.state_dict(),
                    "embedder_state_dict": self.embedder.state_dict(),
                    "classifier_state_dict": self.classifier.state_dict(),
                    "trunk_optimizer_state_dict": self.trunk_optimizer.state_dict(),
                    "embedder_optimizer_state_dict": self.embedder_optimizer.state_dict(),
                    "classifier_optimizer_state_dict": self.classifier_optimizer.state_dict(),
                    }, model_save_path + "/models.h5")

                # save the JSON including model details, this can be improved to take the mean
                self.model_params["final_val_accuracy"] = best_val_accuracy
                self.model_params["Time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                self.model_params = {k:str(self.model_params[k]) for k in self.model_params}
                with open(self.log_save_path + "/model_dict.json", "w") as json_file:
                    json.dump(self.model_params, json_file)
