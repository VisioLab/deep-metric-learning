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
from torch.optim import Adam, RMSprop, AdamW
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
from utils import calc_accuracy
from losses import *
from hdf5_loader import *
from autoaugment import ImageNetPolicy


class Network(nn.Module):

    def __init__(self, layer_sizes, neuron_fc=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(layer_sizes[0], neuron_fc),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(neuron_fc, layer_sizes[1]),
            #nn.Linear(layer_sizes[0], layer_sizes[1]),
        )

    def forward(self, x):
        return self.classifier(x)


class ThreeStageNetwork():

    def __init__(self,
                 num_classes=101,
                 embedding_size=512,
                 efficient_version="efficientnet-b0",
                 trunk_optim=RMSProp,
                 embedder_optim=RMSProp,
                 classifier_optim=RMSProp,
                 trunk_lr=1e-3,
                 embedder_lr=1e-3, 
                 classifier_lr=1e-3,
                 weight_decay=1.5e-6,
                 trunk_decay=0.95,
                 embedder_decay=0.95,
                 classifier_decay=0.95):

        # build three stage network
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.efficientnet_version = efficient_version
        self.trunk = EfficientNet.from_pretrained(efficientnet_version, num_classes=1024)
        self.model_output_size = self.trunk._fc.in_features
        self.trunk._fc = torch.nn.Identity()
        self.trunk = nn.DataParallel(self.trunk.to(device))
        self.embedder = nn.DataParallel(Network([self.model_output_size, self.embedding_size]).to(device))
        self.classifier = nn.DataParallel(Network([self.embedding_size, self.num_classes]).to(device))

        # build optimizers
        self.trunk_optimizer = trunk_optim(self.trunk.parameters(), lr=trunk_lr, weight_decay=weight_decay)
        self.embedder_optimizer = embedder_optim(self.embedder.parameters(), lr=embedder_lr, weight_decay=weight_decay)
        self.classifier_optimizer = classifier_optim(self.classifier.parameters(), lr=classifier_lr, weight_decay=weight_decay)

        # build schedulers
        self.trunk_scheduler = ExponentialLR(self.trunk_optimizer, gamma=trunk_decay)
        self.embedder_scheduler = ExponentialLR(self.embedder_optimizer, gamma=embedder_decay)
        self.classifier_scheduler = ExponentialLR(self.classifier_optimizer,  gamma=classifier_decay)

        # build pair based losses and the miner
        self.triplet = losses.TripletMarginLoss(margin=2).cuda()
        self.multisimilarity = losses.MultiSimilarityLoss(alpha = 2, beta = 50, base = 1).cuda()
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        # build proxy anchor loss
        self.proxy_anchor = Proxy_Anchor(nb_classes = num_classes, sz_embed = embedding_size, mrg = 0.1, alpha = 32).cuda()
        self.proxy_optimizer = AdamW(self.proxy_anchor.parameters(), lr=trunk_lr*100, weight_decay=1.5E-6)
        self.proxy_scheduler = ExponentialLR(self.proxy_optimizer, gamma=trunk_decay)
        # finally crossentropy loss
        self.crossentropy = torch.nn.CrossEntropyLoss().cuda()

        # log some of this information
        self.model_params = {"Trunk_Model":self.efficientnet_version,
                             "Optimizers":[trunk_optim, embedder_optim, classifier_optim],
                             "Weight_Decay":weight_decay,
                             "Scheduler_Decays":[trunk_decay, embedder_decay, classifier_decay],
                             "Embedding_Size":embedding_size,
                             "Learning_Rates":[trunk_lr, embedder_lr, classifier_lr],
                             "Miner":str(miner)}


    def load_weights(self, path):

        weights = torch.load(path)

        self.trunk.load_state_dict(weights["trunk_state_dict"])
        self.embedder.load_state_dict(weights["embedder_state_dict"])
        self.classifier.load_state_dict(weights["classifier_state_dict"])
        self.trunk_optimizer.load_state_dict(weights["trunk_optimizer_state_dict"])
        self.embedder_optimizer.load_state_dict(weights["embedder_optimizer_state_dict"])
        self.classifier_optimizer.load_state_dict(weights["classifier_optimizer_state_dict"])


    def get_embeddings_logits(self, dataset, indices, batch_size=256):
        """
        This can be used for inference
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

        del temp_loader, curr_sampler

        return tot_embeds, tot_logits, tot_labels, np.mean(accuracies)


    def save_model(self, path):
        
        print("Saving model to", path)
        torch.save({
                "trunk_state_dict": self.trunk.state_dict(),
                "embedder_state_dict": self.embedder.state_dict(),
                "classifier_state_dict": self.classifier.state_dict(),
                "trunk_optimizer_state_dict": self.trunk_optimizer.state_dict(),
                "embedder_optimizer_state_dict": self.embedder_optimizer.state_dict(),
                "classifier_optimizer_state_dict": self.classifier_optimizer.state_dict(),
                }, path + "/models")


    def train(self,
              batch_size,
              epochs,
              loss_ratios,
              model_save_path="models", 
              log_save_path="logs", 
              training_data_path="data"):
        
        # Get dataloaders
        augmentations = data_augmentation(hflip=True,
                                          crop=False,
                                          colorjitter=False,
                                          rotations=False,
                                          affine=False,
                                          imagenet=True)
        trainloader, val_dataset, train_ids, val_ids, holdout_ids, labels = get_dataloaders(h5_path=training_data_path,
                                                                                            batch_size=batch_size,
                                                                                            num_workers=20,
                                                                                            augmentations=augmentations)

        # Set up the GPU handle to report useage/vram useage
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        # Set up the logging
        batch_history = {"Iteration":[],
                         "Loss":[],
                         "Accuracy":[],
                         "GPU_useage":[],
                         "GPU_mem":[],
                         "Time":[]}
        epoch_history = {"Epoch":[],
                         "Train_Accuracy":[],
                         "Val_Accuracy":[],
                         "Learning_Rates":[],
                         "Time":[]}
        batch_log_path = log_save_path + "/batch_history.csv"
        epoch_log_path = log_save_path + "/epoch_history.csv"

        if log_train is True and os.path.exists(batch_log_path) is False:
            with open(batch_log_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(list(batch_history.keys()))
            with open(epoch_log_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(list(epoch_history.keys()))


        # This is purely used for model checkpoints to save the best epoch model
        best_val_accuracy = 0

        print(f"Starting training with {n_epochs} Epochs.")
        n_iters = np.int(trainloader.sampler.__len__() / batch_size)
        for epoch in range(n_epochs):
            # set our models to train mode
            self.trunk.train()
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

            with tqdm(total=int(n_iters)) as t:
                start_t = time()
                for i, data in enumerate(trainloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # zero the parameter gradients
                    self.trunk_optimizer.zero_grad()
                    self.embedder_optimizer.zero_grad()
                    self.classifier_optimizer.zero_grad()
                    self.func3_optimizer.zero_grad()
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
                    hard_pairs = self.miner(embeddings, labels)
                    performance_dict["Mining"] += time() - time_check

                    # compute loss
                    time_check = time()
                    loss = 0
                    loss += self.triplet(embeddings, labels, hard_pairs) * loss_ratios[0]
                    loss += self.multisimilarity(embeddings, labels, hard_pairs) * loss_ratios[1]
                    loss += self.proxy_anchor(embeddings, labels) * loss_ratios[2]
                    loss += self.crossentropy(logits, labels.cuda().long()) * loss_ratios[3]
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
                    if len(batch_acc_queue) >= 2048//batch_size:
                        batch_acc_queue.pop(0)
                        batch_loss_queue.pop(0)
                    batch_acc = np.mean(batch_acc_queue)
                    batch_loss = np.mean(batch_loss_queue)

                    res = nvmlDeviceGetUtilizationRates(handle)
                    # log the current batch information
                    if log_train:
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
                    t.set_postfix(loss=batch_loss, acc=batch_acc, gpu=res.gpu, gpuram=res.memory)
                    t.update()
                    performance_dict["Logging"] += time() - time_check

            # build and save performance dictionary, keep in mind Load_Data is inaccurate due to prefetching
            performance_dict["Load_Data"] = np.sum(performance_dict[key] for key in performance_dict)
            performance_dict["Total"] = time() - start_t
            performance_dict["Load_Data"] = performance_dict["Total"] - performance_dict["Load_Data"]
            performance_dict = {key:np.round(performance_dict[key], 2) for key in performance_dict}

            with open(log_save_path + "/performance.json", "w") as json_file:
                json.dump(performance_dict, json_file)
            print(performance_dict)

            # Train accuracy, embeddings and potential UMAP
            print("Training")
            #get_embeddings(trunk, embedder, classifier, dataset, indices, batch_size=256)
            em, lo, la, train_accuracy = get_embeddings_logits(val_dataset, train_indices)
            
            # Validation accuracy, loss, embeddings and potential UMAP
            print("Validation")
            em, lo, la, val_accuracy = get_embeddings_logits(val_dataset, val_indices)

            # finally we log the batch metrics
            if log_train:
                epoch_history["Learning_Rates"].append([trunk_scheduler.get_last_lr(),
                                                        embedder_scheduler.get_last_lr(),
                                                        classifier_scheduler.get_last_lr()])
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
                    }, model_save_path + "/models")

                # save the JSON including model details, this can be improved to take the mean
                model_params["final_val_accuracy"] = best_val_accuracy
                model_params["Time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                with open(log_save_path + "/model_dict.json", "w") as json_file:
                    json.dump(model_params, json_file)
