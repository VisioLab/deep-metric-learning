import numpy as np
import torch
from knn import TorchKNN, torch_get_top_k

import ray
from ray.tune import Trainable, run
from ray.tune.schedulers import PopulationBasedTraining


def train_val(labels, N):

    train = []
    for i in np.unique(labels):
        train.append(np.random.choice(np.where(labels == i)[0], 
                                          size=N, 
                                          replace=False))
    train = np.concatenate(train, axis = 0)
    val = np.setdiff1d(np.arange(len(labels)),
                       train)

    return train, val


class PBTkNN(Trainable):


    def setup(self, config):

        # initialize value
        self.k = config["k"]

        # load the embeddings
        a = np.load("/dev/shm/visiolab_embeds.npz")
        self.embeds = np.array(a["embeds"], dtype=np.float32)
        self.labels = np.array(a["labels"], dtype=np.int32)

        self.embeds = torch.from_numpy(self.embeds).cpu()
        self.labels = torch.from_numpy(self.labels).cpu()

        # get train and val indices for 5-Shot learning
        self.train, self.val = train_val(self.labels, 5)

        self.smooth_coef = 1e-5
        # get a train test split, and let's start with 1 image per class!
        # instantiate our KNN and allocate the number of images we have to VRAM
        self.knn = TorchKNN(n_classes=torch.unique(self.labels).size()[0], 
                            dimension=512, 
                            cells_allocate=1000, 
                            verbose=2)

        # add our onboarding data
        self.knn.add(self.embeds[self.train], self.labels[self.train])

        # randomly shuffle our remaining image data
        self.new_images = np.random.choice(np.arange(len(self.labels[self.val])), 
                                           size=len(self.labels[self.val]), 
                                           replace=False)

        # first image
        self.id = 0


    def step(self):

        # now we loop over the remaining images
        correct = 0

        # loop over next 1000 images
        for i in self.new_images[self.id:self.id+1_000]:

            # find the K nearest neighbours where K is determined automatically
            nearest, distance = self.knn.search(embedding=self.embeds[self.val[i]],
                                                K=self.k,
                                                return_distance=True)
            bins = torch.bincount(nearest, weights=distance.reciprocal(), minlength=self.knn.n_classes) * 1e5
            # local normalization
            bins = (bins + self.smooth_coef) / (torch.bincount(nearest, minlength=self.knn.n_classes) + 1)

            # our prediction
            predicted = torch.argmax(bins)
            
            # get the most often occuring value
            if predicted == self.labels[self.val[i]]:
                correct += 1

        # now add 1 image to index
        self.knn.add(embeddings=self.embeds[self.val[self.id]], 
                     labels=self.labels[self.val[self.id]])
        self.id += 1 # so that during the next step() we get the next image

        self.accuracy = correct / 1_000 # compute this steps accuracy

        return {
            "mean_accuracy": self.accuracy,
            "cur_k": self.k
        }

    def save_checkpoint(self, checkpoint_dir):
        return {
            "accuracy": self.accuracy,
            "k": self.k
        }

    def load_checkpoint(self, checkpoint):
        self.accuracy = checkpoint["accuracy"]

    def reset_config(self, new_config):
        self.k = new_config["k"]
        self.config = new_config
        return True


if __name__ == "__main__":

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            # distribution for resampling
            "k": list(range(1, 101)),
        })

    run(
        PBTkNN,
        name="pbt_knn3",
        scheduler=pbt,
        reuse_actors=True,
        checkpoint_freq=20,
        verbose=False,
        stop={
            "training_iteration": 300,
        },
        num_samples=48,
        config={
            "k": 3,
        })
