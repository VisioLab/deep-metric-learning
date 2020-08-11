import numpy as np
from metric_learn import LMNN
import os
import torch
import numpy as np
from multiprocessing import Pool
from model.utils import *
from model.model import ThreeStageNetwork
from sklearn.preprocessing import LabelEncoder
import os
from tqdm.auto import tqdm


class NewData(torch.utils.data.Dataset):
    def __init__(self, h5_path=None, transform=None):
        """
        Inputs:
            h5_path (Str): specifying path of HDF5 file to load
            transform (torch transforms): if None is skipped, otherwise torch
                                          applies transforms
        """
        self.transform = transform

    def __getitem__(self, index):
        """
        Method for pulling images and labels from the initialized HDF5 file
        """
        X = Image.open(image_paths[index])
        y = labels[index]

        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(labels)


class TorchKNN():

    def __init__(self, n_classes, dimension=512, cells_allocate=5_000, verbose=0):
        """
        This instantiates a TorchKNN object and allocates the index/labels
        arrays for future use. It also determines the device if cuda is available.

        Inputs:
            n_classes - int for number of classes
            dimension - int size of our embedding space
            cells_allocate - initial and subsequent allocation of cells
            verbose - int (0, 1 or 2) to get some simple print messages
        """

        # build our initial index
        self.n_classes = n_classes
        self.dimension = dimension
        self.cells_allocate = cells_allocate
        self.verbose = verbose
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # pre-allocate some cells for more efficient addition of future data
        if self.verbose >= 1:
            print(f"Pre-allocating {cells_allocate} Cells for {np.round(cells_allocate*(dimension+1)*4/1E6,2)}MB")
        self.index = torch.zeros([cells_allocate, self.dimension], dtype=torch.float32).to(self.device)
        self.labels = torch.zeros([cells_allocate], dtype=torch.int32).to(self.device)

        # here we will count the number of items per class to optimize our K
        self.categories_count = torch.zeros([n_classes], dtype=torch.int32).to(self.device)
        self.allocated = self.labels.size()[0] # size of the entire array
        self.length = 0 # number of entries we have added

    def add(self, embeddings, labels):
        """
        This method takes an embedding and label pair and adds it to our
        built up index. This essentially builds up our inventory for future
        searches and comparisons.

        This method will first check whether we have hit our preallocated limit
        in which case it will extend by allocating more space in our index.
        Then it adds the embeddings/labels (and coerces to torch tensor).
        Finally we compute the number of images per class to help us quickly
        compute optimal K value.

        Inputs:
            embeddings - preferably torch tensor but can be nparray
            labels - preferably torch tensor but can be nparray
        """

        # check the data types if the embeddings aren't of Tensor type
        if isinstance(embeddings, torch.Tensor) is False:
            # check if single label (int) and convert
            if isinstance(labels, int):
                labels = np.array([labels])
                embeddings = np.expand_dims(embeddings, axis=0)

            elif len(labels.shape) == 0:
                labels = np.expand_dims(labels, axis=0)
                embeddings = np.expand_dims(embeddings, axis=0)

            # amount of data we are adding to our index
            add_length = labels.shape[0]
            
            # convert to torch
            embeddings = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
            labels = torch.from_numpy(labels.astype(np.int32)).to(self.device)
        
        if len(labels.size()) == 0:
            labels = labels.unsqueeze(0)
        # get the size of the data we are adding
        add_length = labels.size()[0]
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)

        # if the data we are adding is bigger than our allocated array
        # then we extend it by bytes_allocate/2 (our data type is 16bit -> 2byte)
        if self.length + add_length >= self.allocated + 1:
            self.index = torch.cat([self.index,
                                    torch.zeros([self.cells_allocate,
                                                 self.dimension], 
                                                 dtype=torch.float32).to(self.device)],
                                   dim=0)
            self.labels = torch.cat([self.labels,
                                     torch.zeros([self.cells_allocate],
                                                 dtype=torch.int32).to(self.device)],
                                     dim=0)
            if self.verbose > 1:
                print(f"Extending Index by {self.cells_allocate} Cells")
        
        # assign the new values
        self.labels[self.length:self.length+add_length] = labels
        self.index[self.length:self.length+add_length] = embeddings

        # recompute optimal K
        self.length += add_length
        self.allocated = self.labels.size()[0]
        self.categories_count += torch.bincount(labels, minlength=self.n_classes)
        self.optimal_K = self.get_optimal_k()

    def search(self, embedding, K="auto", return_distance=False):
        """
        This method takes in a new embedding vector and searches for the K nearest
        neighbours within our built up index.

        Inputs:
            embedding - torch (or np.array) of an embedding to KNN search for
            K - int or "auto" specifying number of neighbours. "auto" will use
                the optimal_K computed from get_optimal_k each time .add is called
            return_distance - Bool on whether to return array of distances associated
                              with the K nearest neighbours
        Outputs:
            labels associated with the indices
            if return_distance is True then distances are also returned
        """
        if K == "auto":
            K = self.optimal_K
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding.astype(np.float16))
        embedding = embedding.to(self.device)

        # now compute L2 distance sum((index - embedding)**2) and find K smallest
        topk = torch.topk(torch.sum(torch.pow(self.index[:self.length].sub(embedding), 2), dim=1), k=K, largest=False, dim=0)

        # return the indices of the smallest items, distance can also be returned
        if return_distance:
            return self.labels[topk.indices], topk.values
        else:
            return self.labels[topk.indices]

    def get_optimal_k(self):
        """
        This can be massively improved, right now it looks at the smallest
        number of items per class and chooses that divided by 2, with a max
        value of 25.

        Outputs:
            int - K (the number of neighbours)
        """

        # get the least occuring category
        min_occurance = torch.min(self.categories_count)

        if min_occurance < 50:
            if min_occurance == 1:
                return 3
            else:
                return min_occurance//2 + 2
        else:
            return 25


def torch_get_top_k(bins, k=3):
    """
    Useful for top k accuracy
    """

    if len(bins) < k:
        k = len(bins)
    return torch.topk(bins, k=k).indices


def run_simulation(train_embeds, train_labels, test_embeds, test_labels):

    # get a train test split, and let's start with 1 image per class!
    # instantiate our KNN and allocate the number of images we have to VRAM
    a = TorchKNN(n_classes=torch.unique(train_labels).size()[0], 
                 dimension=512, 
                 cells_allocate=5000, 
                 verbose=2)

    smooth_coef = 1e-5

    # add our onboarding data
    a.add(train_embeds, train_labels)

    # randomly shuffle our remaining image data
    new_images = np.random.choice(np.arange(len(test_labels)), size=len(test_labels), replace=False)

    # now we loop over the remaining images
    correct = 0
    top_3 = 0

    for i in new_images[:15000]:

        # find the K nearest neighbours where K is determined automatically
        nearest, distance = a.search(embedding=test_embeds[i],
                                     K="auto",
                                     return_distance=True)
        bins = torch.bincount(nearest, weights=distance.reciprocal(), minlength=a.n_classes) * 1e5
        # local normalization
        bins = (bins + smooth_coef) / (torch.bincount(nearest, minlength=a.n_classes) + 1)

        # our prediction
        predicted = torch.argmax(bins)

        # get the most often occuring value
        if predicted == test_labels[i]:
            correct += 1
        # also get the top 3 accuracy
        # better top3 test
        nearest, distance = a.search(embedding=test_embeds[i],
                        K=25,
                        return_distance=True)
        bins = torch.bincount(nearest, weights=distance.reciprocal(), minlength=a.n_classes) * 1e5
        bins = (bins + smooth_coef) / (torch.bincount(nearest, minlength=a.n_classes) + 1)

        if test_labels[i] in torch_get_top_k(bins, k=3):
            top_3 += 1

    return correct/len(new_images[:15000]), top_3/len(new_images[:15000])


def get_images(N):
    path = "example/cropped/"
    directories = os.listdir(path)
    dir_len = [len(os.listdir(path + directory)) for directory in directories]
    # get only directories with more than 50 images
    clean_directories = []
    for i in range(len(directories)):
        if dir_len[i] > 85:
            clean_directories.append(directories[i])

    print(len(clean_directories))

    if N > len(clean_directories):
        N = len(clean_directories)
    # select N directories
    directories = np.random.choice(clean_directories, 
                                size=N, 
                                replace=False)

    image_paths = []
    categories = []

    for directory in directories:
        ims = os.listdir(path + directory)
        for im in ims:
            image_paths.append(path + directory + "/" + im)
            categories.append(directory)

    # now make encoded labels
    le = LabelEncoder()
    le.fit(categories)
    labels = le.transform(categories)

    return labels, image_paths


def get_repeats(n):

    if n == 1:
        return 50
    elif n == 2:
        return 25
    elif n == 3:
        return 20
    elif n == 4:
        return 15
    elif n == 5:
        return 10
    elif n == 10:
        return 5
    else:
        return 1


def get_k(labels):

    min_lab = np.min(np.bincount(labels))

    if min_lab < 5:
        return min_lab
    else:
        return 5


def make_npz(i, N, classes, labels):
    
    classes = np.array(classes)
    labels = np.array(labels)
    
    train = []
    validate = []
    for c in classes:
        t_indices = np.random.choice(np.where(labels == c)[0], size=N, replace=False)
        v_indices = np.setdiff1d(np.where(labels == c)[0],
                                 t_indices)
        train.append(t_indices)
        validate.append(v_indices)
    train = np.concatenate(train, axis=0)
    validate = np.concatenate(validate, axis=0)

    np.savez(f"example/{N}_class_{i}.npz",
             train=train,
             val=validate,
             holdout=0)


if __name__ == '__main__':

    # get the classes/label array/path array
    labels, image_paths = get_images(100)

    # setup the unique labels we will train/validate on
    make_npz(1, 1, np.unique(labels), labels)
    make_npz(1, 2, np.unique(labels), labels)
    make_npz(1, 3, np.unique(labels), labels)
    make_npz(1, 4, np.unique(labels), labels)
    make_npz(1, 5, np.unique(labels), labels)
    make_npz(1, 10, np.unique(labels), labels)
    make_npz(1, 50, np.unique(labels), labels)

    for n in [1, 2, 3, 4, 5, 10]:

        # setup the neural network architecture
        model = ThreeStageNetwork(num_classes=100,
                              efficientnet_version="efficientnet-b0",
                              trunk_optim="SGD",
                              embedder_optim="SGD",
                              classifier_optim="SGD",
                              trunk_lr=1e-4,
                              embedder_lr=1e-3,
                              classifier_lr=1e-3,
                              trunk_decay=0.8,
                              embedder_decay=0.8,
                              classifier_decay=0.9,
                              log_train=False)

        # Now we load the pretrained weights, of course don't load the classifier
        model.load_weights("example/final_b0.h5", load_classifier=False, load_optimizers=False)

        repeats = get_repeats(n)

        # We pass our data class, choose a batch size of 64 and load the indices
        # from our indices path. We also pass the associated label array.
        # We choose M (images per class per epoch) to be 4.
        model.setup_data(dataset=NewData,
                         batch_size=128,
                         load_indices=True,
                         num_workers=16,
                         M=4,
                         repeat_indices=1,
                         labels = labels,
                         indices_path=f"example/{n}_class_1.npz")

        # let's get the embeddings for our training/validation data prior to training
        tembeds, _, tlabels, tacc = model.get_embeddings_logits(model.val_dataset, 
                                                                model.train_indices, 
                                                                batch_size=256,
                                                                num_workers=16)

        vembeds, _, vlabels, vacc = model.get_embeddings_logits(model.val_dataset, 
                                                                model.val_indices, 
                                                                batch_size=256,
                                                                num_workers=16)

        tembeds = torch.from_numpy(tembeds).cuda()
        tlabels = torch.from_numpy(tlabels).cuda()
        vembeds = torch.from_numpy(vembeds).cuda()
        vlabels = torch.from_numpy(vlabels).cuda()

        # run LMNN
        if n > 1:
            lmnn = LMNN(k=get_k(tlabels.cpu().numpy()), learn_rate=1e-4, verbose=True, max_iter=5000)
            lmnn.fit(tembeds.cpu().numpy(), tlabels.cpu().numpy())
            W_cuda = torch.from_numpy(lmnn.components_.T).cuda().float()

        #top1 knn
        top1_knn_before, top3_knn_before = run_simulation(tembeds, tlabels, vembeds, vlabels)

        if n > 1:
            # transform into LMNN found space
            tembeds = torch.matmul(tembeds, W_cuda)
            vembeds = torch.matmul(vembeds, W_cuda)
            # top1 lmnn
            top1_lmnn_before, top3_lmnn_before = run_simulation(tembeds, tlabels, vembeds, vlabels)
        else:
            top1_lmnn_before, top3_lmnn_before = 0, 0

        # We pass our data class, choose a batch size of 64 and load the indices
        # from our indices path. We also pass the associated label array.
        # We choose M (images per class per epoch) to be 4.
        model.setup_data(dataset=NewData,
                         batch_size=128,
                         load_indices=True,
                         num_workers=16,
                         M=4,
                         repeat_indices=repeats,
                         labels = labels,
                         indices_path=f"example/{n}_class_1.npz")

        # now train the neural network
        model.train(n_epochs=25,
                    loss_ratios=[1,10,1,5],
                    model_save_path="example",
                    model_name="finetuned_0.1.h5",
                    epoch_val=False)

        # get embeddings again after fine tuning has finished
        tembeds, _, tlabels, tacc = model.get_embeddings_logits(model.val_dataset, 
                                                                model.train_indices, 
                                                                batch_size=256,
                                                                num_workers=16)

        vembeds, _, vlabels, vacc = model.get_embeddings_logits(model.val_dataset, 
                                                                model.val_indices, 
                                                                batch_size=256,
                                                                num_workers=16)

        tembeds = torch.from_numpy(tembeds).cuda()
        tlabels = torch.from_numpy(tlabels).cuda()
        vembeds = torch.from_numpy(vembeds).cuda()
        vlabels = torch.from_numpy(vlabels).cuda()

        # run LMNN
        if n > 1:
            lmnn = LMNN(k=get_k(tlabels.cpu().numpy()), learn_rate=1e-4, verbose=True, max_iter=5000)
            lmnn.fit(tembeds.cpu().numpy(), tlabels.cpu().numpy())
            W_cuda = torch.from_numpy(lmnn.components_.T).cuda().float()

        #top1 knn
        top1_knn_after, top3_knn_after = run_simulation(tembeds, tlabels, vembeds, vlabels)
        
        if n > 1:
            # transform into LMNN found space
            tembeds = torch.matmul(tembeds, W_cuda)
            vembeds = torch.matmul(vembeds, W_cuda)

            # top1 lmnn
            top1_lmnn_after, top3_lmnn_after = run_simulation(tembeds, tlabels, vembeds, vlabels)
        else:
            top1_lmnn_after, top3_lmnn_after = 0, 0

        print(top1_knn_before, top3_knn_before,
              top1_lmnn_before, top3_lmnn_before,
              top1_knn_after, top3_knn_after,
              top1_lmnn_after, top3_lmnn_after,
              vacc)

        # save np array
        np.save(f"n_{n}_results.npy",
                np.array([top1_knn_before, top3_knn_before,
                          top1_lmnn_before, top3_lmnn_before,
                          top1_knn_after, top3_knn_after,
                          top1_lmnn_after, top3_lmnn_after,
                          vacc]))
