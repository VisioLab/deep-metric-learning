import torch
import numpy as np

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
            embedding = torch.from_numpy(embedding.astype(np.float32))
        embedding = embedding.to(self.device)

        # now compute L2 distance sum((index - embedding)**2) and find K smallest
        topk = torch.topk(torch.sum(torch.pow(self.index[:self.length].sub(embedding), 2), dim=1), k=K, largest=False, dim=0)

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


def knn_sim(embeddings, labels, k=3, distance_weighted=False, local_normalization=False, num_classes=None):
    """
    Simulates kNN for every embedding. Basically operates under the principle
    that when we compute the pairwise distance we are computing the distance
    from the diagonal index to every other embedding. Therefore we can use
    the diagonal indices as our "samples" and are "running" kNN on the remainder.
    """

    # precompute some required values, also looks cleaner
    batch_size = labels.shape[0]
    if num_classes is None:
        num_classes = labels.max().item() + 1

    # get the K nearest values distance weighted (basically torch.bincount with weighting)
    ans = torch.cdist(embeddings, embeddings) # get pairwise distance matrix to simulate kNN
    torch.diagonal(ans).fill_(float("Inf")) # so that we don't select that value
    dist, nearest = torch.topk(ans, k = k, largest=False) # get k closest to each value
    if distance_weighted:
        x = torch.zeros([batch_size, num_classes]).cuda() # prepare a hacky way to implement bincount for multiD
        x = x.scatter_add(1, labels[nearest], dist.reciprocal()) # compute the bincount with reciprocal distance
    else:
        # if no distance weighting then simply return the bincount, basic kNN rule of local majority
        x = torch.zeros([batch_size, num_classes]).cuda() # prepare a hacky way to implement bincount for multiD
        x = x.scatter_add(1, labels[nearest], torch.ones([batch_size, k]).cuda()) # compute the bincount with reciprocal distance

    # can't locally normalize without distance weighting, doesn't make sense.
    if local_normalization is True and distance_weighted is True:
        # local normalization with smooth coefficient 1E-4
        x2 = torch.zeros([batch_size, num_classes]).cuda() # prepare a hacky way to implement bincount for multiD
        x2 = x2.scatter_add(1, labels[nearest], torch.ones([batch_size, k]).cuda())
        x = (x + 1e-4) / (x2 + 1) # divide local reciprocal distance sums by number of local occurances

    return x


def get_weights(preds, labels, normalize=True):
    report = classification_report(preds.cpu().numpy(), labels.cpu().numpy(), output_dict=True)
    f1_scores = [report[key]["f1-score"] for key in list(report.keys())[:-3]]
    weights = torch.from_numpy(np.array(f1_scores) + 0.05).reciprocal()
    if normalize:
        weights /= torch.sum(weights)
    return weights.numpy(), np.array([int(i) for i in list(report.keys())[:-3]])


def impostor_weights(preds, labels, k, num_classes):

    # dark wizardry to extract Nth column (labels) of each prediction
    index = torch.stack([torch.zeros(len(labels), dtype=torch.int32).cuda(), labels]).T

    # number of positive neighbors per class and impostors
    num_positives = preds.gather(dim=1, 
                                 index=index)[:,1]
    num_impostors = torch.abs(num_positives - k)

    # count number of impostors per label
    x = torch.zeros(num_classes).cuda()
    x = x.scatter_add(0, labels, num_impostors)

    return x
