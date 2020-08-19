from data import FoodData
from model.utils import *
from model.model import ThreeStageNetwork
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import psutil
from io import BytesIO
import copy


class CIFAR100(torch.utils.data.Dataset):
    def __init__(self, h5_path=None, transform=None):
        """
        Inputs:
            h5_path (Str): specifying path of HDF5 file to load
            transform (torch transforms): if None is skipped, otherwise torch
                                          applies transforms
        """
        self.transform = transform
        self.images = copy.deepcopy(images)
        self.labels = copy.deepcopy(labels)

    def __getitem__(self, index):
        """
        Method for pulling images and labels from the initialized HDF5 file
        """
        X = Image.fromarray(self.images[index])
        y = self.labels[index]

        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.labels)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':

    # prepare the experiment
    prepare_experiment()

    # CIFAR100
    t = unpickle("train")
    v = unpickle("test")
    images = np.vstack([t[b"data"], v[b"data"]])
    labels = np.hstack([t[b"fine_labels"], v[b"fine_labels"]])
    images = images.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    # save indices for training
    np.savez("cifar_indices.npz",
             train=np.arange(50_000),
             val=np.arange(50_000, 60_000),
             holdout=[])

    # now make encoded labels
    le = LabelEncoder()
    le.fit(categories)
    labels = le.transform(labels)

    print(len(labels))

    # build the model
    model = ThreeStageNetwork(num_classes=len(np.unique(labels)),
                              trunk_architecture="efficientnet-b0",
                              trunk_optim="adamW",
                              embedder_optim="adamW",
                              classifier_optim="adamW",
                              trunk_lr=1e-4,
                              embedder_lr=1e-3,
                              classifier_lr=1e-3,
                              trunk_decay=0.9,
                              embedder_decay=0.9,
                              classifier_decay=0.9,
                              log_train=True)

    model.load_weights("final_b0.h5", load_classifier=True, load_optimizers=False)
    model.setup_data(dataset=CIFAR100,
                     batch_size=280,
                     load_indices=True,
                     num_workers=8,
                     M=4,
                     labels=labels,
                     indices_path="cifar_indices.npz")

    print(len(model.labels))
    print(len(np.unique(model.labels)))
    print(len(model.train_indices))
    model.train(n_epochs=10,
                loss_ratios=[1,5,1,5],
                class_weighting=False,
                epoch_train=False,
                epoch_val=True)

    try:
        # let's get the embeddings and save those too for some visualization
        model.save_all_logits_embeds("logs/logits_embeds.npz")
    except:
        pass

    # finish experiment and zip up
    experiment_id = zip_files(["models", "logs"],
                              experiment_id="cifar_test")
    upload_to_s3(file_name=f"experiment_{experiment_id}.zip",
                 destination=None,
                 bucket="msc-thesis")
