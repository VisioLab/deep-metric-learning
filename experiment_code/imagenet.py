from data import FoodData
from model.utils import *
from model.model import ThreeStageNetwork
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import psutil
from io import BytesIO
import copy


class ImageNet(torch.utils.data.Dataset):
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

    # load all 10 training data batches
    train_files = [np.load(f"Imagenet32_train_npz/train_data_batch_{i}.npz", allow_pickle=True) for i in range(1, 11)]
    train_data = []
    train_labels = []
    for file in train_files:
        train_data.append(file["data"])
        train_labels.append(file["labels"])
    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)

    # # now choose 500 random images per label
    # indices = []
    # for i in np.unique(train_labels):
    #     indices.append(np.random.choice(np.where(train_labels == i)[0],
    #                                     size=500))
    # indices = np.hstack(indices)
    # np.save("imagenet_indices", indices)

    # train_data = train_data[indices]
    # train_labels = train_labels[indices]

    # choose 10 val images per image
    v = np.load("val_data.npz", allow_pickle=True)
    vdat =  v["data"]
    vlab = np.array(v["labels"])
    indices = []
    for i in np.unique(vlab):
        indices.append(np.random.choice(np.where(vlab == i)[0],
                                        size=10))
    indices = np.hstack(indices)

    vdat = vdat[indices]
    vlab = vlab[indices]

    # stack images and reshape
    images = np.vstack([train_data, vdat])
    labels = np.hstack([train_labels, vlab])
    images = images.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    # save indices for training
    np.savez("imagenet_indices.npz",
             train=np.arange(len(train_labels)),
             val=np.arange(len(train_labels), len(labels)),
             holdout=[])


    categories = labels.copy()
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
                              embedder_lr=2e-3,
                              classifier_lr=2e-3,
                              weight_decay=0.1,
                              trunk_decay=0.97,
                              embedder_decay=0.97,
                              classifier_decay=0.97,
                              log_train=True)

    model.load_weights("models3.h5", load_classifier=False, load_optimizers=False)
    model.setup_data(dataset=ImageNet,
                     batch_size=32,
                     load_indices=True,
                     num_workers=32,
                     M=4,
                     labels=labels,
                     indices_path="imagenet_indices.npz",
                     max_batches=200)

    print(len(model.labels))
    print(len(np.unique(model.labels)))
    print(len(model.train_indices))
    model.train(n_epochs=100,
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
                              experiment_id="imagenet_exp1")
    upload_to_s3(file_name=f"experiment_{experiment_id}.zip",
                 destination=None,
                 bucket="msc-thesis")
