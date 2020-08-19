from data import FoodData
from model.utils import *
from model.model import ThreeStageNetwork
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import psutil
from io import BytesIO
import copy

class NewData(torch.utils.data.Dataset):
    def __init__(self, h5_path=None, transform=None):
        """
        Inputs:
            h5_path (Str): specifying path of HDF5 file to load
            transform (torch transforms): if None is skipped, otherwise torch
                                          applies transforms
        """
        self.transform = transform
        self.im_paths = copy.deepcopy(image_paths)
        self.labels = copy.deepcopy(labels)

    def __getitem__(self, index):
        """
        Method for pulling images and labels from the initialized HDF5 file
        """
        X = Image.open(self.im_paths[index]).convert("RGB")
        y = self.labels[index]

        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':

    # prepare the experiment
    prepare_experiment()


    path = "UECFOOD256/"
    directories = os.listdir(path)

    image_paths = []
    categories = []

    # this checks for broken images by trying to open them first
    for directory in tqdm(directories):
        ims = os.listdir(path + directory)
        for im in ims:
            if im.split(".")[-1] in ["jpg", "png", "jpeg"]:
                try:
                    im2 = Image.open(path + directory + "/" + im)
                    image_paths.append(path + directory + "/" + im)
                    categories.append(directory)
                except:
                    pass

    # now make encoded labels
    le = LabelEncoder()
    le.fit(categories)
    labels = le.transform(categories)

    print(len(labels))

    # build the model
    model = ThreeStageNetwork(num_classes=len(np.unique(labels)),
                              trunk_architecture="efficientnet-b0",
                              trunk_optim="adamW",
                              embedder_optim="adamW",
                              classifier_optim="adamW",
                              trunk_lr=1e-4,
                              embedder_lr=3e-3,
                              classifier_lr=3e-3,
                              weight_decay=0.1,
                              trunk_decay=0.95,
                              embedder_decay=0.95,
                              classifier_decay=0.95,
                              log_train=True)

    model.load_weights("models/models.h5", load_classifier=False, load_optimizers=False)
    model.setup_data(dataset=NewData,
                     batch_size=256,
                     load_indices=True,
                     num_workers=8,
                     M=8,
                     labels=labels,
                     indices_path="logs/data_indices.npz",
                     train_split=0.90,
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
                              experiment_id="efficientnet_food256")
    upload_to_s3(file_name=f"experiment_{experiment_id}.zip",
                 destination=None,
                 bucket="msc-thesis")
