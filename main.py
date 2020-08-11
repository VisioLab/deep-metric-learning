from data import FoodData
from model.utils import *
from model.model import ThreeStageNetwork
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import psutil
from io import BytesIO

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
        #X = Image.open(image_paths[index]).convert('RGB')
        if index in cached_indices:
            X = Image.open(images[index]).convert('RGB')
        else:
            X = Image.open(image_paths[index]).convert('RGB')
        y = labels[index]

        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(labels)


if __name__ == '__main__':


    # data_224 = "data_224_v1"
    # data_128 = "data_128_v2"

    # # prepare the experiment
    prepare_experiment()
    # training_data_path = f"data/data/{data_224}.hdf5"
    # if os.path.exists(training_data_path) is False:
    #     download_from_s3(file_name=f"{data_224}.hdf5",
    #                      destination="data/data",
    #                      bucket="msc-thesis")


    path = "larger_data/"
    directories = os.listdir(path)
    #directories = np.random.choice(directories, size=len(directories), replace=False)

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

    # load entire dataset into memory cause YEEEEEET
    print("Cache 40gb of images")
    #images = []
    images = {}
    cached_indices = set()
    for i in tqdm(range(len(image_paths))):
        if i % 5_000 == 0:
            mem_percentage = psutil.virtual_memory().percent
        if mem_percentage < 90:
            with open(image_paths[i], 'rb') as f:
                #images.append(BytesIO(f.read()))
                images[i] = BytesIO(f.read())
            #images.append(Image.open(image_paths[i]).convert('RGB'))
            cached_indices.add(i)
        else:
            break

    # build the model
    model = ThreeStageNetwork(num_classes=len(np.unique(labels)),
                              trunk_architecture="efficientnet-b0",
                              trunk_optim="adamW",
                              embedder_optim="adamW",
                              classifier_optim="adamW",
                              trunk_lr=1e-4,
                              embedder_lr=1e-3,
                              classifier_lr=1e-3,
                              trunk_decay=0.8,
                              embedder_decay=0.8,
                              classifier_decay=0.8,
                              log_train=True)

    #model.load_weights("models/models.h5", load_classifier=False, load_optimizers=False)
    model.setup_data(dataset=NewData,
                     batch_size=282,
                     load_indices=False,
                     num_workers=8,
                     M=3,
                     labels = labels,
                     train_split=0.95)

    print(len(model.labels))
    print(len(np.unique(model.labels)))
    print(len(model.train_indices))
    model.train(n_epochs=10,
                loss_ratios=[1,5,1,5],
                epoch_train=False,
                epoch_val=True)

    # let's get the embeddings and save those too for some visualization
    model.save_all_logits_embeds("logs/logits_embeds.npz")

    # finish experiment and zip up
    experiment_id = zip_files(["models", "logs"],
                              experiment_id="test_knn_classifier")
    upload_to_s3(file_name=f"experiment_{experiment_id}.zip",
                 destination=None,
                 bucket="msc-thesis")
