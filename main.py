from model.utils import *
from model.model import ThreeStageNetwork

if __name__ == '__main__':


    data_224 = "data_224_v1"
    data_128 = "data_128_v2"

    # prepare the experiment
    prepare_experiment()
    training_data_path = f"data/data/{data_224}.hdf5"
    if os.path.exists(training_data_path) is False:
        download_from_s3(file_name=f"{data_224}.hdf5",
                         destination="data/data",
                         bucket="msc-thesis")

    # build the model
    model = ThreeStageNetwork()
    model.load_weights("models/models.h5")
    model.setup_data(path=training_data_path,
                     batch_size=128,
                     load_indices=True,
                     indices_path="logs/data_indices.npz")
    model.train(n_epochs=10,
                loss_ratios=[1,1,1,3])

    # let's get the embeddings and save those too for some visualization
    model.save_all_logits_embeds("logs/logits_embeds.npz")

    # finish experiment and zip up
    experiment_id = zip_files(["models", "logs"],
                              experiment_id=21)
    upload_to_s3(file_name=f"experiment_{experiment_id}.zip",
                 destination=None,
                 bucket="msc-thesis")
