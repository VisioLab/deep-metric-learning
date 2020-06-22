from utils import *
from model import ThreeStageNetwork

if __name__ == '__main__':

    # prepare the experiment
    prepare_experiment()
    training_data_path = "data/data_224_v1.hdf5"
    if os.path.exists(training_data_path) is False:
        download_from_s3(file_name="data_224_v1.hdf5",
                         destination="data",
                         bucket="msc-thesis")

    # build the model
    model = ThreeStageNetwork()
    model.train(batch_size=128,
                epochs=10,
                loss_ratios=[1,1,1,3])

    # finish experiment and zip up
    experiment_id = zip_files([args.model_dir, args.log_dir],
                              experiment_id=21)
    upload_to_s3(file_name=f"experiment_{experiment_id}.zip",
                 destination=None,
                 bucket="msc-thesis")