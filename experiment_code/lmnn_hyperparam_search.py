from mlpack import lmnn
import numpy as np
from time import time
import torch
from knn import TorchKNN, torch_get_top_k
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from tqdm.auto import tqdm
import ray
import ray.tune as tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
import os


def knn_sim(embeds, labels, train, val):

    smooth_coef = 1e-5
    # get a train test split, and let's start with 1 image per class!
    # instantiate our KNN and allocate the number of images we have to VRAM
    a = TorchKNN(n_classes=torch.unique(labels).size()[0], 
                 dimension=512, 
                 cells_allocate=1000, 
                 verbose=2)

    # add our onboarding data
    a.add(embeds[train], labels[train])

    # randomly shuffle our remaining image data
    new_images = np.random.choice(np.arange(len(labels[val])), size=len(labels[val]), replace=False)

    # now we loop over the remaining images
    correct = 0
    correct_list = []
    top_3 = 0
    top_3_list = []

    for i in new_images[:5000]:

        # find the K nearest neighbours where K is determined automatically
        nearest, distance = a.search(embedding=embeds[val[i]],
                        K="auto",
                        return_distance=True)
        bins = torch.bincount(nearest, weights=distance.reciprocal(), minlength=a.n_classes) * 1e5
        # local normalization
        bins = (bins + smooth_coef) / (torch.bincount(nearest, minlength=a.n_classes) + 1)

        # our prediction
        predicted = torch.argmax(bins)
        
        # get the most often occuring value
        if predicted == labels[val[i]]:
            correct += 1
            correct_list.append(1)
        else:
            correct_list.append(0)
        # also get the top 3 accuracy
        
        # better top3 test
        nearest, distance = a.search(embedding=embeds[val[i]],
                        K=25,
                        return_distance=True)
        bins = torch.bincount(nearest, weights=distance.reciprocal(), minlength=a.n_classes) * 1e5
        bins = (bins + smooth_coef) / (torch.bincount(nearest, minlength=a.n_classes) + 1)

        if labels[val[i]] in torch_get_top_k(bins, k=3):
            top_3 += 1
            top_3_list.append(1)
        else:
            top_3_list.append(0)

    print("Accuracy is", correct/len(new_images[:5000]))
    print("Top 3 Accuracy is", top_3/len(new_images[:5000]))

    return correct/len(new_images[:5000]), top_3/len(new_images[:5000])


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


def full_sim(args):

    #LR, K, optimizer, freq = args["LR"], args["K"], args["optimizer"], args["freq"]
    K, max_iter, freq, reg, tol = args["K"], args["max_iter"], args["freq"], args["reg"], args["tol"]

    # average of 3 runs to get average of images
    top1_vals = []
    top3_vals = []
    l_times = []
    for i in range(5):

        # get new images
        train, val = train_val(labels, 5)

        print("Starting LMNN training")
        start = time()
        d = lmnn(batch_size=64, center=False, distance=np.eye(512),
                input=embeds[train], k=K, labels=labels[train],
                linear_scan=False, max_iterations=max_iter, normalize=False,
                optimizer="lbfgs", passes=100, print_accuracy=False, range=freq, rank=512,
                regularization=reg, step_size=0.5, tolerance=tol,
                verbose=False)
        l_time = time() - start

        print("LMNN run:", time() - start)#, "Optimizer:", optimizer)

        W_matrix = d['output']

        # transform the matrix -> embeds
        W_embeds = embeds @ W_matrix  # transformed embeds
        W_embeds = torch.from_numpy(W_embeds).cpu()
        W_labels = torch.from_numpy(labels).cpu()

        print("Starting KNN predictions")
        start = time()
        w1, w3 = knn_sim(W_embeds, W_labels, train, val)
        print("KNN run:", time() - start)
        top1_vals.append(w1)
        top3_vals.append(w3)
        l_times.append(l_time)

        # if the time per run takes more than 6 mins we only do 3 runs
        if l_time > 240 and i >= 2:
            break

    # write to csv file
    #print(os.getcwd())
    # with open("../lmnn_log.txt", "a+") as f:
    #     f.write(f"\n{optimizer}, {LR}, {K}, {freq}, {np.round(np.mean(l_times), 2)}, {np.round(np.mean(top1_vals)*100, 3)}, {np.round(np.mean(top3_vals)*100, 3)}")

    with open("../lmnn_log.txt", "a+") as f:
        f.write(f"\n{max_iter}, {reg}, {K}, {freq}, {tol}, {np.round(np.mean(l_times), 2)}, {np.round(np.mean(top1_vals)*100, 3)}, {np.round(np.mean(top3_vals)*100, 3)}")

    acc = np.mean(top1_vals)
    if l_time > 300:
        # punish for taking more than 5 mins
        acc -= l_time/50000

    tune.report(accuracy=acc)

    #return {'accuracy': np.mean(top1_vals), 'status': STATUS_OK}


if __name__ == '__main__':

    ray.init(configure_logging=False)

    a = np.load("visiolab_embeds.npz")
    embeds = np.array(a["embeds"], dtype=np.float32)
    labels = np.array(a["labels"], dtype=np.int32)

    # train, val = train_val(labels, 5)

    # start = time()
    # acc = w1 = knn_sim(torch.from_numpy(embeds), torch.from_numpy(labels), train, val)
    # print(time() - start)

    with open("my_results/experiment7/lmnn_log.txt", "a+") as f:
        f.write("Optimizer, LR, K, Freq, Tol, Time, Top1, Top3")

    param_hyperopt = {
        'LR': hp.loguniform('LR', np.log(1e-6), np.log(1e-1)),
        'K': hp.choice('K', [1, 2, 3, 4]),
        'optimizer': hp.choice('optimizer', ["amsgrad", "lbfgs"]),
        'freq': scope.int(hp.choice('freq', np.arange(1, 1000, 10, dtype=np.int32)))
    }

    lbfgs_hyperopt = {
        'K': hp.choice('K', [1, 2, 3, 4]),
        'max_iter': scope.int(hp.choice('max_iter', np.arange(1_000, 200_000, 500, dtype=np.int32))),
        'freq': scope.int(hp.choice('freq', np.arange(1, 1000, 10, dtype=np.int32))),
        'reg': hp.uniform('reg', 0, 0.7),
        'tol': hp.uniform('tol', 1e-4, 0.3),
    }

    algo = HyperOptSearch(lbfgs_hyperopt, metric="accuracy", mode="max", max_concurrent=24)
    scheduler = AsyncHyperBandScheduler(metric="accuracy", mode="max")
    tune.run(full_sim, 
             search_alg=algo,
             scheduler=scheduler,
             num_samples=240,
             verbose=2,
             name="experiment7",
             local_dir="my_results")

    print(tune)


    #trials = SparkTrials(parallelism=2)
    # trials = Trials()
    # best_param = fmin(full_sim, 
    #                   param_hyperopt, 
    #                   algo=tpe.suggest, 
    #                   max_evals=10, 
    #                   trials=trials,
    #                   rstate= np.random.RandomState(42))
    # print(best_param)
    # best_param_values = [x for x in best_param.values()]
    # print(best_param_values)
    # np.save("test.npy", best_param_values)
