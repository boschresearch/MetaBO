from metabo.environment.objectives import hpo
import pickle as pkl
import os
import json


# choose model
def generate_taf_data_hpo(model, datapath):
    # dimension of task
    D = 2

    # load training datasets
    trainsets_file = os.path.join(datapath, model, "train_datasets_iclr2020.txt")
    with open(trainsets_file, "r") as f:
        trainsets = json.load(f)
    M = len(trainsets)

    # load training data
    data_file = os.path.join(datapath, model, "objectives.pkl")
    with open(data_file, "rb") as f:
        hpo_data = pkl.load(f)

    # load gp hyperparameters
    gp_hyperparameters_file = os.path.join(datapath, model, "gp_hyperparameters.pkl")
    with open(gp_hyperparameters_file, "rb") as f:
        gp_hyperparameters = pkl.load(f)

    # load input grid
    Xs = []
    Ys = []
    kernel_lengthscales = []
    kernel_variances = []
    noise_variances = []
    use_prior_mean_function = True
    N = None
    for dataset in trainsets:
        Xs.append(hpo_data[dataset]["X"])
        if N is None:
            N = Xs[-1].shape[0]
        else:
            assert Xs[-1].shape[0] == N  # all X shall contain the same number of points
        Ys.append(hpo(Xs[-1], data=hpo_data, dataset=dataset))
        kernel_lengthscales.append(gp_hyperparameters[dataset]["lengthscale"])
        kernel_variances.append(gp_hyperparameters[dataset]["variance"])
        noise_variances.append(gp_hyperparameters[dataset]["noise_variance"])

    data = {"D": D,
            "M": M,
            "X": Xs,
            "Y": Ys,
            "kernel_lengthscale": kernel_lengthscales,
            "kernel_variance": kernel_variances,
            "noise_variance": noise_variances,
            "use_prior_mean_function": M * [use_prior_mean_function]}

    this_path = os.path.dirname(os.path.realpath(__file__))
    datafile = os.path.join(this_path, "taf_{}_M_{:d}_N_{:d}.pkl".format(model, M, N))
    with open(datafile, "wb") as f:
        pkl.dump(data, f)
