import os
import numpy as np
from metabo.environment.util import scale_from_unit_square_to_domain
from metabo.environment.objectives import bra_var
import sobol_seq
import pickle as pkl


def generate_taf_data_branin(M, N):
    # dimension of task
    D = 2

    # generate grid of function parameters for source tasks
    # number of source tasks
    bound_scaling = 0.1
    bound_translation = 0.1
    fct_params_domain = np.array([[-bound_translation, bound_translation],
                                  [-bound_translation, bound_translation],
                                  [1 - bound_scaling, 1 + bound_scaling]])
    fct_params_grid = sobol_seq.i4_sobol_generate(dim_num=3, n=M)  # 2 translations, 1 scaling
    fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

    # generate grid of control parameters
    # number of parameter configurations
    input_grid = sobol_seq.i4_sobol_generate(dim_num=D, n=N)

    # generate data
    branin_kernel_lengthscale = [0.235, 0.578]
    branin_kernel_variance = 2.0
    branin_noise_variance = 8.9e-16
    use_prior_mean_function = False
    data = {"D": D,
            "M": M,
            "X": M * [input_grid],
            "Y": M * [None],  # is computed below
            "kernel_lengthscale": M * [branin_kernel_lengthscale],
            "kernel_variance": M * [branin_kernel_variance],
            "noise_variance": M * [branin_noise_variance],
            "use_prior_mean_function": M * [use_prior_mean_function]}

    for i, fct_params in enumerate(fct_params_grid):
        t = np.array([fct_params[0], fct_params[1]])
        s = fct_params[2]
        fct_eval = bra_var(x=input_grid, t=t, s=s)
        data["Y"][i] = fct_eval

    this_path = os.path.dirname(os.path.realpath(__file__))
    datafile = os.path.join(this_path, "taf_branin_M_{:d}_N_{:d}.pkl".format(M, N))
    with open(datafile, "wb") as f:
        pkl.dump(data, f)
