import os
import numpy as np
from metabo.environment.util import scale_from_unit_square_to_domain
from metabo.environment.objectives import hm3_var
import sobol_seq
import pickle as pkl


def generate_taf_data_hm3(M, N):
    # dimension of task
    D = 3

    # generate grid of function parameters for source tasks
    # number of source tasks
    bound_scaling = 0.1
    bound_translation = 0.1
    fct_params_domain = np.array([[-bound_translation, bound_translation],
                                  [-bound_translation, bound_translation],
                                  [-bound_translation, bound_translation],
                                  [1 - bound_scaling, 1 + bound_scaling]])
    fct_params_grid = sobol_seq.i4_sobol_generate(dim_num=4, n=M)  # 3 translations, 1 scaling
    fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

    # generate grid of control parameters
    # number of parameter configurations
    input_grid = sobol_seq.i4_sobol_generate(dim_num=D, n=N)

    # generate data
    hm3_kernel_lengthscale = np.array([0.716, 0.298, 0.186])
    hm3_kernel_variance = 0.83
    hm3_noise_variance = 1.688e-11
    use_prior_mean_function = False
    data = {"D": D,
            "M": M,
            "X": M * [input_grid],
            "Y": M * [None],  # is computed below
            "kernel_lengthscale": M * [hm3_kernel_lengthscale],
            "kernel_variance": M * [hm3_kernel_variance],
            "noise_variance": M * [hm3_noise_variance],
            "use_prior_mean_function": M * [use_prior_mean_function]}

    for i, fct_params in enumerate(fct_params_grid):
        t = np.array([fct_params[0], fct_params[1], fct_params[2]])
        s = fct_params[3]
        fct_eval = hm3_var(x=input_grid, t=t, s=s)
        data["Y"][i] = fct_eval

    this_path = os.path.dirname(os.path.realpath(__file__))
    datafile = os.path.join(this_path, "taf_hm3_M_{:d}_N_{:d}.pkl".format(M, N))
    with open(datafile, "wb") as f:
        pkl.dump(data, f)
