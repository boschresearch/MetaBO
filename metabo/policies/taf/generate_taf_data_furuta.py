import os
import numpy as np
from metabo.environment.util import scale_from_unit_square_to_domain
from metabo.environment.furuta import init_furuta_simulation, furuta_simulation
import sobol_seq
import pickle as pkl


def generate_taf_data_furuta(M):
    # generate grid of physical parameters for source tasks
    # number of source tasks
    true_length_arm = 0.085
    true_length_pendulum = 0.129
    true_mass_arm = 0.095
    true_mass_pendulum = 0.024
    low_mult = 0.75
    high_mult = 1.25
    length_arm_low = low_mult * true_length_arm
    length_arm_high = high_mult * true_length_arm
    length_pendulum_low = low_mult * true_length_pendulum
    length_pendulum_high = high_mult * true_length_pendulum
    mass_arm_low = low_mult * true_mass_arm
    mass_arm_high = high_mult * true_mass_arm
    mass_pendulum_low = low_mult * true_mass_pendulum
    mass_pendulum_high = high_mult * true_mass_pendulum
    physical_params_domain = np.array([[mass_pendulum_low, mass_pendulum_high],
                                       [mass_arm_low, mass_arm_high],
                                       [length_pendulum_low, length_pendulum_high],
                                       [length_arm_low, length_arm_high]])
    physical_params_grid = sobol_seq.i4_sobol_generate(dim_num=4, n=M)
    physical_params_grid = scale_from_unit_square_to_domain(X=physical_params_grid, domain=physical_params_domain)

    # generate grid of control parameters
    # number of control parameter configurations
    N = 200
    control_params_domain = np.array([[-0.5, 0.2],
                                      [-1.6, 4.0],
                                      [-0.1, 0.04],
                                      [-0.04, 0.1]])
    simcore_idx = [0, 1, 2, 3]
    d = 4
    idx = np.array(simcore_idx)
    furuta_pos = np.array([0, 1, 2, 3])
    furuta_pos = furuta_pos[idx[0:d]].tolist()
    control_params_domain_domain = control_params_domain[idx[0:d], :].tolist()
    control_params_grid = sobol_seq.i4_sobol_generate(dim_num=4, n=N)

    # generate data
    furuta_kernel_lengthscale = 0.10 * np.ones((4,))
    furuta_kernel_variance = 1.5
    furuta_noise_variance = 1e-2
    use_prior_mean_function = True
    data = {"D": 4,
            "M": M,
            "X": M * [control_params_grid],
            "Y": M * [None],  # is computed below
            "kernel_lengthscale": M * [furuta_kernel_lengthscale],
            "kernel_variance": M * [furuta_kernel_variance],
            "noise_variance": M * [furuta_noise_variance],
            "use_prior_mean_function": M * [use_prior_mean_function]}

    for i, physical_params in enumerate(physical_params_grid):
        init_tuple = init_furuta_simulation(mass_pendulum=physical_params[0],
                                            mass_arm=physical_params[1],
                                            length_pendulum=physical_params[2],
                                            length_arm=physical_params[3])

        neg_logcosts = -furuta_simulation(init_tuple=init_tuple,
                                          params=scale_from_unit_square_to_domain(X=control_params_grid,
                                                                                  domain=control_params_domain),
                                          D=data["D"],
                                          pos=furuta_pos)
        data["Y"][i] = neg_logcosts[:, None]

    this_path = os.path.dirname(os.path.realpath(__file__))
    datafile = os.path.join(this_path, "taf_furuta_M_{:d}_N_{:d}.pkl".format(M, N))
    with open(datafile, "wb") as f:
        pkl.dump(data, f)
