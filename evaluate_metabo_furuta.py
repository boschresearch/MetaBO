# Copyright (c) 2019 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ******************************************************************
# evaluate_metabo_furuta.py
# Reproduce results from MetaBO paper on Furuta control task in simulation
# For convenience, we provide the pretrained weights resulting from the experiments described in the paper.
# These weights can be reproduced using train_metabo_furuta.py
# ******************************************************************

import os
from metabo.eval.evaluate import eval_experiment
from metabo.eval.plot_results import plot_results
from metabo.policies.taf.generate_taf_data_furuta import generate_taf_data_furuta
from gym.envs.registration import register, registry
from datetime import datetime

# set evaluation parameters
afs_to_evaluate = ["MetaBO", "TAF-ME", "TAF-RANKING", "EI"]
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metabo")
logpath = os.path.join(rootdir, "iclr2020", "furuta", "full", "MetaBO-Furuta-v0")
savepath = os.path.join(logpath, "eval", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
n_workers = 10
n_episodes = 100

# evaluate all afs
for af in afs_to_evaluate:
    # set af-specific parameters
    if af == "MetaBO":
        features = ["posterior_mean", "posterior_std", "x"]
        pass_X_to_pi = False
        n_init_samples = 0
        load_iter = 1405  # best ppo iteration during training, determined via metabo/ppo/util/get_best_iter_idx
        deterministic = True
        policy_specs = {}  # will be loaded from the logfiles
    elif af == "TAF-ME" or af == "TAF-RANKING":
        generate_taf_data_furuta(M=100)
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep", "x"]
        pass_X_to_pi = True
        n_init_samples = 0
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        policy_specs = {"TAF_datafile": os.path.join(rootdir, "policies", "taf", "taf_furuta_M_100_N_200.pkl")}
    else:
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep"]
        pass_X_to_pi = False
        n_init_samples = 1
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        if af == "EI":
            policy_specs = {}
        elif af == "Random":
            policy_specs = {}
        else:
            raise ValueError("Unknown AF!")

    # define environment
    true_mass_arm = 0.095
    true_mass_pendulum = 0.024
    true_length_arm = 0.085
    true_length_pendulum = 0.129
    low_mult = 0.1
    high_mult = 2.0
    env_spec = {
        "env_id": "MetaBO-Furuta-v0",
        "D": 4,
        "f_type": "Furuta",
        "f_opts": {"furuta_domain": [[-0.5, 0.2],
                                     [-1.6, 4.0],
                                     [-0.1, 0.04],
                                     [-0.04, 0.1]],
                   "mass_arm_low": low_mult * true_mass_arm,
                   "mass_arm_high": high_mult * true_mass_arm,
                   "mass_pendulum_low": low_mult * true_mass_pendulum,
                   "mass_pendulum_high": high_mult * true_mass_pendulum,
                   "length_arm_low": low_mult * true_length_arm,
                   "length_arm_high": high_mult * true_length_arm,
                   "length_pendulum_low": low_mult * true_length_pendulum,
                   "length_pendulum_high": high_mult * true_length_pendulum,
                   "pos": [0, 1, 2, 3]},
        "features": features,
        "T": 50,
        "n_init_samples": n_init_samples,
        "pass_X_to_pi": pass_X_to_pi,
        "kernel_lengthscale": [0.1, 0.1, 0.1, 0.1],
        "kernel_variance": 1.5,
        "noise_variance": 1e-2,
        "use_prior_mean_function": True,
        "local_af_opt": True,
        "N_MS": 10000,
        "N_LS": 1000,
        "k": 5,
        "reward_transformation": "none",
    }

    # register gym environment
    if env_spec["env_id"] in registry.env_specs:
        del registry.env_specs[env_spec["env_id"]]
    register(
        id=env_spec["env_id"],
        entry_point="metabo.environment.metabo_gym:MetaBO",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )

    # define evaluation run
    eval_spec = {
        "env_id": env_spec["env_id"],
        "env_seed_offset": 100,
        "policy": af,
        "logpath": logpath,
        "load_iter": load_iter,
        "deterministic": deterministic,
        "policy_specs": policy_specs,
        "savepath": savepath,
        "n_workers": n_workers,
        "n_episodes": n_episodes,
        "T": env_spec["T"],
    }

    # perform evaluation
    print("Evaluating {} on {}...".format(af, env_spec["env_id"]))
    eval_experiment(eval_spec)
    print("Done! Saved result in {}".format(savepath))
    print("**********************\n\n")

# plot (plot is saved to savepath)
print("Plotting...")
plot_results(path=savepath, logplot=False)
print("Done! Saved plot in {}".format(savepath))
