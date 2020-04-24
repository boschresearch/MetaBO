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
# evaluate_metabo_adaboost.py
# Reproduce results from MetaBO paper on ADABOOST-hyperparameter optimization
# For convenience, we provide the pretrained weights resulting from the experiments described in the paper.
# These weights can be reproduced using train_metabo_adaboost.py
# ******************************************************************

# Note: due to licensing issues, the datasets used in this experiment cannot be shipped with the MetaBO package.
# However, you can download the datasets yourself from https://github.com/nicoschilling/ECML2016
# Put the folder "data/adaboost" from this repository into metabo/environment/hpo/data

import os
from metabo.eval.evaluate import eval_experiment
from metabo.eval.plot_results import plot_results
from metabo.environment.hpo.prepare_data import prepare_hpo_data
from metabo.policies.taf.generate_taf_data_hpo import generate_taf_data_hpo
from gym.envs.registration import register, registry
from datetime import datetime

# set evaluation parameters
afs_to_evaluate = ["MetaBO", "TAF-ME", "TAF-RANKING", "EI"]
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metabo")
logpath = os.path.join(rootdir, "iclr2020", "hpo", "MetaBO-ADABOOST-v0")
savepath = os.path.join(logpath, "eval", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
n_workers = 1
n_episodes = 15  # 15 test sets

prepare_hpo_data(model="adaboost", datapath=os.path.join(rootdir, "environment", "hpo", "data", "adaboost"))

# evaluate all afs
for af in afs_to_evaluate:
    # set af-specific parameters
    if af == "MetaBO":
        features = ["posterior_mean", "posterior_std", "timestep", "budget", "x"]
        pass_X_to_pi = False
        n_init_samples = 0
        load_iter = 499  # determined via leave-one-out cross-validation on the training set
        deterministic = True
        policy_specs = {}  # will be loaded from the logfiles
    elif af == "TAF-ME" or af == "TAF-RANKING":
        generate_taf_data_hpo(model="adaboost", datapath=os.path.join(rootdir, "environment", "hpo", "processed"))
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep", "x"]
        pass_X_to_pi = True
        n_init_samples = 0
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        policy_specs = {"TAF_datafile": os.path.join(rootdir, "policies", "taf", "taf_adaboost_M_35_N_108.pkl")}
    else:
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep"]
        pass_X_to_pi = False
        n_init_samples = 0  # no initial design for discrete domain
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        if af == "EI":
            policy_specs = {}
        elif af == "Random":
            policy_specs = {}
        else:
            raise ValueError("Unknown AF!")

    # define environment
    env_spec = {
        "env_id": "MetaBO-ADABOOST-v0",
        "D": 2,
        "f_type": "HPO",
        "f_opts": {
            "hpo_data_file": os.path.join(rootdir, "environment", "hpo", "processed", "adaboost", "objectives.pkl"),
            "hpo_gp_hyperparameters_file": os.path.join(rootdir, "environment", "hpo", "processed", "adaboost",
                                                        "gp_hyperparameters.pkl"),
            "hpo_datasets_file": os.path.join(rootdir, "environment", "hpo", "processed", "adaboost",
                                              "test_datasets_iclr2020.txt"),
            "draw_random_datasets": False,  # present each test function once
            "min_regret": 0.0},
        "features": features,
        "T": 15,
        "n_init_samples": n_init_samples,
        "pass_X_to_pi": pass_X_to_pi,
        # GP hyperparameters will be set individually for each new function, the parameters were determined off-line
        # via type-2-ML on all available data
        "kernel_lengthscale": None,
        "kernel_variance": None,
        "noise_variance": None,
        "use_prior_mean_function": True,
        "local_af_opt": False,  # discrete domain
        "cardinality_domain": 108,
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
plot_results(path=savepath, logplot=True)
print("Done! Saved plot in {}".format(savepath))
