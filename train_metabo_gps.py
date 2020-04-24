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
# train_metabo_gps.py
# Train MetaBO on GP-samples
# The weights, stats, logs, and the learning curve are stored in metabo/log and can
# be evaluated using metabo/eval/evaluate.py
# ******************************************************************

import os
import multiprocessing as mp
from datetime import datetime
from metabo.policies.policies import NeuralAF
from metabo.ppo.ppo import PPO
from metabo.ppo.plot_learning_curve_online import plot_learning_curve_online
from gym.envs.registration import register

rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metabo")

# specifiy environment
kernel = "Matern52"
env_spec = {
    "env_id": "MetaBO-GP-v0",
    "D": 3,
    "f_type": "GP",
    "f_opts": {"kernel": kernel,
               "lengthscale_low": 0.05,
               "lengthscale_high": 0.5,
               "noise_var_low": 0.1,
               "noise_var_high": 0.1,
               "signal_var_low": 1.0,
               "signal_var_high": 1.0,
               "min_regret": 1e-6},
    "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep", "budget"],
    "T": 30,
    "n_init_samples": 0,
    "pass_X_to_pi": False,
    # will be set individually for each new function to the sampled hyperparameters
    "kernel": kernel,
    "kernel_lengthscale": None,
    "kernel_variance": None,
    "noise_variance": None,
    "use_prior_mean_function": True,
    "local_af_opt": False,
    "cardinality_domain": 2000,
    "reward_transformation": "neg_log10"  # true maximum not known
}

# specify PPO parameters
n_iterations = 2000
batch_size = 1200
n_workers = 10
arch_spec = 2 * [20]
ppo_spec = {
    "batch_size": batch_size,
    "max_steps": n_iterations * batch_size,
    "minibatch_size": batch_size // 20,
    "n_epochs": 4,
    "lr": 1e-4,
    "epsilon": 0.15,
    "value_coeff": 1.0,
    "ent_coeff": 0.01,
    "gamma": 0.98,
    "lambda": 0.98,
    "loss_type": "GAElam",
    "normalize_advs": True,
    "n_workers": n_workers,
    "env_id": env_spec["env_id"],
    "seed": 0,
    "env_seeds": list(range(n_workers)),
    "policy_options": {
        "activations": "relu",
        "arch_spec": arch_spec,
        "exclude_t_from_policy": True,
        "exclude_T_from_policy": True,
        "use_value_network": True,
        "t_idx": -2,
        "T_idx": -1,
        "arch_spec_value": arch_spec
    }
}

# register environment
register(
    id=env_spec["env_id"],
    entry_point="metabo.environment.metabo_gym:MetaBO",
    max_episode_steps=env_spec["T"],
    reward_threshold=None,
    kwargs=env_spec
)

# log data and weights go here, use this folder for evaluation afterwards
logpath = os.path.join(rootdir, "log", env_spec["env_id"], datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))

# set up policy
policy_fn = lambda observation_space, action_space, deterministic: NeuralAF(observation_space=observation_space,
                                                                            action_space=action_space,
                                                                            deterministic=deterministic,
                                                                            options=ppo_spec["policy_options"])

# do training
print("Training on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=1)
# learning curve is plotted online in separate process
p = mp.Process(target=plot_learning_curve_online, kwargs={"logpath": logpath, "reload": True})
p.start()
ppo.train()
p.terminate()
plot_learning_curve_online(logpath=logpath, reload=False)
