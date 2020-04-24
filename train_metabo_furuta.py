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
# train_metabo_furuta.py
# Train MetaBO on Furuta control task in simulation
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
true_mass_arm = 0.095
true_mass_pendulum = 0.024
true_length_arm = 0.085
true_length_pendulum = 0.129
low_mult = 0.75
high_mult = 1.25
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
    "features": ["posterior_mean", "posterior_std", "x"],
    "T_min": 1,
    "T_max": 25,
    "n_init_samples": 0,
    "pass_X_to_pi": False,
    "kernel_lengthscale": [0.1, 0.1, 0.1, 0.1],
    "kernel_variance": 1.5,
    "noise_variance": 1e-2,
    "use_prior_mean_function": True,
    "local_af_opt": True,
    "N_MS": 10000,
    "N_LS": 1000,
    "k": 1,
    "reward_transformation": "neg_linear"  # true maximum not known
}

# specify PPO parameters
n_iterations = 2000
batch_size = 1200
n_workers = 10
arch_spec = 4 * [200]
ppo_spec = {
    "batch_size": batch_size,
    "max_steps": n_iterations * batch_size,
    "minibatch_size": batch_size // 50,
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
    max_episode_steps=env_spec["T_max"],
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
