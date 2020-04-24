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
# evaluate.py
# Evaluation of performance of MetaBO and benchmark AFs.
# ******************************************************************

import os
import json
import numpy as np
import gym
import pickle as pkl
import torch
from datetime import datetime
from collections import namedtuple
from metabo.policies.policies import NeuralAF, UCB, EI, PI, TAF, EpsGreedy, GMM_UCB
from metabo.ppo.batchrecorder import BatchRecorder, Transition

Result = namedtuple("Result",
                    "logpath env_id env_specs policy policy_specs deterministic load_iter T n_episodes rewards")


def write_overview_logfile(savepath, timestamp, env, env_seeds, policy, policy_specs, taf_datafile=None, verbose=False):
    fname = "000_eval_overview_{}.txt".format(policy)
    s = ""
    s += "********* OVERVIEW ENVIRONMENT PARAMETERS *********\n"
    s += "Evaluation timestamp: {}\n".format(timestamp)
    s += "Environment-ID: {}\n".format(env.spec.id)
    s += "Environment-kwargs:\n"
    s += json.dumps(env.spec._kwargs, indent=2)
    s += "\n"
    s += "Environment-seeds:\n"
    s += str(env_seeds)
    s += "\n"
    s += "Policy-specs:\n"
    s += json.dumps(policy_specs, indent=2)
    if taf_datafile is not None:
        s += "\n"
        s += "TAF-Datafile: {}".format(taf_datafile)
    with open(os.path.join(savepath, fname), "w") as f:
        print(s, file=f)
    if not verbose:
        print(s)


def load_metabo_policy(logpath, load_iter, env, device, deterministic):
    with open(os.path.join(logpath, "params_" + str(load_iter)), "rb") as f:
        train_params = pkl.load(f)

    pi = NeuralAF(observation_space=env.observation_space,
                  action_space=env.action_space,
                  deterministic=deterministic,
                  options=train_params["policy_options"]).to(device)
    with open(os.path.join(logpath, "weights_" + str(load_iter)), "rb") as f:
        pi.load_state_dict(torch.load(f))
    with open(os.path.join(logpath, "stats_" + str(load_iter)), "rb") as f:
        stats = pkl.load(f)

    return pi, train_params, stats


def eval_experiment(eval_spec):
    env_id = eval_spec["env_id"]
    env_seed_offset = eval_spec["env_seed_offset"]
    policy = eval_spec["policy"]
    logpath = eval_spec["logpath"]
    policy_specs = eval_spec["policy_specs"]
    savepath = eval_spec["savepath"]
    n_workers = eval_spec["n_workers"]
    n_episodes = eval_spec["n_episodes"]
    assert n_episodes % n_workers == 0
    T = eval_spec["T"]
    if policy != "MetaBO":
        pi = None
        deterministic = None
        load_iter = None

    os.makedirs(savepath, exist_ok=True)

    env_seeds = env_seed_offset + np.arange(n_workers)
    dummy_env = gym.make(env_id)
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
    taf_datafile = policy_specs["TAF_datafile"] if "TAF_datafile" in policy_specs else None
    write_overview_logfile(savepath=savepath, timestamp=timestamp, env=dummy_env, policy=policy,
                           env_seeds=env_seeds, taf_datafile=taf_datafile, policy_specs=policy_specs)
    env_specs = dummy_env.spec._kwargs

    # prepare the policies
    if policy == "GP-UCB":
        feature_order = dummy_env.unwrapped.feature_order_eval_envs
        D = dummy_env.unwrapped.D
        policy_fn = lambda *_: UCB(feature_order=feature_order,
                                   kappa=policy_specs["kappa"],
                                   D=D,
                                   delta=policy_specs["delta"])
    elif policy == "EI":
        feature_order = dummy_env.unwrapped.feature_order_eval_envs
        policy_fn = lambda *_: EI(feature_order=feature_order)
    elif policy == "TAF-ME":
        policy_fn = lambda *_: TAF(datafile=policy_specs["TAF_datafile"], mode="me")
    elif policy == "TAF-RANKING":
        policy_fn = lambda *_: TAF(datafile=policy_specs["TAF_datafile"], mode="ranking", rho=1.0)
    elif policy == "PI":
        feature_order = dummy_env.unwrapped.feature_order_eval_envs
        policy_fn = lambda *_: PI(feature_order=feature_order, xi=policy_specs["xi"])
    elif policy == "EPS-GREEDY":
        feature_order = dummy_env.unwrapped.feature_order_eps_greedy
        policy_fn = lambda *_: EpsGreedy(datafile=policy_specs["datafile"], feature_order=feature_order,
                                         eps=policy_specs["eps"])
    elif policy == "GMM-UCB":
        feature_order = dummy_env.unwrapped.feature_order_gmm_ucb
        policy_fn = lambda *_: GMM_UCB(datafile=policy_specs["datafile"], feature_order=feature_order,
                                       ucb_kappa=policy_specs["ucb_kappa"], w=policy_specs["w"],
                                       n_components=policy_specs["n_components"])
    elif policy == "MetaBO":
        load_iter = eval_spec["load_iter"]
        deterministic = eval_spec["deterministic"]
        pi, policy_specs, _ = load_metabo_policy(logpath=logpath, load_iter=load_iter, env=dummy_env,
                                                 device="cpu", deterministic=deterministic)

        policy_fn = lambda osp, asp, det: NeuralAF(observation_space=osp,
                                                   action_space=asp,
                                                   deterministic=det,
                                                   options=policy_specs["policy_options"])
    elif policy == "Random":
        pass  # will be dealt with separately below
    else:
        raise ValueError("Unknown policy!")
    dummy_env.close()

    # evaluate the experiment
    if policy != "Random":
        br = BatchRecorder(size=T * n_episodes, env_id=env_id, env_seeds=env_seeds, policy_fn=policy_fn,
                           n_workers=n_workers, deterministic=deterministic)
        if policy == "MetaBO":
            br.set_worker_weights(pi=pi)
        br.record_batch(gamma=1.0, lam=1.0)  # gamma, lam do not matter for evaluation
        transitions = Transition(*zip(*br.memory.copy()))
        rewards = transitions.reward
        br.cleanup()
    else:
        env = gym.make(env_id)
        env.seed(env_seed_offset)
        rewards = []
        for _ in range(n_episodes):
            rewards = rewards + env.unwrapped.get_random_sampling_reward()
        env.close()

    # save result
    result = Result(logpath=logpath, env_id=env_id, env_specs=env_specs, policy=policy, policy_specs=policy_specs,
                    deterministic=deterministic, load_iter=load_iter, T=T, n_episodes=n_episodes, rewards=rewards)
    fn = "result_metabo_iter_{:04d}".format(load_iter) if policy == "MetaBO" else "result_{}".format(policy)
    with open(os.path.join(savepath, fn), "wb") as f:
        pkl.dump(result, f)
