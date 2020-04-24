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
# plot_results.py
# Functionality for plotting performance of AFs.
# ******************************************************************

import os
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
from metabo.eval.evaluate import Result  # for unpickling


def plot_results2(path, logplot=False):
    # this function corrects for approximate maxima in contrast to plot_results1
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # collect results in savepath
    results = []
    for fn in os.listdir(path):
        if fn.startswith("result"):
            with open(os.path.join(path, fn), "rb") as f:
                result = pkl.load(f)
                results.append(result)

    env_id = results[0].env_id

    # determine best final regret to correct for approximate maxima
    R_best = np.ones((results[0].n_episodes,)) * np.inf
    for result in results:
        # prepare rewards_dict
        rewards_dict = {}
        for i, rew in enumerate(result.rewards):
            if isinstance(rew, tuple):
                t = rew[1]
                reward = rew[0]
            else:
                t = i % result.T + 1
                reward = rew

            if str(t) in rewards_dict:
                rewards_dict[str(t)].append(reward)
            else:
                rewards_dict[str(t)] = [reward]

        # correct for approximate maxima
        for i in range(result.n_episodes):
            cur_R_best = rewards_dict[str(result.T)][i]
            if cur_R_best < R_best[i]:
                R_best[i] = cur_R_best

    # do the plot
    for result in results:
        # prepare rewards_dict
        rewards_dict = {}
        for i, rew in enumerate(result.rewards):
            if isinstance(rew, tuple):
                t = rew[1]
                reward = rew[0]
            else:
                t = i % result.T + 1
                reward = rew

            if str(t) in rewards_dict:
                rewards_dict[str(t)].append(reward)
            else:
                rewards_dict[str(t)] = [reward]

        # correct for approximate maxima
        for i in range(result.n_episodes):
            if R_best[i] < 0:
                for t in range(1, result.T + 1):
                    rewards_dict[str(t)][i] += -R_best[i]

        t_vec, loc, err_low, err_high = [], [], [], []
        for key, val in rewards_dict.items():
            t_vec.append(int(key))
            cur_loc = np.median(val)
            cur_err_low = np.percentile(val, q=70)
            cur_err_high = np.percentile(val, q=30)
            loc.append(cur_loc)
            err_low.append(cur_err_low)
            err_high.append(cur_err_high)

        t_vec, loc, err_low, err_high = np.array(t_vec), np.array(loc), np.array(err_low), np.array(err_high)
        # sort the arrays according to T
        sort_idx = np.argsort(t_vec)
        t_vec = t_vec[sort_idx]
        loc = loc[sort_idx]
        err_low = err_low[sort_idx]
        err_high = err_high[sort_idx]

        if not logplot:
            line = ax.plot(t_vec, loc, label=result.policy)[0]
            ax.fill_between(t_vec, err_low, err_high, alpha=0.2, facecolor=line.get_color())
        else:
            line = ax.semilogy(t_vec, loc, label=result.policy)[0]
            ax.fill_between(t_vec, err_low, err_high, alpha=0.2, facecolor=line.get_color())

    fig.suptitle(env_id)
    ax.grid(alpha=0.3)
    ax.set_xlabel("t", labelpad=0)
    ax.set_ylabel("simple regret")
    ax.legend()

    fig.savefig(fname=os.path.join(path, "plot2.png"))
    plt.close(fig)
