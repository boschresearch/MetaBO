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
# batchrecorder.py
# Implementation of the batchrecorder used in ppo.py
# The Batchrecorder spawns n_workers parallel CPU worker processes (EnvRunner) to step the environments.
# ******************************************************************

import os

os.environ["OMP_NUM_THREADS"] = "1"  # on some machines this is needed to restrict torch to one core

from namedlist import namedlist
import random
import gym
import numpy as np
import multiprocessing as mp
import torch
import itertools
import time

Transition = namedlist("Transition", ["state", "action", "reward", "value", "new", "tdlamret", "adv"])


class EnvRunner(mp.Process):
    def __init__(self, worker_id, size, env_id, seed, policy_fn, task_queue, res_queue, deterministic=False):
        mp.Process.__init__(self)
        self.worker_id = worker_id
        self.env = gym.make(env_id)
        self.seed = seed
        self.task_queue = task_queue
        self.res_queue = res_queue

        # policy
        self.pi = policy_fn(self.env.observation_space, self.env.action_space, deterministic)
        self.pi.set_requires_grad(False)  # we need no gradients here

        # connect policy and environment
        self.env.unwrapped.set_af_functions(af_fun=self.pi.af)

        # empty batch recorder
        assert size > 0
        self.size = size
        self.clear()

        self.set_all_seeds()

    def clear(self):
        self.memory = []
        self.cur_size = self.size
        self.reward_sum = 0
        self.n_new = 0
        self.initial_rewards = []
        self.terminal_rewards = []
        self.next_new = None
        self.next_state = None
        self.next_value = None

    def set_all_seeds(self):
        self.env.seed(self.seed)
        # these seeds are PROCESS-local
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def push(self, state, action, reward, value, new):
        assert not self.is_full()
        assert self.env.observation_space.contains(state)
        assert self.env.action_space.contains(action)
        self.memory.append(Transition(state, action, reward, value, new, None, None))
        self.reward_sum += reward
        self.n_new += int(new)

    def record_batch(self, gamma, lam):
        if self.next_state is None:
            self.pi.reset()
            state = self.env.reset()
            new = 1
        else:
            state = self.next_state
            new = self.next_new
        self.clear()

        while not self.is_full():
            action, value = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.push(state, action, reward, value, new)

            if done:
                self.pi.reset()
                state = self.env.reset()
                new = 1
            else:
                state = next_state
                new = 0

        self.next_new = new
        self.next_state = state
        self.next_value = value

        self.add_tdlamret_and_adv(gamma=gamma, lam=lam)

    def add_tdlamret_and_adv(self, gamma, lam):
        assert self.is_full()
        self.initial_rewards = []  # extraction of initial rewards can happen here w/o overhead
        self.terminal_rewards = []  # extraction of terminal rewards can happen here w/o overhead
        next_new = self.next_new
        next_value = self.next_value
        next_adv = 0
        for i in reversed(range(len(self))):
            nonterminal = 1 - next_new
            value = self.memory[i].value
            reward = self.memory[i].reward
            if self.memory[i].new:
                self.initial_rewards.append(reward)
            if not nonterminal:
                self.terminal_rewards.append(reward)

            delta = -value + reward + gamma * nonterminal * next_value
            self.memory[i].adv = next_adv = delta + lam * gamma * nonterminal * next_adv
            self.memory[i].tdlamret = self.memory[i].adv + value
            next_new = self.memory[i].new
            next_value = value

    def is_full(self):
        return len(self) == self.cur_size

    def is_empty(self):
        return len(self) == 0

    def strip_to_monte_carlo(self):
        assert self.is_full()
        last_new_pos = None
        for i in reversed(range(len(self))):
            if self.memory[i].new == 1:
                last_new_pos = i
                break
            else:
                self.reward_sum -= self.memory[i].reward
        self.next_state = self.memory[last_new_pos].state
        self.next_new = self.memory[last_new_pos].new
        assert self.next_new == 1
        self.next_value = self.memory[last_new_pos].value
        self.memory = self.memory[:last_new_pos]
        self.cur_size = len(self)

    def act(self, state):
        torch.set_num_threads(1)
        with torch.no_grad():
            # to sample the action, the policy uses the current PROCESS-local random seed, don't re-seed in pi.act
            if not self.env.unwrapped.pass_X_to_pi:
                action, value = self.pi.act(torch.from_numpy(state.astype(np.float32)))
            else:
                action, value = self.pi.act(torch.from_numpy(state.astype(np.float32)),
                                            self.env.unwrapped.X,
                                            self.env.unwrapped.gp)
        action = action.numpy()
        value = value.numpy()

        return action, value

    def update_weights(self, pi_state_dict):
        self.pi.load_state_dict(pi_state_dict)

    def __len__(self):
        return len(self.memory)

    def run(self):
        while True:
            task = self.task_queue.get(block=True)
            if task["desc"] == "record_batch":
                self.record_batch(gamma=task["gamma"],
                                  lam=task["lambda"])
                self.res_queue.put((self.worker_id, self.memory, self.reward_sum, self.n_new, self.initial_rewards,
                                    self.terminal_rewards))
                self.task_queue.task_done()
            elif task["desc"] == "set_pi_weights":
                self.update_weights(task["pi_state_dict"])
                self.task_queue.task_done()
            elif task["desc"] == "cleanup":
                self.env.close()
                self.task_queue.task_done()


class BatchRecorder():
    def __init__(self, size, env_id, env_seeds, policy_fn, n_workers, deterministic=False):
        self.env_id = env_id
        self.deterministic = deterministic

        # empty batch recorder
        assert size > 0
        self.n_workers = n_workers
        self.size = size
        self.clear()

        # parallelization
        assert len(env_seeds) == n_workers
        self.env_seeds = env_seeds
        self.task_queue = mp.JoinableQueue()
        self.res_queue = mp.Queue()
        self.worker_batch_sizes = [self.size // self.n_workers] * self.n_workers
        delta_size = self.size - sum(self.worker_batch_sizes)
        assert delta_size == 0, 'All workers shall get assigned the same batch size!'
        self.workers = []
        for i in range(self.n_workers):
            self.workers.append(
                EnvRunner(worker_id=i, size=self.worker_batch_sizes[i], env_id=self.env_id, seed=self.env_seeds[i],
                          policy_fn=policy_fn, task_queue=self.task_queue, res_queue=self.res_queue,
                          deterministic=self.deterministic))
        for i, worker in enumerate(self.workers):
            worker.start()

    def clear(self):
        self.cur_size = self.size
        self.worker_sizes = [0 for _ in range(self.n_workers)]
        self.memory = []
        self.worker_memories = [[] for _ in range(self.n_workers)]
        self.reward_sum = 0
        self.worker_reward_sums = [0 for _ in range(self.n_workers)]
        self.n_new = 0
        self.worker_n_news = [0 for _ in range(self.n_workers)]
        self.initial_rewards = []
        self.worker_initial_rewards = [[] for _ in range(self.n_workers)]
        self.terminal_rewards = []
        self.worker_terminal_rewards = [[] for _ in range(self.n_workers)]

    def overview_dict(self):
        d = {"size": self.size,
             "n_workers": self.n_workers,
             "worker_batch_sizes": self.worker_batch_sizes,
             "env_seeds": self.env_seeds,
             "deterministic": self.deterministic}

        return d

    def record_batch(self, gamma, lam):
        now = time.time()
        task = dict([("desc", "record_batch"),
                     ("gamma", gamma),
                     ("lambda", lam)])
        for _ in range(self.n_workers):
            self.task_queue.put(task)

        self.clear()
        res_count = 0
        while res_count < self.n_workers:
            res_count += 1
            worker_id, cur_memory, cur_rew_sum, cur_n_new, cur_initial_reward, cur_terminal_reward = self.res_queue.get()
            self.worker_memories[worker_id] += cur_memory.copy()
            self.worker_sizes[worker_id] += len(cur_memory)
            self.worker_reward_sums[worker_id] += cur_rew_sum
            self.worker_n_news[worker_id] += cur_n_new
            self.worker_initial_rewards[worker_id] += cur_initial_reward
            self.worker_terminal_rewards[worker_id] += cur_terminal_reward

        self.task_queue.join()

        self.memory = list(itertools.chain.from_iterable([self.worker_memories[i] for i in range(self.n_workers)]))
        self.initial_rewards = list(itertools.chain.from_iterable(self.worker_initial_rewards))
        self.terminal_rewards = list(itertools.chain.from_iterable(self.worker_terminal_rewards))

        assert self.is_full()

        self.reward_sum = sum(self.worker_reward_sums)
        self.n_new = sum(self.worker_n_news)

        return time.time() - now

    def set_worker_weights(self, pi):
        now = time.time()
        pi.to("cpu")
        task = dict([("desc", "set_pi_weights"),
                     ("pi_state_dict", pi.state_dict())])

        for _ in self.workers:
            self.task_queue.put(task)
        self.task_queue.join()

        return time.time() - now

    def cleanup(self):
        for _ in range(self.n_workers):
            self.task_queue.put(dict([("desc", "cleanup")]))
        for worker in self.workers:
            worker.terminate()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_batch_stats(self):
        assert self.is_full()
        batch_stats = dict()
        batch_stats["size"] = len(self)
        batch_stats["worker_sizes"] = self.worker_sizes
        batch_stats["avg_step_reward"] = self.reward_sum / len(self)
        batch_stats["avg_initial_reward"] = np.mean(self.initial_rewards)
        batch_stats["avg_terminal_reward"] = np.mean(self.terminal_rewards)
        batch_stats["avg_ep_reward"] = self.reward_sum / self.n_new
        batch_stats["avg_ep_len"] = len(self) / self.n_new
        batch_stats["n_new"] = self.n_new
        batch_stats["worker_n_news"] = self.worker_n_news
        return batch_stats

    def iterate(self, minibatch_size, shuffle):
        assert self.is_full()
        pos = 0
        idx = list(range(len(self)))
        if shuffle:
            # we use the random state of the main process here, NO re-seeding
            random.shuffle(idx)
        while pos < len(self):
            if pos + 2 * minibatch_size > len(self):
                # enlarge the last minibatch s.t. all minibatches are at least of size minibatch_size
                cur_minibatch_size = len(self) - pos
            else:
                cur_minibatch_size = minibatch_size
            cur_idx = idx[pos:pos + cur_minibatch_size]
            yield [self.memory[i] for i in cur_idx]
            pos += cur_minibatch_size

    def is_empty(self):
        return len(self) == 0

    def is_full(self):
        return len(self) == self.cur_size

    def __len__(self):
        return len(self.memory)
