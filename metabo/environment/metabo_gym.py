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
# metabo_gym.py
# Implementation of the basic gym-environment for the MetaBO framework.
# ******************************************************************

import gym
import gym.spaces
import numpy as np
import sobol_seq
import GPy
import pickle as pkl
import json
from metabo.environment.util import create_uniform_grid, scale_from_unit_square_to_domain, \
    scale_from_domain_to_unit_square, get_cube_around
from metabo.environment.objectives import SparseSpectrumGP, bra_var, bra_max_min_var, gprice_var, gprice_max_min_var, \
    hm3_var, hm3_max_min_var, hpo, hpo_max_min, get_hpo_domain, rhino_translated, rhino_max_min_translated, \
    rhino2, rhino2_max_min
from metabo.environment.furuta import init_furuta_simulation, furuta_simulation
from matplotlib import pyplot as plt
import os

class MetaBO(gym.Env):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # number of dimensions
        self.D = kwargs["D"]

        # the domain (unit hypercube)
        self.domain = np.zeros((self.D,))
        self.domain = np.stack([self.domain, np.ones(self.D, )], axis=1)

        # optimization horizon
        self.T = None  # will be set in self.reset
        if "T" in kwargs:
            self.T_min = self.T_max = kwargs["T"]
        else:
            self.T_min = kwargs["T_min"]
            self.T_max = kwargs["T_max"]
        assert self.T_min > 0
        assert self.T_min <= self.T_max

        # the initial design
        self.n_init_samples = kwargs["n_init_samples"]
        assert self.n_init_samples <= self.T_max
        self.initial_design = sobol_seq.i4_sobol_generate(self.D, self.n_init_samples)

        # the AF and its optimization
        self.af = None
        self.neg_af_and_d_neg_af_d_state = None
        self.do_local_af_opt = kwargs["local_af_opt"]
        if self.do_local_af_opt:
            self.discrete_domain = False

            # prepare xi_t
            self.xi_t = None  # is determined adaptively in each BO step
            self.af_opt_startpoints_t = None  # best k evaluations of af on multistart_grid
            self.af_maxima_t = None  # the resulting local af_maxima
            self.N_MS = kwargs["N_MS"]
            N_MS_per_dim = np.int(np.floor(self.N_MS ** (1 / self.D)))
            self.multistart_grid, _ = create_uniform_grid(self.domain, N_MS_per_dim)
            self.N_MS = self.multistart_grid.shape[0]
            self.k = kwargs["k"]  # number of multistarts
            self.cardinality_xi_local_t = self.k
            self.cardinality_xi_global_t = self.N_MS
            self.cardinality_xi_t = self.cardinality_xi_local_t + self.cardinality_xi_global_t

            # hierarchical gridding or gradient-based optimization?
            self.N_LS = kwargs["N_LS"]
            self.local_search_grid = sobol_seq.i4_sobol_generate(self.D, self.N_LS)
            self.af_max_search_diam = 2 * 1 / N_MS_per_dim
        else:
            self.discrete_domain = True
            # self.xi_t = None  # will be set for each new function
            self.cardinality_xi_t = kwargs["cardinality_domain"]
            self.xi_t = sobol_seq.i4_sobol_generate(self.D, self.cardinality_xi_t)

        # the features
        self.features = kwargs["features"]
        self.feature_order_eval_envs = ["posterior_mean", "posterior_std", "incumbent", "timestep"]
        self.feature_order_eps_greedy = ["posterior_mean", "posterior_std"] + \
                                        ["x"] * self.D + ["incumbent", "timestep", "budget"]
        self.feature_order_gmm_ucb = ["posterior_mean", "posterior_std"] + ["x"] * self.D + ["timestep", "budget"]

        # observation space
        self.n_features = 0
        if "posterior_mean" in self.features:
            self.n_features += 1
        if "posterior_std" in self.features:
            self.n_features += 1
        if "left_budget" in self.features:
            self.n_features += 1
        if "budget" in self.features:
            self.n_features += 1
        if "incumbent" in self.features:
            self.n_features += 1
        if "timestep_perc" in self.features:
            self.n_features += 1
        if "timestep" in self.features:
            self.n_features += 1
        if "x" in self.features:
            self.n_features += self.D
        self.observation_space = gym.spaces.Box(low=-100000.0, high=100000.0,
                                                shape=(self.cardinality_xi_t, self.n_features),
                                                dtype=np.float32)
        self.pass_X_to_pi = kwargs["pass_X_to_pi"]

        # action space: index of one of the grid points
        self.action_space = gym.spaces.Discrete(self.cardinality_xi_t)

        # optimization step
        self.t = None

        # the reward
        self.reward_transformation = kwargs["reward_transformation"]

        # the ground truth function
        self.f_type = kwargs["f_type"]
        self.f_opts = kwargs["f_opts"]
        self.f = None
        self.y_max = None
        self.y_min = None
        self.x_max = None

        # the training data
        self.X = self.Y = None  # None means empty
        self.gp_is_empty = True

        # the surrogate GP
        self.mf = None
        self.gp = None
        self.kernel_variance = kwargs["kernel_variance"]
        self.kernel_lengthscale = np.array(kwargs["kernel_lengthscale"])
        self.noise_variance = kwargs["noise_variance"]
        if "use_prior_mean_function" in kwargs and kwargs["use_prior_mean_function"]:
            self.use_prior_mean_function = True
        else:
            self.use_prior_mean_function = False

        # seeding
        self.rng = None
        self.seeded_with = None

        # plot count
        self.plot_count = 0

    def seed(self, seed=None):
        # sets up the environment-internal random number generator and seeds it with seed
        self.rng = np.random.RandomState()
        self.seeded_with = seed
        self.rng.seed(self.seeded_with)

    def set_af_functions(self, af_fun):
        # connect the policy with the environment for setting up the adaptive grid

        if not self.pass_X_to_pi:
            self.af = af_fun
        else:
            self.af = lambda state: af_fun(state, self.X, self.gp)

    def reset(self):
        # reset step counters
        self.reset_step_counters()

        # draw a new function from self.f_type
        self.draw_new_function()

        # reset the GP
        self.reset_gp()

        # optimize the AF
        self.optimize_AF()

        # plot
        if self.kwargs.get("plot"):
            self.plot()

        return self.get_state(self.xi_t)

    def step(self, action):
        assert self.t < self.T  # if self.t == self.T one should have called self.reset() before calling this method
        if self.Y is None:
            assert self.t == 0
        else:
            assert self.t == self.Y.size

        if self.t < self.n_init_samples:
            # ignore action, x_action if there are points from the initial design left
            x_action = self.initial_design[self.t, :].reshape(1, self.D)
        else:
            # sample function, add it to the GP and retrieve resulting state
            x_action = self.convert_idx_to_x(action)
        self.add_data(x_action)  # do this BEFORE calling get_reward()
        reward = self.get_reward()
        self.update_gp()  # do this AFTER calling get_reward()
        self.optimize_AF()
        if self.kwargs.get("plot"):
            self.plot()
        next_state = self.get_state(self.xi_t)
        done = self.is_terminal()

        return next_state, reward, done, {}

    def reset_step_counters(self):
        self.t = 0

        if self.T_min == self.T_max:
            self.T = self.T_min
        else:
            self.T = self.rng.randint(low=self.T_min, high=self.T_max)
        assert self.T > 0  # if T was set outside of init

    def close(self):
        pass

    def draw_new_function(self):
        if self.f_type == "GP":
            seed = self.rng.randint(100000)
            n_features = 500
            lengthscale = self.rng.uniform(low=self.f_opts["lengthscale_low"],
                                           high=self.f_opts["lengthscale_high"])
            noise_var = self.rng.uniform(low=self.f_opts["noise_var_low"],
                                         high=self.f_opts["noise_var_high"])
            signal_var = self.rng.uniform(low=self.f_opts["signal_var_low"],
                                          high=self.f_opts["signal_var_high"])
            kernel = self.f_opts["kernel"] if "kernel" in self.f_opts else "RBF"
            ssgp = SparseSpectrumGP(seed=seed, input_dim=self.D, noise_var=noise_var, length_scale=lengthscale,
                                    signal_var=signal_var, n_features=n_features, kernel=kernel)
            x_train = np.array([]).reshape(0, self.D)
            y_train = np.array([]).reshape(0, 1)
            ssgp.train(x_train, y_train, n_samples=1)
            self.f = lambda x: ssgp.sample_posterior_handle(x).reshape(-1, 1)

            # load gp-hyperparameters
            self.kernel_lengthscale = lengthscale
            self.kernel_variance = signal_var
            self.noise_variance = 1e-20

            if self.do_local_af_opt:
                # determine approximate maximum
                N_samples_dim = np.ceil(1000 ** (1 / self.D))
                assert N_samples_dim > 3
                x_vec, _ = create_uniform_grid(self.domain, N_samples_dim=N_samples_dim)
            else:
                # determine true maximum on grid
                x_vec = self.xi_t
            y_vec = self.f(x_vec)
            self.x_max = x_vec[np.argmax(y_vec)].reshape(1, self.D)
            self.y_max = np.max(y_vec)
            self.y_min = np.min(y_vec)

            self.f = lambda x: ssgp.sample_posterior_handle(x).reshape(-1, 1) - self.f_opts["min_regret"]
        elif self.f_type == "BRA-var":
            # the branin function with translations and scalings
            if "M" in self.f_opts and self.f_opts["M"] is not None:
                # use M fixed source datasets evenly spread over the training set
                M = self.f_opts["M"]
                bound_scaling = self.f_opts["bound_scaling"]
                bound_translation = self.f_opts["bound_translation"]
                fct_params_domain = np.array([[-bound_translation, bound_translation],
                                              [-bound_translation, bound_translation],
                                              [1 - bound_scaling, 1 + bound_scaling]])
                fct_params_grid = sobol_seq.i4_sobol_generate(dim_num=3, n=M)  # 2 translations, 1 scaling
                fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

                param_idx = self.rng.choice(M)

                t = fct_params_grid[param_idx, 0:2]
                s = fct_params_grid[param_idx, 2]
            else:
                if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, 2))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
                elif "translation_min" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=self.f_opts["translation_min"],
                                         high=self.f_opts["translation_max"], size=(1, 2))
                    # sample signs of translation
                    for i in range(t.shape[1]):
                        coin = self.rng.uniform(low=0.0, high=1.0)
                        if coin > 0.5:
                            t[0, i] = -t[0, i]

                    # sample scaling
                    s = self.rng.uniform(low=self.f_opts["scaling_min"], high=self.f_opts["scaling_max"])
                else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: bra_var(x, t=t, s=s)

            max_pos, max, _, min = bra_max_min_var(t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min
        elif self.f_type == "GPRICE-var":
            # the goldstein-price function with translations and scalings
            if "M" in self.f_opts and self.f_opts["M"] is not None:
                # use M fixed source datasets evenly spread over the training set
                M = self.f_opts["M"]
                bound_scaling = self.f_opts["bound_scaling"]
                bound_translation = self.f_opts["bound_translation"]
                fct_params_domain = np.array([[-bound_translation, bound_translation],
                                              [-bound_translation, bound_translation],
                                              [1 - bound_scaling, 1 + bound_scaling]])
                fct_params_grid = sobol_seq.i4_sobol_generate(dim_num=3, n=M)  # 2 translations, 1 scaling
                fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

                param_idx = self.rng.choice(M)

                t = fct_params_grid[param_idx, 0:2]
                s = fct_params_grid[param_idx, 2]
            else:
                if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, 2))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
                elif "translation_min" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=self.f_opts["translation_min"],
                                         high=self.f_opts["translation_max"], size=(1, 2))
                    # sample signs of translation
                    for i in range(t.shape[1]):
                        coin = self.rng.uniform(low=0.0, high=1.0)
                        if coin > 0.5:
                            t[0, i] = -t[0, i]

                    # sample scaling
                    s = self.rng.uniform(low=self.f_opts["scaling_min"], high=self.f_opts["scaling_max"])
                else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: gprice_var(x, t=t, s=s)

            max_pos, max, _, min = gprice_max_min_var(t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min
        elif self.f_type == "HM3-var":
            # the hartmann-3D function with translations and scalings
            if "M" in self.f_opts and self.f_opts["M"] is not None:
                # use M fixed source datasets evenly spread over the training set
                M = self.f_opts["M"]
                bound_scaling = self.f_opts["bound_scaling"]
                bound_translation = self.f_opts["bound_translation"]
                fct_params_domain = np.array([[-bound_translation, bound_translation],
                                              [-bound_translation, bound_translation],
                                              [-bound_translation, bound_translation],
                                              [1 - bound_scaling, 1 + bound_scaling]])
                fct_params_grid = sobol_seq.i4_sobol_generate(dim_num=4, n=M)  # 3 translations, 1 scaling
                fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

                param_idx = self.rng.choice(M)

                t = fct_params_grid[param_idx, 0:3]
                s = fct_params_grid[param_idx, 3]
            else:
                if "bound_translation" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                         high=self.f_opts["bound_translation"], size=(1, 3))

                    # sample scaling
                    s = self.rng.uniform(low=1 - self.f_opts["bound_scaling"], high=1 + self.f_opts["bound_scaling"])
                elif "translation_min" in self.f_opts:
                    # sample translation
                    t = self.rng.uniform(low=self.f_opts["translation_min"],
                                         high=self.f_opts["translation_max"], size=(1, 3))
                    # sample signs of translation
                    for i in range(t.shape[1]):
                        coin = self.rng.uniform(low=0.0, high=1.0)
                        if coin > 0.5:
                            t[0, i] = -t[0, i]

                    # sample scaling
                    s = self.rng.uniform(low=self.f_opts["scaling_min"], high=self.f_opts["scaling_max"])
                else:
                    raise ValueError("Missspecified translation/scaling parameters!")

            self.f = lambda x: hm3_var(x, t=t, s=s)

            max_pos, max, _, min = hm3_max_min_var(t=t, s=s)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min
        elif self.f_type == "HPO":
            # load data
            if not hasattr(self, "hpo_data"):
                with open(self.f_opts["hpo_data_file"], "rb") as f:
                    self.hpo_data = pkl.load(f)
                with open(self.f_opts["hpo_gp_hyperparameters_file"], "rb") as f:
                    self.hpo_gp_hyperparameters = pkl.load(f)
                with open(self.f_opts["hpo_datasets_file"], "r") as f:
                    self.hpo_datasets = json.load(f)

            # sample dataset
            if self.f_opts["draw_random_datasets"]:
                dataset = self.rng.choice(self.hpo_datasets, size=1)[0]
            else:
                if not hasattr(self, "dataset_counter"):
                    self.dataset_counter = 0
                if self.dataset_counter >= len(self.hpo_datasets):
                    self.dataset_counter = 0
                dataset = self.hpo_datasets[self.dataset_counter]
                self.dataset_counter += 1

            # load gp-hyperparameters
            self.kernel_lengthscale = self.hpo_gp_hyperparameters[dataset]["lengthscale"]
            self.kernel_variance = self.hpo_gp_hyperparameters[dataset]["variance"]
            self.noise_variance = self.hpo_gp_hyperparameters[dataset]["noise_variance"]

            # set xi_t
            self.xi_t = get_hpo_domain(data=self.hpo_data, dataset=dataset)
            assert self.xi_t.shape[0] == self.cardinality_xi_t

            self.f = lambda x: hpo(x, data=self.hpo_data, dataset=dataset) - self.f_opts["min_regret"]

            max_pos, max, _, min = hpo_max_min(data=self.hpo_data, dataset=dataset)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min
        elif self.f_type == "Furuta":
            if "M" in self.f_opts and self.f_opts["M"] is not None:
                # use M fixed sets of physical parameters evenly spread over the training set
                M = self.f_opts["M"]
                furuta_domain = np.array(self.f_opts["furuta_domain"])
                length_arm_low = self.f_opts["length_arm_low"]
                length_arm_high = self.f_opts["length_arm_high"]
                length_pendulum_low = self.f_opts["length_pendulum_low"]
                length_pendulum_high = self.f_opts["length_pendulum_high"]
                mass_arm_low = self.f_opts["mass_arm_low"]
                mass_arm_high = self.f_opts["mass_arm_high"]
                mass_pendulum_low = self.f_opts["mass_pendulum_low"]
                mass_pendulum_high = self.f_opts["mass_pendulum_high"]

                physical_params_domain = np.array([[mass_pendulum_low, mass_pendulum_high],
                                                   [mass_arm_low, mass_arm_high],
                                                   [length_pendulum_low, length_pendulum_high],
                                                   [length_arm_low, length_arm_high]])
                physical_params_grid = sobol_seq.i4_sobol_generate(dim_num=4, n=M)
                physical_params_grid = scale_from_unit_square_to_domain(X=physical_params_grid,
                                                                        domain=physical_params_domain)

                param_idx = self.rng.choice(M)

                mass_pendulum = physical_params_grid[param_idx, 0]
                mass_arm = physical_params_grid[param_idx, 1]
                length_pendulum = physical_params_grid[param_idx, 2]
                length_arm = physical_params_grid[param_idx, 3]
            else:
                # sample physical parameters randomly
                furuta_domain = np.array(self.f_opts["furuta_domain"])
                length_arm = self.rng.uniform(low=self.f_opts["length_arm_low"],
                                              high=self.f_opts["length_arm_high"])
                length_pendulum = self.rng.uniform(low=self.f_opts["length_pendulum_low"],
                                                   high=self.f_opts["length_pendulum_high"])
                mass_arm = self.rng.uniform(low=self.f_opts["mass_arm_low"],
                                            high=self.f_opts["mass_arm_high"])
                mass_pendulum = self.rng.uniform(low=self.f_opts["mass_pendulum_low"],
                                                 high=self.f_opts["mass_pendulum_high"])

            init_tuple = init_furuta_simulation(mass_arm=mass_arm,
                                                length_arm=length_arm,
                                                mass_pendulum=mass_pendulum,
                                                length_pendulum=length_pendulum)

            K = np.copy(init_tuple[6]._K)
            self.f = lambda x: -furuta_simulation(
                init_tuple=init_tuple,
                params=scale_from_unit_square_to_domain(x, furuta_domain),
                D=self.D,
                pos=self.f_opts["pos"]).reshape(-1, 1)

            # we don't know the optimal values
            # the lqr_params and lqr_logcost are not really the optimum (nonlinear system) but they are good guesses
            lqr_params = K.squeeze()[self.f_opts["pos"]]
            self.x_max = scale_from_domain_to_unit_square(lqr_params, furuta_domain).reshape(1, self.D)
            self.y_max = self.f(self.x_max)
            self.y_min = -5.0  # = -log(chrashcost)
        elif self.f_type == "RHINO":
            # use a discrete domain
            self.xi_t = np.linspace(0.0, 1.0, self.f_opts["cardinality_domain"]).reshape(
                self.f_opts["cardinality_domain"], self.D)
            assert self.xi_t.shape[0] == self.cardinality_xi_t

            # sample translation
            t = self.rng.uniform(low=-self.f_opts["bound_translation"],
                                 high=self.f_opts["bound_translation"], size=(1, 1))

            self.f = lambda x: rhino_translated(x=x, t=t)

            max_pos, max, _, min = rhino_max_min_translated(t=t)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min

            # y = self.f(self.xi_t)
            # self.x_max = self.xi_t[np.argmax(y)]
            # self.y_max = np.max(y)
            # self.y_min = np.min(y)
        elif self.f_type == "RHINO2":
            # use a discrete domain
            self.xi_t = np.linspace(0.0, 1.0, self.f_opts["cardinality_domain"]).reshape(
                self.f_opts["cardinality_domain"], self.D)
            assert self.xi_t.shape[0] == self.cardinality_xi_t

            # sample translation
            h = self.rng.uniform(low=self.f_opts["h_min"],
                                 high=self.f_opts["h_max"], size=(1, 1))

            self.f = lambda x: rhino2(x=x, h=h)

            max_pos, max, _, min = rhino2_max_min(h=h)
            self.x_max = max_pos
            self.y_max = max
            self.y_min = min

            # y = self.f(self.xi_t)
            # self.x_max = self.xi_t[np.argmax(y)]
            # self.y_max = np.max(y)
            # self.y_min = np.min(y)
        else:
            raise ValueError("Unknown f_type!")

        assert self.y_max is not None  # we need this for the reward
        assert self.y_min is not None  # we need this for the incumbent of empty training set

    def reset_gp(self):
        # reset training data
        self.X = self.Y = None

        # reset gp
        if "kernel" in self.kwargs:
            if self.kwargs["kernel"] == "RBF":
                kernel_fun = GPy.kern.RBF
            elif self.kwargs["kernel"] == "Matern32":
                kernel_fun = GPy.kern.Matern32
            elif self.kwargs["kernel"] == "Matern52":
                kernel_fun = GPy.kern.Matern52
            else:
                raise ValueError("Unknown kernel function for GP model!")
        else:
            kernel_fun = GPy.kern.RBF

        self.kernel = kernel_fun(input_dim=self.D,
                                 variance=self.kernel_variance,
                                 lengthscale=self.kernel_lengthscale,
                                 ARD=True)

        if self.use_prior_mean_function:
            self.mf = GPy.core.Mapping(self.D, 1)
            self.mf.f = lambda X: np.mean(self.Y, axis=0)[0] if self.Y is not None else 0.0
            self.mf.update_gradients = lambda a, b: 0
            self.mf.gradients_X = lambda a, b: 0
        else:
            self.mf = None

        normalizer = False

        # this is only dummy data as GPy is not able to have empty training set
        # for prediction, the GP is not used for empty training set
        X = np.zeros((1, self.D))
        Y = np.zeros((1, 1))
        self.gp_is_empty = True
        self.gp = GPy.models.gp_regression.GPRegression(X, Y,
                                                        noise_var=self.noise_variance,
                                                        kernel=self.kernel,
                                                        mean_function=self.mf,
                                                        normalizer=normalizer)
        self.gp.Gaussian_noise.variance = self.noise_variance
        self.gp.kern.lengthscale = self.kernel_lengthscale
        self.gp.kern.variance = self.kernel_variance

    def add_data(self, X):
        assert X.ndim == 2

        # evaluate f at X and add the result to the GP
        Y = self.f(X)

        if self.X is None:
            self.X = X
            self.Y = Y
        else:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)

        self.t += X.shape[0]

    def update_gp(self):
        assert self.Y is not None
        self.gp.set_XY(self.X, self.Y)
        self.gp_is_empty = False

    def optimize_AF(self):
        if self.do_local_af_opt:
            # obtain maxima of af
            self.get_af_maxima()
            self.xi_t = np.concatenate((self.af_maxima_t, self.multistart_grid), axis=0)
            assert self.xi_t.shape[0] == self.cardinality_xi_t
        else:
            pass  # nothing to be done, we just use the grid self.xi_t which is static in this case

    def get_state(self, X):
        # fill the state
        feature_count = 0
        idx = 0
        state = np.zeros((X.shape[0], self.n_features), dtype=np.float32)
        gp_mean, gp_std = self.eval_gp(X)
        if "posterior_mean" in self.features:
            feature_count += 1
            state[:, idx:idx + 1] = gp_mean.reshape(X.shape[0], 1)
            idx += 1
        if "posterior_std" in self.features:
            feature_count += 1
            state[:, idx:idx + 1] = gp_std.reshape(X.shape[0], 1)
            idx += 1
        if "x" in self.features:
            feature_count += 1
            state[:, idx:idx + self.D] = X
            idx += self.D
        if "incumbent" in self.features:
            feature_count += 1
            incumbent_vec = np.ones((X.shape[0],)) * self.get_incumbent()
            state[:, idx] = incumbent_vec
            idx += 1
        if "timestep_perc" in self.features:
            feature_count += 1
            t_perc = self.t / self.T
            t_perc_vec = np.ones((X.shape[0],)) * t_perc
            state[:, idx] = t_perc_vec
            idx += 1
        if "timestep" in self.features:
            feature_count += 1
            # clip timestep
            if "T_training" in self.kwargs and self.kwargs["T_training"] is not None:
                t = np.min([self.t, self.kwargs["T_training"]])
            else:
                t = self.t
            t_vec = np.ones((X.shape[0],)) * t
            state[:, idx] = t_vec
            idx += 1
        if "budget" in self.features:
            feature_count += 1
            if "T_training" in self.kwargs and self.kwargs["T_training"] is not None:
                T = self.kwargs["T_training"]
            else:
                T = self.T
            budget_vec = np.ones((X.shape[0],)) * T
            state[:, idx] = budget_vec
            idx += 1

        assert idx == self.n_features  # make sure the full state has been filled
        if not feature_count == len(self.features):
            raise ValueError("Invalid feature specification!")

        return state

    def get_reward(self):
        # make sure you already increased the step counter self.t before calling this method!
        # make sure you updated the training set but did NOT update the gp before calling this method!
        assert self.Y is not None  # this method should not be called with empty training set
        negativity_check = False

        # compute the simple regret
        y_diffs = self.y_max - self.Y
        simple_regret = np.min(y_diffs)
        reward = np.asscalar(simple_regret)

        # apply reward transformation
        if self.reward_transformation == "none":
            reward = reward
        elif self.reward_transformation == "neg_linear":
            reward = -reward
        elif self.reward_transformation == "neg_log10":
            if reward < 1e-20:
                print("Warning: logarithmic reward may be invalid!")
            reward, negativity_check = np.max((1e-20, reward)), True
            assert negativity_check
            reward = -np.log10(reward)
        else:
            raise ValueError("Unknown reward transformation!")

        return reward

    def get_af_maxima(self):
        state_at_multistarts = self.get_state(self.multistart_grid)
        af_at_multistarts = self.af(state_at_multistarts)
        self.af_opt_startpoints_t = self.multistart_grid[np.argsort(-af_at_multistarts)[:self.k, ...]]

        local_grids = [scale_from_unit_square_to_domain(self.local_search_grid,
                                                        domain=get_cube_around(x,
                                                                               diam=self.af_max_search_diam,
                                                                               domain=self.domain))
                       for x in self.af_opt_startpoints_t]
        local_grids = np.concatenate(local_grids, axis=0)
        state_on_local_grid = self.get_state(local_grids)
        af_on_local_grid = self.af(state_on_local_grid)
        self.af_maxima_t = local_grids[np.argsort(-af_on_local_grid)[:self.cardinality_xi_local_t]]

        assert self.af_maxima_t.shape[0] == self.cardinality_xi_local_t

    def get_incumbent(self):
        if self.Y is None:
            Y = np.array([self.y_min])
        else:
            Y = self.Y

        incumbent = np.max(Y)
        return incumbent

    def eval_gp(self, X_star):
        # evaluate the GP on X_star
        assert X_star.shape[1] == self.D

        if self.gp_is_empty:
            gp_mean = np.zeros((X_star.shape[0],))
            gp_var = self.kernel_variance * np.ones((X_star.shape[0],))
        else:
            gp_mean, gp_var = self.gp.predict_noiseless(X_star)
            gp_mean = gp_mean[:, 0]
            gp_var = gp_var[:, 0]
        gp_std = np.sqrt(gp_var)

        return gp_mean, gp_std

    def neg_af(self, x):
        x = x.reshape(1, self.D)  # the optimizer queries one point at a time
        state = self.get_state(x)
        neg_af = -self.af(state)

        return neg_af

    def get_random_sampling_reward(self):
        self.reset_step_counters()
        self.draw_new_function()

        self.X, self.Y = None, None
        rewards = []
        for t in range(self.T):
            if not self.discrete_domain:
                random_sample = self.rng.rand(1, self.D)
            else:
                random_sample = self.xi_t[self.rng.choice(np.arange(self.cardinality_xi_t)), :].reshape(1, -1)
            self.X = np.concatenate((self.X, random_sample), axis=0) if self.X is not None else random_sample
            f_x = self.f(random_sample)
            self.Y = np.concatenate((self.Y, f_x), axis=0) if self.Y is not None else f_x
            rewards.append(self.get_reward())
            self.t += 1

        assert self.is_terminal()

        return rewards

    def plot(self):
        assert self.D == 1 or self.D == 2

        if self.D == 1:
            width = 2.1
            height = width / 1.618
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(width, height))
            fig.subplots_adjust(hspace=0.1)
            fig.subplots_adjust(left=.04, bottom=.14, right=.96, top=.97)

            # grid for plotting
            X = np.linspace(0.0, 1.0, 250).reshape(-1, 1)

            # Plot GP, ground truth function, and training set
            ax = axes[0]
            # ax.set_title("GP and ground truth function")

            # plot ground truth function and maximum position
            ax.plot(X, self.f(X), color="k", ls="-", label="objective")
            ax.axvline(self.x_max, color="r", ls="--", alpha=.5)

            # plot the GP
            gp_mean, gp_std = self.eval_gp(X)
            ax.plot(X, gp_mean, "C0", ls="--", label="GP")
            ax.fill_between(X.squeeze(), gp_mean + gp_std, gp_mean - gp_std, color="C0", alpha=0.2)
            # ax.plot(X, gp_mean + gp_std, "b--")
            # ax.plot(X, gp_mean - gp_std, "b--")
            ax.xaxis.set_ticklabels([])
            ax.get_yaxis().set_visible(False)
            ax.set_ylim([-1.5, 1.5])
            # ax.legend()

            # plot the training set
            if self.X is not None and self.X.size > 0:
                ax.scatter(self.X[:-1], self.Y[:-1], color="g", marker="x", s=20)
                ax.scatter(self.X[-1], self.Y[-1], color="r", marker="x", s=20)
            # ax.grid()

            # Plot AF
            ax = axes[1]
            # ax.set_title("Neural Acquisition Function (MetaBO)")

            # plot the af
            state = self.get_state(X)
            af = self.af(state)
            ax.plot(X, af.reshape(X.shape), color="C0")
            if self.t >= self.n_init_samples:
                if self.kwargs.get("local_af_opt"):
                    for af_max in self.af_maxima_t:
                        ax.axvline(x=af_max)
                else:
                    ax.axvline(x=X[np.argmax(af)], color="r", ls="--", alpha=0.5)
            ax.yaxis.set_ticklabels([])
            ax.get_yaxis().set_visible(False)
            # ax.grid()

        elif self.D == 2:
            raise NotImplementedError

        fig.savefig(fname=os.path.join(self.kwargs["plotpath"], "plot_{:d}.pdf".format(self.plot_count)))
        plt.close(fig)
        self.plot_count += 1

    def convert_idx_to_x(self, idx):
        if not isinstance(idx, np.ndarray):
            idx = np.array([idx])

        return self.xi_t[idx, :].reshape(idx.size, self.D)

    def is_terminal(self):
        return self.t == self.T
