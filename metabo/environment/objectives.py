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
# objectives.py
# Implementation of objectives for the experiments on global optimization benchmark functions, GP samples and HPO.
# ******************************************************************

import numpy as np


## Global optimization benchmark functions
# Branin
def bra(x):
    # the Branin function (2D)
    # https://www.sfu.ca/~ssurjano/branin.html
    x1 = x[:, 0]
    x2 = x[:, 1]

    # scale x
    x1 = x1 * 15.
    x1 = x1 - 5.
    x2 = x2 * 15.

    # parameters
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    bra = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    # normalize
    mean = 54.44
    std = 51.44
    bra = 1 / std * (bra - mean)

    # maximize
    bra = -bra

    return bra.reshape(x.shape[0], 1)


def bra_max_min():
    max_pos = np.array([[-np.pi, 12.275]])
    max_pos[0, 0] += 5.
    max_pos[0, 0] /= 15.
    max_pos[0, 1] /= 15.
    max = bra(max_pos)

    min_pos = np.array([[0.0, 0.0]])
    min = bra(min_pos)

    return max_pos, max, min_pos, min


def bra_var(x, t, s):
    x_new = x.copy()
    # apply translation
    # bound the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.12, 0.87],
                        [-0.81, 0.18]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t

    return s * bra(x_new)


def bra_max_min_var(t, s):
    max_pos, max, min_pos, min = bra_max_min()
    # apply translation
    # clip the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.12, 0.87],
                        [-0.81, 0.18]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    max_pos = max_pos + t
    min_pos = min_pos + t

    return max_pos, s * max, min_pos, s * min


# Goldstein-Price
def gprice(x):
    # the goldstein price function (2D)
    # https://www.sfu.ca/~ssurjano/goldpr.html
    x1 = x[:, 0]
    x2 = x[:, 1]

    # scale x
    x1 = x1 * 4.
    x1 = x1 - 2.
    x2 = x2 * 4.
    x2 = x2 - 2.

    gprice = (1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)) * \
             (30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))

    # lognormalize
    mean = 8.693
    std = 2.427
    gprice = 1 / std * (np.log(gprice) - mean)

    # maximize
    gprice = -gprice

    return gprice.reshape(x.shape[0], 1)


def gprice_max_min():
    max_pos = np.array([[0.0, -1.0]])
    max_pos[0, 0] += 2.
    max_pos[0, 0] /= 4.
    max_pos[0, 1] += 2.
    max_pos[0, 1] /= 4.
    max = gprice(max_pos)

    min_pos = np.array([[0.066, 1.0]])
    min = gprice(min_pos)

    return max_pos, max, min_pos, min


def gprice_var(x, t, s):
    x_new = x.copy()
    # apply translation
    # clip the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.5, 0.5],
                        [-0.25, 0.75]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t

    return s * gprice(x_new)


def gprice_max_min_var(t, s):
    # do the transformation in opposite order as in hm3_var!

    max_pos, max, min_pos, min = gprice_max_min()

    # apply translation
    t_range = np.array([[-0.5, 0.5],
                        [-0.25, 0.75]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    max_pos = max_pos + t
    min_pos = min_pos + t

    return max_pos, s * max, min_pos, s * min


# Hartmann-3
def hm3(x):
    # the hartmann3 function (3D)
    # https://www.sfu.ca/~ssurjano/hart3.html

    # parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381, 5743, 8828]])

    x = x.reshape(x.shape[0], 1, -1)
    B = x - P
    B = B ** 2
    exponent = A * B
    exponent = np.einsum("ijk->ij", exponent)
    C = np.exp(-exponent)
    hm3 = -np.einsum("i, ki", alpha, C)

    # normalize
    mean = -0.93
    std = 0.95
    hm3 = 1 / std * (hm3 - mean)

    # maximize
    hm3 = -hm3

    return hm3.reshape(x.shape[0], 1)


def hm3_max_min():
    max_pos = np.array([[0.114614, 0.555649, 0.852547]])
    max = hm3(max_pos)

    min_pos = np.array([[1.0, 1.0, 0.0]])
    min = hm3(min_pos)

    return max_pos, max, min_pos, min


def hm3_var(x, t, s):
    x_new = x.copy()
    # apply translation
    # clip the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.11, 0.88],
                        [-0.55, 0.44],
                        [-0.85, 0.14]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t

    return s * hm3(x_new)


def hm3_max_min_var(t, s):
    # do the transformation in opposite order as in hm3_var!

    max_pos, max, min_pos, min = hm3_max_min()

    # apply translation
    t_range = np.array([[-0.11, 0.88],
                        [-0.55, 0.44],
                        [-0.85, 0.14]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    max_pos = max_pos + t
    min_pos = min_pos + t

    return max_pos, s * max, min_pos, s * min


# General function class (GP samples)
class SparseSpectrumGP:
    """
    Implements the sparse spectrum approximation of a GP following the predictive
    entropy search paper.

    Note: This approximation assumes that we use a GP with squared exponential kernel.
    """

    def __init__(self, input_dim, seed, noise_var=1.0, length_scale=1.0, signal_var=1.0, n_features=100, kernel="RBF"):
        self.seed = seed
        self.rng = np.random.RandomState()
        self.rng.seed(self.seed)

        self.input_dim = input_dim
        self.noise_var = noise_var
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.n_features = n_features
        self.kernel = kernel
        assert kernel == "RBF" or kernel == "Matern32" or kernel == "Matern52"
        self.phi = self._compute_phi()
        self.jitter = 1e-10

        self.X = None
        self.Y = None

        # Statistics of the weights that give us random function samples
        # f(x) ~ phi(x).T @ theta, theta ~ N(theta_mu, theta_var)
        self.theta_mu = None
        self.theta_var = None

    def train(self, X, Y, n_samples):
        """
        Pre-compute all necessary variables for efficient prediction and sampling.
        """
        self.X = X
        self.Y = Y

        phi_train = self.phi(X)
        a = phi_train.T @ phi_train + self.noise_var * np.eye(self.n_features)
        a_inv = np.linalg.inv(a)
        self.theta_mu = a_inv @ phi_train.T @ Y
        self.theta_var = self.noise_var * a_inv

        # Generate handle to n_samples function samples that can be evaluated at x.
        var = self.theta_var + self.jitter * np.eye(self.theta_var.shape[0])
        var = (var + var.T) / 2
        chol = np.linalg.cholesky(var)
        self.theta_samples = self.theta_mu + chol @ self.rng.randn(self.n_features, n_samples)

    def predict(self, Xs, full_variance=False):
        raise NotImplementedError

    def sample_posterior(self, Xs):
        """
        Generate n_samples function samples from GP posterior at points Xs.
        """
        h = self.sample_posterior_handle
        return h(Xs)

    def sample_posterior_handle(self, x):
        x = np.atleast_2d(x).T if x.ndim == 1 else x
        return self.theta_samples.T @ self.phi(x).T

    def _compute_phi(self):
        """
        Compute random features.
        """
        if self.kernel == "RBF":
            w = self.rng.randn(self.n_features, self.input_dim) / self.length_scale
        elif self.kernel == "Matern32":
            w = self.rng.standard_t(3, (self.n_features, self.input_dim)) / self.length_scale
        elif self.kernel == "Matern52":
            w = self.rng.standard_t(5, (self.n_features, self.input_dim)) / self.length_scale
        b = self.rng.uniform(0, 2 * np.pi, size=self.n_features)
        return lambda x: np.sqrt(2 * self.signal_var / self.n_features) * np.cos(x @ w.T + b)


# Hyperparameter optimization experiments
def hpo(x, data, dataset):
    X = data[dataset]["X"]
    Y = data[dataset]["Y"]
    idx = np.where(np.all(x == X, axis=1))
    ret = Y[idx, :][0]
    assert np.size(ret) == x.shape[0]
    return ret


def hpo_max_min(data, dataset):
    X = data[dataset]["X"]
    Y = hpo(X, data=data, dataset=dataset)
    max_idx = np.argmax(Y)
    max_pos = X[max_idx, :].reshape(1, 2)
    max = Y[max_idx]
    min_idx = np.argmin(Y)
    min_pos = X[min_idx, :].reshape(1, 2)
    min = Y[min_idx]
    return max_pos, max, min_pos, min


def get_hpo_domain(data, dataset):
    return data[dataset]["X"]


# Rhino-function
def rhino(x):
    def bump(x, mu, sigma):
        return np.exp(-1 / 2 * (x - mu) ** 2 / sigma ** 2)

    bump1 = bump(x, mu=0.3, sigma=0.1)
    bump2 = bump(x, mu=0.7, sigma=0.01)
    rhino = 0.5 * bump1 + 3.0 * bump2

    return rhino


def rhino_max_min():
    max_pos = np.array(0.7).reshape(1, 1)
    min_pos = np.array(1.0).reshape(1, 1)
    max = rhino(max_pos)
    min = rhino(min_pos)

    return max_pos, max, min_pos, min


def rhino_translated(x, t):
    x_new = x.copy()
    # apply translation
    # clip the translations s.t. both maxima stay well in domain
    t_range = np.array([[-0.25, 0.25]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t

    return rhino(x_new)


def rhino_max_min_translated(t):
    max_pos, max, min_pos, min = rhino_max_min()

    # apply translation
    t_range = np.array([[-0.25, 0.25]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])

    max_pos = max_pos + t
    min_pos = min_pos + t

    return max_pos, max, min_pos, min


# Rhino2-function
def rhino2(x, h):
    assert .5 <= h <= 1.0

    def bump(x, mu, sigma):
        return np.exp(-1 / 2 * (x - mu) ** 2 / sigma ** 2)

    # bump1 = bump(x, mu=0.2, sigma=0.05)
    bump1 = bump(x, mu=0.2, sigma=0.1)
    # bump2 = bump(x, mu=h, sigma=0.03)
    bump2 = bump(x, mu=h, sigma=0.01)
    rhino = h * bump1 + 2 * bump2
    rhino = rhino - 1.0

    return rhino


def rhino2_max_min(h):
    max_pos = np.array(h).reshape(1, 1)
    min_pos = np.array(0.0).reshape(1, 1)
    max = rhino2(max_pos, h)
    min = rhino2(min_pos, h)

    return max_pos, max, min_pos, min
