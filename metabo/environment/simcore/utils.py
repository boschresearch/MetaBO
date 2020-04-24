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

import numpy as np
from scipy.linalg import expm


def matrix_exponential(a):
    """
    Computing the matrix exponential of a matrix A using taylor-series.
    """
    exp = expm(a)
    return exp


def discretize_b(ac, bc, sample_time):
    """
    approximates the integral int(e^(A*nu)*B)d_nu, nu=0...sample_time using step
    size 'step'
    """
    nx = ac.shape[0]
    nu = bc.shape[1]
    tmp = np.hstack([ac, bc])
    tmp2 = np.hstack([np.zeros((nu, nx)), np.zeros((nu, nu))])

    exponent = sample_time * np.vstack((tmp, tmp2))
    exp = expm(exponent)
    bd = exp[0:nx, nx:(nx + nu)]

    return bd


def solve_dariccati(ad, bd, q, r, epsilon=0.001):
    """
    Solving discrete algebraic riccati equation (convergence of finite-horizon
    equation),
    just for testing, use np.array(solve_discrete_are(ad, bd, q, r)) instead
    """
    tmp = np.zeros(ad.shape)
    p = q
    while np.any(abs(p - tmp) / np.linalg.norm(p, ord=1) >= epsilon):
        # break up when (P(k)-P(k-1))/norm(P(k)) < epsilon
        tmp = p
        k1 = np.dot(ad.T, np.dot(tmp, ad))
        k2 = np.dot(ad.T, np.dot(tmp, bd))
        k3 = np.array(r + np.dot(bd.T, np.dot(tmp, bd)),
                      dtype=float)
        k4 = np.dot(bd.T, np.dot(tmp, ad))
        p = k1 - np.dot(k2, np.dot(np.linalg.inv(k3), k4)) + q
        print(p[0][0])  # check convergence

    return p


def get_noise(cov_diag, sample_time):
    cov = np.diag(cov_diag * sample_time)
    noise = np.zeros(cov.shape[0])
    noise = np.random.multivariate_normal(np.zeros(len(noise)), cov)
    return noise.reshape(len(noise), 1)


def rotmat_x(angle):
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


def rotmat_y(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def rotmat_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
