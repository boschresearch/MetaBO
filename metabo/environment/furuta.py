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
# furuta.py
# Implementation of the objective function for the Furuta control experiment in simulation.
# ******************************************************************

import numpy as np
import warnings

from metabo.environment.simcore.controller import dlqr
from metabo.environment.simcore.environment import furuta as env_furuta
from metabo.environment.simcore.parameters import simulation_parameters as simparam


def init_furuta_simulation(mass_pendulum, mass_arm, length_pendulum, length_arm, damp_arm=0.0, damp_pendulum=0.0):
    sim_param = simparam.SimulationParameters(runtime=3.00, dt=0.01)
    environment = env_furuta.FurutaPendulum(sim_param, noise=False, estimate=False, animate=False, visualize=False,
                                            mass_arm=mass_arm, mass_pendulum=mass_pendulum,
                                            length_arm=length_arm, length_pendulum=length_pendulum,
                                            damp_arm=damp_arm, damp_pendulum=damp_pendulum)

    x0 = np.array([0.0, np.deg2rad(10), 0.0, 0.0])  # Initial state
    xr = np.array([0.0, 0.0, 0.0, 0.0])  # Reference state

    # Setup controller
    q = np.diag([5, 5, 0.1, 0.1])
    r = np.diag([0.1])
    ctrl = dlqr.Dlqr(environment, q, r, xr)

    proc_noise = 0 * np.array([0.01, np.deg2rad(0.1), 0.0, 0.0])

    return sim_param, environment, x0, xr, q, r, ctrl, proc_noise


def compute_cost(X, xr, U, q, r, runtime):
    max_computation_length = X.shape[0] - 1
    # -1 as the last X is already corrupted and last U was not computed anymore:
    n_computed = np.min((np.sum(~np.isnan(X[:, 0])),
                         np.sum(~np.isnan(X[:, 1])),
                         np.sum(~np.isnan(X[:, 2])),
                         np.sum(~np.isnan(X[:, 3])))) - 1
    assert n_computed >= 0

    cost = np.einsum("ik,kl,il", X[:n_computed, :] - xr, q, X[:n_computed, :] - xr) \
           + np.einsum("ik,kl,il", U[:n_computed], r, U[:n_computed, :])

    # if we got only nans, we want to have cost_per_second = 1e5
    cost += (runtime * 1e5) * (max_computation_length - n_computed) / max_computation_length

    cost_per_second = cost / runtime
    return cost_per_second


def furuta_simulation(init_tuple, params, D, pos):
    sim_param, environment, x0, xr, q, r, ctrl, proc_noise = init_tuple
    if D != 0 and params.ndim != 2:
        params = params.reshape(-1, D)
    elif D == 0:
        params = np.array([[]])

    logcosts = []
    for param in params:
        # use the parameters given in params instead of the LQR parameters
        assert param.size == D
        if D != 0:
            K = ctrl._K
            K[0, pos] = param

        # perform simulation
        n_simsteps = int(sim_param.runtime / sim_param.dt)
        u = np.zeros([n_simsteps, environment.param.nu]) * np.nan
        x = np.zeros([n_simsteps + 1, environment.param.nx]) * np.nan
        x[0, :] = x0
        for k in np.arange(n_simsteps):
            u[k, :] = ctrl.calc_input(x[k, :], xr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                next_state = environment.euler_next(x[k, :], u[k, :]) + proc_noise * np.random.randn(4, )
            x[k + 1, :] = next_state
            if np.isnan(next_state).any() or (np.abs(next_state) > 100.0).any():
                break

        logcosts.append(np.log10(compute_cost(x, xr, u, q, r, runtime=sim_param.runtime)))

    return np.array(logcosts)
