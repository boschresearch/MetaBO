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
from scipy.linalg import solve_discrete_are

import metabo.environment.simcore.controller.base_controller as cont


class Dlqr(cont.Controller):
    """
    Time-discrete linear-quadratic regulator

    """

    def __init__(self, environment, q, r, xr=None, ur=None, ctrl_dt=None):
        self.nx = environment.param.nx
        self.nu = environment.param.nu
        cont.Controller.__init__(self)

        if xr is None:
            xr = np.zeros(self.nx)

        if ur is None:
            ur = np.zeros(self.nu)

        # rest position and u0 is currently set to zero
        if ctrl_dt is None:
            ad, bd = environment.linearize(xr, ur)
        else:
            ad, bd = environment.linearize(xr, ur, dt=ctrl_dt)

        # get the solution of the discrete riccati equation
        p = np.array(solve_discrete_are(ad, bd, q, r))

        # calculate feedback gain
        self._K = np.dot(np.linalg.inv(
            np.array(r + np.dot(bd.T.dot(p), bd), dtype=float)),
            np.dot(bd.T.dot(p), ad))

    def calc_input(self, state, xr):
        self.output = -self._K @ (state - xr)
        return self.output
