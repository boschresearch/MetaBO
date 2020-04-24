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
from sympy import sin, cos, Matrix

import metabo.environment.simcore.environment.base_environment as env
import metabo.environment.simcore.parameters.furuta_parameters as parameter


class FurutaPendulum(env.Environment):
    """
    x = [q, dq]^T, with q = [theta_arm, theta_pendulum]
    default parameters are related to the Quanser Cube Servo 2:
    https://www.quanser.com/products/qube-servo-2/
    Initial position (x = [0, 0, 0, 0]^T) is at the upper equilibrium point
    """

    def __init__(self, simulation_parameters, mass_arm=0.095,
                 mass_pendulum=0.024, damp_arm=0.0005, length_arm=0.112,
                 length_pendulum=0.129, damp_pendulum=0.00005, noise=False,
                 animate=True, estimate=True, visualize=True):
        # set up the parameters
        self.param = parameter.FurutaParameters(m_a=mass_arm, m_p=mass_pendulum,
                                                d_a=damp_arm, d_p=damp_pendulum,
                                                l_a=length_arm,
                                                l_p=length_pendulum)

        # pass through the environment parameters for environment setup
        env.Environment.__init__(self,
                                 simulation_parameters=simulation_parameters,
                                 model_parameters=self.param, animate=animate,
                                 estimate=estimate, noise=noise,
                                 visualize=visualize)

        # constraints
        x_upper_lim = np.array([np.pi / 2, np.inf, np.inf, np.inf])
        x_lower_lim = -1 * x_upper_lim
        u_upper_lim = np.array([np.inf])
        u_lower_lim = -1 * u_upper_lim
        self.x_constraints = np.vstack([x_upper_lim, x_lower_lim])
        self.u_constraints = np.vstack([u_upper_lim, u_lower_lim])

        # deviation of noise
        self.transit_noise_deviation = np.array([0.0, 0.0, 0., 0.])
        self.observation_noise_deviation = np.array([0.0, 0.0])

        # FurutaPendulum can be represented by 2 lines
        self.n_lines = 2
        for i in range(self.n_lines):
            self.coordinates_dict[i] = []

    def fsym(self):
        # load parameters
        # l_p here is the distance to center of mass
        l_a, l_p, m_a, m_p, d_a, d_p, k_a, g = (self.param.l_a,
                                                self.param.l_p / 2,
                                                self.param.m_a, self.param.m_p,
                                                self.param.d_a, self.param.d_p,
                                                self.param.k_c, self.param.g)

        # compute the pendulum's inertia, inertia along the roll axis is
        # neglected
        j2 = 0
        j0 = (m_a * l_a ** 2) / 12
        j1 = (m_p * l_p ** 2) / 12

        u1 = self._sym_input[0]
        x = self._sym_state

        # mass matrix
        m = Matrix([[1.0 * j0 + 2 * j2 + m_p * (l_a ** 2 + l_p ** 2 * sin(x[1]) ** 2),
                     -l_a * l_p * m_p * cos(x[1])], [-l_a * l_p * m_p * cos(x[1]),
                                                     2 * j1 + 1.0 * l_p ** 2 * m_p]])

        # coriolis accelerations and all this stuff
        n = Matrix([[x[3] * l_p * m_p * (2 * x[2] * l_p * cos(x[1]) + x[3] * l_a) * sin(x[1])],
                    [-l_p * m_p * (x[2] * (x[2] * l_p * cos(x[1]) + x[3] * l_a)
                                   + 2 * g) * sin(x[1])]])

        # elasticity and damping forces
        fda = -d_a * Matrix([[x[2]], [0]])
        fdp = -d_p * Matrix([[0], [x[3]]])
        fc = -k_a * Matrix([[x[0]], [0]])

        # torque vector
        u = Matrix([[u1], [0]])

        # build up the equations for accelerations
        qdd = (m ** -1) * (fda + fdp + fc + u - n)

        # concatenate kinetics to end up with a state space model
        dx = qdd.row_insert(0, Matrix([[x[2]], [x[3]]]))

        return dx

    def hsym(self):
        # define output-equation
        return Matrix([[self._sym_state[0]], [self._sym_state[1]]])

    def get_output(self, state):
        return np.array([state[0], state[1]]).reshape(2, 1)
