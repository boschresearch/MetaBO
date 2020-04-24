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
from sympy import symbols, Matrix
import sympy as sym
import matplotlib.pyplot as plt

from metabo.environment.simcore import utils as utl


class Environment:
    """
    Base-Class for all environments, containing simulation and model parameters
    The dynamics of a system are represented as xd = f(x, u), y = h(x).
    Every environment provides a linearization method which returns the
    system matrices Ad and Bd of a discrete state space model
    xk+1 = Ad*xk + Bd*uk.
    In addition, a plot function is provided, plotting the trajectories of
    state, input and output.
    """

    def __init__(self, simulation_parameters, model_parameters, animate,
                 estimate, noise, visualize):
        """
        :param simulation_parameters: runtime, sample-time, and substeps
        :param model_parameters: parameters describing the environment
        :param animate: enable for animation
        :param estimate: enable for state estimation
        :param noise: enable for noise addition
        :param visualize: enable for plotting
        """
        self.simulation_parameters = simulation_parameters
        self.param = model_parameters

        self._sym_state = symbols('x:%i' % self.param.nx, real=True)
        self._sym_input = symbols('u:%i' % self.param.nu, real=True)

        # ac, bc, and cc are python lambda-functions for a fast evaluation of
        # the jacobians with a specific state/action
        self.ac = sym.utilities.lambdify((self._sym_state, self._sym_input),
                                         self.fsym().jacobian(self._sym_state))

        self.bc = sym.utilities.lambdify((self._sym_state, self._sym_input),
                                         Matrix(self.fsym()).
                                         jacobian(Matrix(self._sym_input)))

        self.cc = sym.utilities.lambdify((self._sym_state, self._sym_input),
                                         Matrix(self.hsym()).
                                         jacobian(Matrix(self._sym_state)))

        self.f = sym.utilities.lambdify((self._sym_state, self._sym_input),
                                        self.fsym())

        del self._sym_input, self._sym_state

        # state and action constraints
        self.x_constraints = None
        self.u_constraints = None

        # noise
        self.transit_noise_deviation = None
        self.observation_noise_deviation = None

        # enables
        self.noise = noise
        self.estimate = estimate
        self.animate = animate
        self.visualize = visualize

        # for animation
        self.n_lines = 0
        self.coordinates_dict = dict()

        self.state_history = []
        self.input_history = []
        self.noisy_output_history = []
        self.output_history = []  # for animation only, without noise

    def next_(self, old_state, u):
        """
        returns the state for the next time step based on a given input u and a
        previous state old_state

        solves differential equation using the classical Runge Kutta (RK4)
        depending on 'substeps' RK4 executes 'substeps'
        sub steps for a higher accuracy
        """
        nx = self.param.nx
        n = self.simulation_parameters.rk_substeps
        dt = self.simulation_parameters.dt
        step_size = dt / n
        k1 = np.empty(len(old_state))
        k2 = np.empty(len(old_state))
        k3 = np.empty(len(old_state))
        k4 = np.empty(len(old_state))
        for k in range(0, n):
            k1[:] = self.f(old_state, u).flatten()
            k2[:] = self.f(old_state + step_size * k1 / 2, u).flatten()
            k3[:] = self.f(old_state + step_size * k2 / 2, u).flatten()
            k4[:] = self.f(old_state + step_size * k3, u).flatten()

            phi = k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

            old_state = old_state + step_size * phi

        self.state_history.append(old_state)
        self.input_history.append(u)
        self.output_history.append(self.get_output(old_state))
        return old_state

    def fsym(self):
        """
        returns rhs of nonlinear state space model depending on symbolic
        variables _sym_state and _sym_input. Use only for initialisation
        purposes
        :return: sympy.Matrix type
        """
        raise NotImplementedError

    def euler_next(self, state, u):
        """
        prediction of the next state using euler approximation
        :param state:  current state
        :param u:   current input
        :return:    state prediction
        """
        return state + self.simulation_parameters.dt * self.f(state, u).squeeze()

    def hsym(self):
        """
        :return: Symbolic output y = h(x), dx/dt = f(x, u)
        :rtype: sympy.Matrix
        """
        raise NotImplementedError

    def get_current_a_matrix(self, x0, u0):
        """
        returns the A matrix of a linear state space system xk+1 = A*xk + B*uk
        yk = C*xk by linearisation of the non-linear state
        space model using jacobian at x0, u0
        """
        assert type(x0) is np.ndarray, \
            "rest_position must be an array containing floats"

        a = self.ac(x0, u0)
        return utl.matrix_exponential(self.simulation_parameters.dt * a)

    def get_current_b_matrix(self, x0, u0):
        """
        returns the B matrix of a linear state space system xk+1 = A*xk + B*uk,
        yk = C*xk by linearisation of the non-linear state
        space model using jacobian at x0, u0
        """
        b = self.bc(x0, u0)
        return utl.discretize_b(self.get_current_a_matrix(
            x0, u0), b, self.simulation_parameters.dt)

    def get_current_c_matrix(self, x0, u0):
        """
        returns the C matrix of a linear state space system xk+1 = Ad*xk + Bd*uk
        yk = Cd*xk by linearisation of the non-linear state
        space model using jacobian at x0, u0
        """
        return self.cc(x0, u0)

    def linearize(self, x0, u0, dt=None):
        """
        Linearize the System around the given set-point x0 and u0
        :param x0: state at evaluation point
        :param u0: input at evaluation point
        :param dt: sample time which is used for linearization
        :return: Discrete system matrices Ad and Bd
        :rtype: tuple of numpy-arrays
        """

        ac = self.ac(x0.squeeze(), u0)
        bc = self.bc(x0.squeeze(), u0)
        if dt is None:
            sample_time = self.simulation_parameters.dt
        else:
            sample_time = dt

        ad = np.eye(ac.shape[0]) + sample_time * ac
        bd = sample_time * bc

        return ad, bd

    def observe(self, x_next):
        """
        the observe method returns the output of the System, depending on the
        configurations in the environment initialization:
        If neither estimate nor noise are activated, the state will just get
        passed through.
        If estimate is activated, the function returns the output y = h(x).
        In addition, if noise is activated too, observe returns the out
        :param x_next: current state
        :return: system-output
        """
        out = 0
        if self.noise and not self.estimate:
            out = x_next + utl.get_noise(
                self.transit_noise_deviation, self.simulation_parameters.dt)
        elif not self.noise and not self.estimate:
            out = x_next

        if self.estimate:
            if self.noise:
                out = self.get_output(x_next) + np.reshape(utl.get_noise(
                    self.observation_noise_deviation,
                    self.simulation_parameters.dt), (self.param.ny, 1))
                self.noisy_output_history.append(out)
            else:
                out = self.get_output(x_next)

        self.output_history.append(out)
        return out

    def get_output(self, state):
        """
        :param state: current state
        :return: current output
        """
        raise NotImplementedError

    def plot(self, est_state_history=None):
        """
        This function generates a plot for all state variables and for the input
        u of the system. If the state was estimated, you have to give the
        history as input.

        :param est_state_history: History of the estimated state.
        :return: None
        """
        if est_state_history is None:
            est_state_history = np.zeros(self.param.nx)
        t = np.linspace(0, (len(self.state_history) - 1)
                        * self.simulation_parameters.dt,
                        len(self.state_history))

        # convert histories to arrays
        input_history = np.asarray(self.input_history).reshape(self.param.nu, -1)
        state_history = np.asanyarray(self.state_history)
        est_state_history = np.asanyarray(est_state_history)

        x_data = np.empty([self.param.nx, len(t)])
        xest_data = np.empty([self.param.nx, len(t)])

        # plot state
        for i in range(0, self.param.nx):
            for j in range(0, len(t)):
                x_data[i, j] = state_history[j, i]

        xfig1 = plt.figure()
        x_ax = []
        for i in range(1, self.param.nx + 1):
            x_ax.append(xfig1.add_subplot(int(100 * self.param.nx + 10 + i)))

        for i in range(0, self.param.nx):
            x_ax[i].plot(np.array(t), np.array(x_data[i]), 'b')

        x_ax[0].legend(["x"]),

        # plot input
        u_data = np.empty([self.param.nu, len(t)])
        for i in range(0, self.param.nu):
            for j in range(0, len(t)):
                u_data[i, j] = input_history[i, j]

        ufig = plt.figure()
        u_ax = []
        for i in range(1, self.param.nu + 1):
            u_ax.append(ufig.add_subplot(int(100 * self.param.nu + 10 + i)))

        for i in range(0, self.param.nu):
            u_ax[i].plot(np.array(t), np.array(u_data[i]), 'r')
        u_ax[0].legend(["u"]),

        # if estimator is activated plot estimated state history
        if self.estimate:
            # plot estimated state in same the figure
            for i in range(0, self.param.nx):
                for j in range(0, len(t)):
                    xest_data[i, j] = est_state_history[j, i]

            for i in range(0, self.param.nx):
                x_ax[i].plot(np.array(t), np.array(xest_data[i]), 'r')
                x_ax[i].set_ylabel('x%i' % (i + 1))
                if i == self.param.nx - 1:
                    x_ax[i].set_xlabel('t [s]')

            x_ax[0].legend(["x", "xest"])

            # if noise is activated plot the output history too
            if self.noise:
                # Plot noisy output signals into a separate figure
                yfig = plt.figure()
                y_ax = []
                for i in range(1, self.param.ny + 1):
                    y_ax.append(yfig.add_subplot(100 * self.param.ny + 10 + i))

                y_data = np.empty([self.param.ny, len(t)])

                for i in range(0, self.param.ny):
                    for j in range(0, len(t)):
                        y_data[i, j] = self.noisy_output_history[j][i, 0]

                for i in range(0, self.param.ny):
                    y_ax[i].plot(np.array(t), np.array(y_data[i]), 'g')
                    y_ax[i].set_ylabel('y%i' % (i + 1))
                    if i == self.param.ny - 1:
                        y_ax[i].set_xlabel('t [s]')

        if not self.animate:
            plt.show()

    def generate_step_traj(self, xr, x0=None, percentage_runtime=0.5):
        """
        Generates a reference-trajectory for a step response. Initial reference
        is 0 for all state-entries
        :param xr: final value
        :param x0: initial value
        :type x0: numpy array, shape (nx, 1)
        :param percentage_runtime: percentage of runtime, when the reference
            step should appear
        :return: array containing the trajectory
        """
        ref_traj = np.zeros((int(self.simulation_parameters.runtime /
                                 self.simulation_parameters.dt), self.param.nx))

        for i in range(ref_traj.shape[0]):
            if x0 is not None:
                if i <= self.simulation_parameters.runtime / \
                        self.simulation_parameters.dt * percentage_runtime:
                    ref_traj[i] = x0.reshape(self.param.nx)

            if i > self.simulation_parameters.runtime / \
                    self.simulation_parameters.dt * percentage_runtime:
                ref_traj[i] = xr.reshape(self.param.nx)

        return ref_traj
