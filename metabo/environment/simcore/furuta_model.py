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


import sympy as sym
from sympy import symbols, Matrix, sin, cos

'''
Script to derive the dynamics of a furuta pendulum

xd = f(x,u) with f(x,u) = M^-1*(sum(F)-N) 
derivation Lagrangian mechanics
Frames: when you have the Qube in front of you, the pendulum aiming at yourself,
the x axis aims at you, the y-axis to the top and the z-axis to the left.
In the initial position (pendulum is in the upper equilibrium point) all frames 
have the same orientation 
'''

# define symbolic parameters
J1, J2, J3 = symbols("J1, J2, J3")
J0 = symbols("J0")
m1, m2 = symbols("m1, m2")
l1, l2 = symbols("l1, l2")
g = symbols("g")
theta1, theta2 = symbols("theta1, theta2")
dtheta1, dtheta2 = symbols("dtheta1, dtheta2")
# the inertia of the pendulum for rotation around x- or z axis are the same
J = Matrix.diag((J1, J2, J1))

# generalized coordinates
q = Matrix([[theta1], [theta2]])
dq = Matrix([[dtheta1], [dtheta2]])

# kinetic energy of joint arm. has only rotational kinetic energy
k1 = Matrix([[0.5 * J0 * dtheta1 ** 2]])

# kinetics of pendulum
omega2 = Matrix([[cos(theta1) * dtheta2], [dtheta1], [sin(theta1) * dtheta2]])
v2 = l1 * dtheta1 * Matrix([[-sin(theta1)], [0], [-cos(theta1)]]) \
     + l2 * dtheta1 * Matrix([[cos(theta1) * sin(theta2)], [0],
                              [-sin(theta1) * sin(theta2)]]) \
     + dtheta2 * l2 * Matrix([[sin(theta1) * cos(theta2)], [-sin(theta2)],
                              [cos(theta2) * cos(theta1)]])

# kinetic energy (rotational and translational)
k2r = sym.simplify(omega2.T * J * omega2)
k2t = 0.5 * m2 * sym.simplify(v2.T * v2)

# potential energy of pendulum
p = Matrix([[m2 * 2 * l2 * g * cos(theta2)]])

# total kinetic energy
K = k1 + k2t + k2r

# calculate Lagrangian
L = K - p

# partial derivative to dq
Kp = sym.simplify(K.jacobian(dq))
kpp = sym.simplify(Kp.jacobian(q))
print("unknown k = ", kpp)

M = sym.simplify(Kp.jacobian(dq))
N = sym.simplify(-sym.simplify(L.jacobian(q)).T + Matrix(kpp.dot(dq)))
print("M = ", M)
print("N = ", N)
