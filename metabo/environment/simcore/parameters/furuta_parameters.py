import metabo.environment.simcore.parameters.base_parameters as param


class FurutaParameters(param.Parameter):
    """Parameters:
       nx: dimension state-space
       nu: dimension input
       m_a: mass of arm [kg]
       l_a: length of arm [m]
       d_a: damping constant for rotation of the arm [N*s/rad]
       m_p: mass of pendulum [kg]
       l_p: length of pendulum [m]
       d_p: damping constant for rotation of the pendulum [N*s/rad]
       k_c: stiffness modelling the forced induced by the cable attachment
    """
    def __init__(self, m_a=0.095, l_a=0.112, d_a=0.0005, m_p=0.024, l_p=0.129,
                 d_p=0.00005, k_c=0.016, g=9.81):
        param.Parameter.__init__(self)
        self.ny = 2
        self.nx = 4
        self.nu = 1

        self.m_a = m_a
        self.l_a = l_a
        self.d_a = d_a

        self.m_p = m_p
        self.l_p = l_p
        self.d_p = d_p

        self.k_c = k_c
        self.g = g
