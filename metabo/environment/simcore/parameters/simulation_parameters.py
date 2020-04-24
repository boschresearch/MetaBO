import metabo.environment.simcore.parameters.base_parameters as param


class SimulationParameters(param.Parameter):
    """
    SimulationParameters contains the parameters for the simulation
    configuration, like runtime, sample time
    Additionally it contains a parameter (substeps) determining the number of
    sub steps
    per time step applied by the ode solver
    """
    def __init__(self, runtime, dt, rk_substeps=5):
        param.Parameter.__init__(self)
        self.runtime = runtime
        self.dt = dt
        self.rk_substeps = rk_substeps
        self.n_simsteps = int(runtime/dt)
