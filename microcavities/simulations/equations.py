# -*- coding: utf-8 -*-

class Equations:
    def __init__(self, n_vars, var_types, parameters, plotting_parameters):
        """
        Base class for defining any equations that we want solved
        :param n_vars: number of variables to be solved
        :param var_types: variable type (e.g. complex, float...)
        :param parameters: dictionary of parameter values to be used inside the equations (e.g. hbar, nonlinearity...)
        :param plotting_parameters: dictionary used for displaying the solver's solution. Contains:
                    - SubplotShape: shape to be passed to the plt.subplots command
                    - xlabels, ylabels, titles: tuples of strings. The tuples should have a string for every axis
                    - Plot: tuple of dictionaries. The tuple will have as many elements as lines you want in the plots.
                    The dictionaries will contain:
                        * Position: linear index of the axis in the subplots
                        * Function: name of the pyplot function to use to plot
                        * args: args to be passed to the Function
                        * kwargs: kwargs to be passed to the Function
        """
        self.n_vars = n_vars
        if not hasattr(var_types, '__iter__'):
            var_types = (var_types,)
        if len(var_types) != self.n_vars:
            self.var_types = tuple(var_types) * self.n_vars
        else:
            self.var_types = var_types

        self.eq_parameters = dict(parameters)
        self.plotting_parameters = plotting_parameters

    def eq(self, t, psi, **parameters):
        raise NotImplemented

    @staticmethod
    def to_plot(values):
        raise NotImplemented


class single_reservoir(Equations):
    def __init__(self):
        plotting_parameters = dict(SubplotShape=(1, 1), titles=('',),
                                   xlabels=('Time',), ylabels=('a.u.',),
                                   Plot=(
                                       dict(Position=0, Function='plot', args=('o-',),
                                            kwargs=dict(label='Condensate', color='r')),
                                       dict(Position=0, Function='plot', args=('o-',),
                                            kwargs=dict(label='Reservoir', color='b'))))
        Equations.__init__(self, 2, (complex, float), PARAMETERS, plotting_parameters)

    def eq(self, t, psi, **parameters):
        if len(parameters) == 0:
            parameters = dict(self.eq_parameters)

        nonlin = parameters['Same-spin nonlinearity'] * np.abs(psi[0]) ** 2 + \
                 parameters['Same-spin reservoir nonlinearity'] * psi[1] + \
                 parameters['Pump'] * parameters['Pump nonlinearity']
        gain = 0.5 * (parameters['Condensation rate'] * psi[1] - parameters['Saturation'] * np.abs(psi[1]) ** 2 -
                      parameters['polariton_decay'])

        condensate = (gain - 1j * nonlin) * psi[0]
        reservoir = parameters['Pump'] - parameters['exciton_decay'] * psi[1] - \
                    psi[1] * (parameters['Condensation rate'] * np.abs(psi[0]) ** 2)

        return np.array([condensate, reservoir])

    @staticmethod
    def to_plot(values):
        return np.abs(values[0]) ** 2, np.abs(values[1])
