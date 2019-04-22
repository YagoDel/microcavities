# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint, complex_ode, ode
import h5py, time, warnings
import matplotlib.pyplot as plt

from nplab.utils.log import create_logger


def random(length, types, shape=1, amplitude=0.1):
    result = ()
    for ii in range(length):
        if types[ii] == complex:
            result += (amplitude * np.exp(1j * 2 * np.pi * np.random.random((shape, 1))), )
        elif types[ii] == float:
            result += (amplitude * np.random.random((shape, 1)), )
        else:
            raise TypeError('Type not recognised: ', types[ii])
    return result


class SolverBaseClass:
    def __init__(self, equations, solver_parameters=None,
                 plot='No', save='No', save_parameters=None, **kwargs):
        """

        :param equations: instance of simulations.equations.Equations
        :param solver_parameters: dictionary containing 'Time step', 'Total time' and 'Sampling time'
        :param plot: string with either 'Dynamic', 'Final' or 'No'
        :param save: bool
        :param saving_parameters:
        """
        self._logger = create_logger('Simulations.Solvers')
        if 'loggerLevel' in kwargs:
            self._logger.setLevel(kwargs['loggerLevel'])

        self.n_vars = equations.n_vars
        var_types = equations.var_types
        if not hasattr(var_types, '__iter__'):
            var_types = (var_types, )
        if len(var_types) != self.n_vars:
            self.var_types = tuple(var_types) * self.n_vars
        else:
            self.var_types = var_types

        self.equations = equations

        if solver_parameters is None:
            self.solver_parameters = dict()
        else:
            self.solver_parameters = dict(solver_parameters)
        if 'realisations' not in self.solver_parameters:
            self.solver_parameters['realisations'] = 1

        if 'Time step' not in self.solver_parameters.keys():
            self.solver_parameters['Time step'] = 0.1
        if 'Total time' not in self.solver_parameters.keys():
            self.solver_parameters['Total time'] = 2000
        if 'Sampling time' not in self.solver_parameters.keys():
            self.solver_parameters['Sampling time'] = 25
        self.solver_parameters['Number of Steps'] = int(self.solver_parameters['Total time'] / self.solver_parameters['Time step'])

        self.save = save
        self.plott = plot

        if plot in ['Dynamic', 'Final']:
            self.figure = None
            self.plotting_parameters = equations.plotting_parameters
            dummy = [(x, x not in self.plotting_parameters) for x in ['SubplotShape', 'Plot']]
            for dum in dummy:
                if dum[1]:
                    self._logger.warn('Cannot plot with given %s' % dum[0])
                    self.plott = 'No'

        if save in ['Dynamic', 'Final']:
            if save_parameters is None:
                self.save_parameters = dict()
            else:
                self.save_parameters = dict(save_parameters)
            if 'Filename' not in self.save_parameters:
                self.save_parameters['Filename'] = r'C:\Users\Hera\Desktop/SimulationsDefaultSave.h5'
            if 'Start save time' not in self.save_parameters:
                self.save_parameters['Start save time'] = 0
            self.save_parameters['Total time'] = self.solver_parameters['Total time']
            self.save_parameters['Step Size'] = self.solver_parameters['Time step']

    def successful(self, values):
        if 'nan' in str(values):
            return False
        else:
            return True

    def single_step(self, t, init_cond):
        raise NotImplementedError

    def run(self, init_cond=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            self._logger.info('Starting simulation')
            if init_cond is None:
                init_cond = random(self.n_vars, self.var_types, self.solver_parameters['realisations'])
            save_index = 0
            t = 0
            values = tuple(init_cond)
            self.values = map(np.array, values)

            '''Setting up the files, if we want to save continuously'''
            if self.save == 'Dynamic':
                file_extension = self.save_parameters['Filename'].split('.')[-1]
                if file_extension == 'txt':
                    save_file = open(self.save_parameters['Filename'], 'a')
                elif file_extension == 'h5':
                    save_file = h5py.File(self.save_parameters['Filename'], 'a')
                    if 'DatasetName' not in self.save_parameters:
                        current_names = save_file.keys()
                        lastN = 0
                        for name in current_names:
                            if 'RungeKuttaSolution' in name:
                                if int(name[-4:]) > lastN:
                                    lastN = int(name[-4:])
                        name = 'RungeKuttaSolution%04d' % (lastN + 1)
                    else:
                        name = str(self.save_parameters['DatasetName'])

                    df = save_file.create_dataset(name, (
                    (self.solver_parameters['Number of Steps'] * self.solver_parameters['Time step']) /
                    self.solver_parameters['Sampling time'], len(values) + 1), complex)
                    for parameter in self.save_parameters.keys():
                        df.attrs.create(parameter, self.save_parameters[parameter])  # = dict(self.save_parameters)
                    for parameter in self.equations.eq_parameters.keys():
                        df.attrs.create(parameter, self.equations.eq_parameters[parameter])
                else:
                    raise TypeError('Did not recognise file extension ', file_extension)

            '''LOOP'''
            t0 = time.time()
            for ii in range(self.solver_parameters['Number of Steps']):
                if self.successful(values):
                    t, values = self.single_step(t, values)
                else:
                    self._logger.warn('Diverged')
                    if self.plott == 'Final':
                        self.plot(self.values)
                    if self.save == 'Final':
                        file_extension = self.save_parameters['Filename'].split('.')[-1]
                        times = np.array([np.arange(0, t, self.solver_parameters['Sampling time'])])

                        if file_extension == 'txt':
                            save_file = open(self.save_parameters['Filename'], 'a')
                            values = np.asarray(self.values)
                            for index in range(np.prod(times.shape)):
                                if times[0, index] >= self.save_parameters['Start save time']:
                                    save_file.write(' '.join(map(str, (times[0, index],) + tuple(values[:, index]))) + '\n')
                        elif file_extension == 'h5':
                            save_file = h5py.File(self.save_parameters['Filename'], 'a')
                            if 'DatasetName' not in self.save_parameters:
                                current_names = save_file.keys()
                                lastN = 0
                                for name in current_names:
                                    if 'RungeKuttaSolution' in name:
                                        if int(name[-4:]) > lastN:
                                            lastN = int(name[-4:])
                                name = 'RungeKuttaSolution%04d' % (lastN + 1)
                            else:
                                name = str(self.save_parameters['DatasetName'])

                            save_array = np.append(times, np.asarray(self.values), axis=0)
                            save_array = save_array[..., save_array[0] >= self.save_parameters['Start save time']]
                            df = save_file.create_dataset(name, save_array.shape, save_array.dtype, save_array)
                            for parameter in self.save_parameters.keys():
                                df.attrs.create(parameter, self.save_parameters[parameter])
                            for parameter in self.equations.eq_parameters.keys():
                                df.attrs.create(parameter, self.equations.eq_parameters[parameter])
                        else:
                            raise TypeError('Did not recognise file extension ', file_extension)

                    if self.save in ['Dynamic', 'Final']:
                        save_file.close()

                    return t, values

                if (ii+1) % int(self.solver_parameters['Number of Steps']/4) == 0:
                    self._logger.info('%g%% after %g' % (np.round(100 * float(ii) / self.solver_parameters['Number of Steps']), time.time() - t0))
                if self.solver_parameters['Sampling time'] is not None:
                    if np.abs(self.solver_parameters['Sampling time'] - t % self.solver_parameters['Sampling time']) < self.solver_parameters['Time step']:
                        self.values = map(lambda x, y: np.append(x, y, 1), self.values, values)

                        if self.plott == 'Dynamic':
                            self.plot(self.values)
                        if self.save == 'Dynamic' and t >= self.save_parameters['Start save time']:
                            if file_extension == 'txt':
                                save_file.write(' '.join(map(str, (t,) + tuple(values))) + '\n')
                            elif file_extension == 'h5':
                                df[save_index] = (t,) + tuple(values)
                                save_index += 1

            if self.plott == 'Final':
                self.plot(self.values)
            if self.save == 'Final':
                file_extension = self.save_parameters['Filename'].split('.')[-1]
                times = np.linspace(0, self.save_parameters['Total time'],
                                    self.save_parameters['Total time']/self.solver_parameters['Sampling time'])

                if file_extension == 'txt':
                    save_file = open(self.save_parameters['Filename'], 'a')
                    values = np.asarray(self.values)
                    for index in range(np.prod(times.shape)):
                        if times[0, index] >= self.save_parameters['Start save time']:
                            save_file.write(' '.join(map(str, (times[0,index],) + tuple(values[:,index]))) + '\n')
                elif file_extension == 'h5':
                    save_file = h5py.File(self.save_parameters['Filename'], 'a')
                    if 'DatasetName' not in self.save_parameters:
                        current_names = save_file.keys()
                        lastN = 0
                        for name in current_names:
                            if 'RungeKuttaSolution' in name:
                                if int(name[-4:]) > lastN:
                                    lastN = int(name[-4:])
                        name = 'RungeKuttaSolution%04d' % (lastN + 1)
                    else:
                        name = str(self.save_parameters['DatasetName'])

                    if 'Time' not in save_file.keys():
                        save_file.create_dataset('Time', times.shape, times.dtype, times)
                    save_array = np.asarray(self.values)  # np.append(times, np.asarray(self.values), axis=1)

                    save_array = np.take(save_array, np.nonzero(times >= self.save_parameters['Start save time'])[0], -1)
                    df = save_file.create_dataset(name, save_array.shape, save_array.dtype, save_array)

                    for parameter in self.save_parameters.keys():
                        df.attrs.create(parameter, self.save_parameters[parameter])
                    for parameter in self.equations.eq_parameters.keys():
                        df.attrs.create(parameter, self.equations.eq_parameters[parameter])
                else:
                    raise TypeError('Did not recognise file extension ', file_extension)

            if self.save in ['Dynamic', 'Final']:
                save_file.close()

            return t, values

    def plot(self, values=None):
        if values is None:
            values = self.values
        for realisation in range(self.solver_parameters['realisations']):
            self.figure = None
            if self.figure is None:
                first_time = True
                self.figure, self.axes = plt.subplots(*self.plotting_parameters['SubplotShape'])
                if not hasattr(self.axes, '__iter__'):
                    self.axes = (self.axes,)

                if len(self.axes) != len(self.plotting_parameters['Plot']):
                    self.axes = map(lambda x: (x, ), self.axes)
                    positions = ()
                    for plot in self.plotting_parameters['Plot']:
                        if plot['Position'] in positions:
                            self.axes[plot['Position']] += (self.axes[plot['Position']][0].twinx(),)
                        else:
                            positions += (plot['Position'], )
            else:
                first_time = False


            plot_values = self.equations.to_plot([val[realisation] for val in values])
            indices = [0] * np.prod(self.plotting_parameters['SubplotShape'])
            self.lines = [(),] * np.prod(self.plotting_parameters['SubplotShape'])
            for plot in self.plotting_parameters['Plot']:
                index = plot['Position']
                ax = self.axes[index][indices[index]]
                line = getattr(ax, plot['Function'])(plot_values[np.sum(indices)], *plot['args'], **plot['kwargs'])
                self.lines[index] += (line[0],)
                for tl in ax.get_yticklabels():
                    tl.set_color(plot['kwargs']['color'])
                indices[index] += 1
            if first_time:
                index = 0
                for ax in self.axes:
                    if 'xlabels' in self.plotting_parameters:
                        ax[0].set_xlabel(self.plotting_parameters['xlabels'][index])
                    if 'ylabels' in self.plotting_parameters:
                        ax[0].set_ylabel(self.plotting_parameters['ylabels'][index])
                    if 'titles' in self.plotting_parameters:
                        ax[0].set_title(self.plotting_parameters['titles'][index])

                    labels = [l.get_label() for l in self.lines[index]]
                    ax[-1].legend(self.lines[index], labels, bbox_to_anchor=(0.2, 1.1))
                    index += 1
            if plt.fignum_exists(self.figure.number):
                plt.draw()
                plt.pause(0.1)
