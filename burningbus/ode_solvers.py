from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import numpy as np

def test_eq(t, y):
    dydt = - y
    return dydt

# y0 = [10]

def event1(t, y):
    if t % 10 > 5:
        return 1.
    else:
        return -1.

# fig = plt.figure()
# plt.ion()

def event2(t, y):
    if t % 10 > 5:
        return 1.
    else:
        return -1.

def my_event():
    print 'Hi'

def append_scipy_results(a, b):
    new_result = OdeResult()
    for key in a.keys():
        if type(a[key]) == np.ndarray:
            new_result[key] = np.append(a[key], b[key], -1)
        elif type(a[key]) == list:
            new_result[key] = a[key] + b[key]
        elif type(a[key]) == tuple:
            new_result[key] = a[key] + (b[key], )
        else:
            new_result[key] = (a[key], b[key])
    return new_result

def my_scipy_solver(fun, t_evals, y0, *args, **kwargs):
    full_results = None

    t0 = t_evals[0]
    for t1 in t_evals[1:]:
        solver = solve_ivp(fun, [t0, t1], y0, *args, **kwargs)
        y0 = solver.y[:, -1]
        t0 = solver.t[-1]
        # my_event()
        if full_results is None:
            full_results = solver
        else:
            full_results = append_scipy_results(full_results, solver)
    return full_results

# solver = solve_ivp(test_eq, [0, 100], y0, events=event1)
# solver2 = my_scipy_solver(test_eq, range(0, 101, 10), y0)
#
# plt.semilogy(solver.t, solver.y[0], 'b.-');
# plt.semilogy(solver2.t, solver2.y[0], 'r.-'); plt.show()

# Start/stop like here: https://github.com/scipy/scipy/issues/10070 and plot at the steps
# Also make the simplest RK4 solver and compare the speeds

# Compare for much larger arrays
# Fix the step size in the scipy ones
# Use the scipy RK45 step in my loop

def single_step(fun, t, init_cond, time_step):
    k1 = fun(t, init_cond)
    k2 = fun(t + time_step / 2, init_cond + k1 * time_step / 2)
    k3 = fun(t + time_step / 2, init_cond + k2 * time_step / 2)
    k4 = fun(t + time_step, init_cond + k3 * time_step)

    return t + time_step, init_cond + (k1 + 2 * k2 + 2 * k3 + k4) * time_step / 6

def loop(fun, tfinal, y0, time_step):
    y_results = np.array([y0])
    t_results = [0]
    t = 0
    while t < tfinal:
        t, y0 = single_step(fun, t, y0, time_step)
        # y_results += [y0]
        y_results = np.append(y_results, [y0], 0)
        t_results += [t]
    return np.array(t_results), np.array(y_results).transpose()

# t, y = loop(test_eq, 100, y0[0], 0.1)
#
# plt.semilogy(t, y, 'r.-'); plt.show()


y0 = np.array([1e12]*1)

if __name__ == '__main__':
    import timeit
    print("Simple RK4: %g" % timeit.timeit("loop(test_eq, 100, y0, 0.1)", number=100,
    setup="from __main__ import loop, test_eq, y0"))

    print("Scipy with events: %g" % timeit.timeit("solve_ivp(test_eq, [0, 100], y0, events=event1, first_step=0.2, max_step=0.1)", number=100,
    setup="from __main__ import solve_ivp, test_eq, event1, y0"))

    print("Scipy w/o events: %g" % timeit.timeit("solve_ivp(test_eq, [0, 100], y0, first_step=0.2, max_step=0.1)", number=100,
    setup="from __main__ import solve_ivp, test_eq, y0"))

    print("Scipy start-stop: %g" % timeit.timeit("my_scipy_solver(test_eq, range(0, 101, 10), y0, first_step=0.2, max_step=0.1)", number=100,
    setup="from __main__ import my_scipy_solver, test_eq, y0"))
    # solver = solve_ivp(test_eq, [0, 100], y0, events=event1)
    # solver2 = my_scipy_solver(test_eq, range(0, 101, 10), y0)
