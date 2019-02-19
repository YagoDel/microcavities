# -*- coding: utf-8 -*-
import numpy as np


def depth(lst):
    """
    Returns the depth of a list by recursively checking whether the sub-elements are lists:
        depth([0,1,2,3,4]) is 1
        depth([[0,1],[2,3]]) is 2

    :param lst: list or list of lists
    :return: int. Depth of the list of lists provided
    """
    return isinstance(lst, list) and max(map(depth, lst)) + 1


def run_once(f):
    """
    Given a function that can run many times, returns a function that can run only once

    :param f: function
    :return:
    """
    def wrapper(*args, **kwargs):
        if args[1] not in wrapper.dicc.keys():
            wrapper.dicc[args[1]] = False
        if not wrapper.dicc[args[1]]:
            wrapper.dicc[args[1]] = True
            return f(*args, **kwargs)

    wrapper.dicc = {}
    return wrapper


def is_prime(integer):
    """
    Checks whether integer is divisible by any other integer smaller than itself.

    :param integer: int
    :return: bool. True if a is prime, False otherwise
    """
    return all(integer % i for i in xrange(2, integer))


def square(integer, _iter=0):
    """
    Finds the two integers that most closely make a square of the input integer.
    E.g. square(9) = (3, 3); square(10) = (5, 2); square(3) = (3, 1); square(11) = (4, 3)

    :param integer: int
    :param _iter: int. Keeps track of what level of recursion we are at to avoid diverging.
    :return: two-tuple
    """
    square_root = np.sqrt(integer)
    a = np.round(square_root) + _iter
    if is_prime(integer) and integer > 10:
        return square(integer + 1, _iter)

    if np.mod(square_root, 1) == 0:
        return int(square_root), int(square_root)
    elif np.mod(integer, a) != 0 and _iter < integer:
        return square(integer, _iter + 1)
    else:
        return np.max([int(a), int(integer / a)]), np.min([int(a), int(integer / a)])
