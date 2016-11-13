"""This module provides functionality for finding approximation of reachable set.

Considered model is x'(t) = A(t)*x(t) + C(t)u(t).
t belongs to [t0, t1]
x(t0) belongs to start set M0, which is ellipsoid
u(t) - control function, which belongs to U(t)
    which is also ellipsoid for any non-negative t
"""

# pylint: disable=C0103
# pylint: disable=R0913

import math

import numpy as np
from scipy.integrate import odeint

from operable import Operable


def matrix_to_array(matrix):
    """Convert matrix to array representation.

    Used to convert matrix differential equation to system of differential equations.
    Returns array of size n*m where n - number of rows in matrix,
    m - number of columns in matrix."""
    rows, cols = np.shape(matrix)
    return np.reshape(matrix, (rows*cols))


def array_to_matrix(array, rows, cols):
    """Convert array that represents matrix to matrix.

    Used to convert system of differential equations back to matrix form.
    Returns matrix of shape (rows, cols)."""
    return np.reshape(array, (rows, cols))


def solution_to_matrix_form(solution, rows, cols, timestamps_count):
    """Convert numerical solution of system of ODE to matrix form.

    Initially solution is represented in two dimensional form, where
    each row corresponds to certain timestamp and value is array of size
    rows*cols.
    This will be transformed into representation,
    where value for each timestamp will be matrix of shape of ellipsoid
    that corresponds to certain timestamp.
    """
    return np.reshape(solution, (timestamps_count, rows, cols))


def get_value_from_discrete_solution(solution, time, timepoints):
    """Returns matrix corresponding to time by given discrete solution.

    solution - discrete solution
    time - given timepoint
    timepoints - discrete time interval"""
    closest = min(range(len(timepoints)), key=lambda i: math.fabs(timepoints[i] - time))
    return solution[closest]


def calc_in_time_point(matrix, time):
    """Calculates matrix in given time point"""
    op = np.vectorize(lambda el: Operable(el)(time), otypes=[np.float])
    return op(matrix)

def find_R(A, P0, C, M, G, N, n, t_array):
    """Returns matrix R(s)."""
    G_t = np.transpose(G)
    GNG = np.dot(G_t, N)
    GNG = np.dot(GNG, G)
    def diff(func, time):
        """Describes system of equations.

        Returns array of values - value of system in given time point."""
        R = array_to_matrix(func, n, n)
        AR = np.dot(A, R)
        AR = calc_in_time_point(AR, time)

        A_t = np.transpose(A)
        RA = np.dot(R, A_t)
        RA = calc_in_time_point(RA, time)

        M_inv = calc_in_time_point(M, time)
        M_inv = np.linalg.inv(M_inv)

        C_time = calc_in_time_point(C, time)
        CMC = np.dot(C_time, M_inv)
        CMC = np.dot(CMC, np.transpose(C_time))

        GNG_time = calc_in_time_point(GNG, time)
        RGNGR = np.dot(R, GNG_time)
        RGNGR = np.dot(RGNGR, R)

        res = np.add(AR, RA)
        res = np.add(res, CMC)
        res = np.subtract(res, RGNGR)
        # res = calc_in_time_point(res, time)
        res = matrix_to_array(res)
        return res
    initial_condition = np.linalg.inv(P0)
    initial_condition = matrix_to_array(initial_condition)
    sol = odeint(diff, initial_condition, t_array)
    shape_matrix = solution_to_matrix_form(sol, n, n, np.shape(t_array)[0])
    return shape_matrix


def find_x(A, a, C, G, N, y, v0, w0, R, t_array):
    """Returns estimation of x."""
    y_w0_dif = np.subtract(y, w0)
    GN = np.dot(np.transpose(G), N)
    Cv0 = np.dot(C, v0)
    def system(x, time):
        """Describes system of equations."""
        A_time = calc_in_time_point(A, time)
        Ax = np.dot(A_time, x)

        G_time = calc_in_time_point(G, time)
        Gx = np.dot(G_time, x)

        R_time = get_value_from_discrete_solution(R, time, t_array)

        GN_time = calc_in_time_point(GN, time)
        RGN = np.dot(R_time, GN_time)

        y_w0_dif_time = calc_in_time_point(y_w0_dif, time)
        temp = np.subtract(y_w0_dif_time, Gx)

        RGN = np.dot(RGN, temp)

        Cv0_time = calc_in_time_point(Cv0, time)

        res = np.add(Ax, RGN)
        res = np.add(res, Cv0_time)
        return res
    sol = odeint(system, a, t_array)
    return sol

def find_r(N, y, G, w0, mu, x, t_array):
    """Returns function r(t)."""
    y_w0_dif = np.subtract(y, w0)

    def system(k, time):
        """Represents system of equations."""
        x_time = get_value_from_discrete_solution(x, time, t_array)
        y_w0_dif_time = calc_in_time_point(y_w0_dif, time)
        G_time = calc_in_time_point(G, time)
        N_time = calc_in_time_point(N, time)

        Gx = np.dot(G_time, x_time)

        vec = np.subtract(y_w0_dif_time, Gx)
        vec_N = np.dot(vec, N_time)
        el = np.dot(vec_N, vec)
        return el
    sol = odeint(system, 0, t_array)
    return [mu**2 - el[0] for el in sol] # here should be minus

def solve(A, a, P0, C, M, N, y, G, v0, w0, mu, n, t_start, t_end, t_count):
    t_array = np.linspace(t_start, t_end, t_count)
    R = find_R(A, P0, C, M, G, N, n, t_array)
    x = find_x(A, a, C, G, N, y, v0, w0, R, t_array)
    r = find_r(N, y, G, w0, mu, x, t_array)
    P = np.empty_like(R)
    for i in range(len(t_array)):
        P[i] = np.dot(r[i], R[i]) # maybe 1/r[i] ???
    return t_array, x, P, get_error_func(R, r, t_array)

def get_error_func(R, r, timestamps):
    """Returns error func.

    Error function gives estimate in each given timepoint
    how approximation differs from real solution."""
    def error_func(time):
        """Error function."""
        R_time = get_value_from_discrete_solution(R, time, timestamps)
        r_time = get_value_from_discrete_solution(r, time, timestamps)
        max_eig = max(np.linalg.eigvalsh(R_time))
        return math.sqrt(r_time * max_eig)
    return error_func
     