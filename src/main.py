"""Lab 1.

Approximation of reachable set.
Considered model is x'(t) = A(t)*x(t) + C(t)v(t).
t belongs to [t0, t1]
x(t0) belongs to start set P0, which is ellipsoid
v(t) - control function, which belongs to M(t)
    which is also ellipsoid for any non-negative t
"""

import numpy as np

from approximation import solve
from operable import Operable
from plot_utils import plot_approximation_result


def main():
# pylint: disable=C0103
    """Entry point for the app."""
    # dimension
    n = 4

    # set up model parameters
    # weights
    M1 = 2
    M2 = 3

    # friction forces
    B = 4
    B1 = 3
    B2 = 5

    # stiffnesses
    K = 2
    K1 = 2
    K2 = 2

    # set up start set P0
    A0 = [1, 1, 1, 1]
    PV1 = [1, 0, 0, 0]
    PV2 = [0, 1, 0, 0]
    PV3 = [0, 0, 1, 0]
    PV4 = [0, 0, 0, 1]
    P0_SEMI_AXES = [1, 2, 3, 4]
    P0_LAMBDA = [
        [
            (0 if j != i else 1/P0_SEMI_AXES[i]**2) for j in range(n)
        ]
        for i in range(n)
    ]

    P0_EIGEN_VECTORS_MATRIX = np.transpose([PV1, PV2, PV3, PV4])
    P0_EIGEN_VECTORS_MATRIX_INV = np.linalg.inv(P0_EIGEN_VECTORS_MATRIX)

    P0 = np.dot(P0_EIGEN_VECTORS_MATRIX, P0_LAMBDA)
    P0 = np.dot(P0, P0_EIGEN_VECTORS_MATRIX_INV)

    # set up matrix of the system (i. e. matrix A(t))
    A = [
        [0, 1, 0, 0],
        [-(K + K1)/M1, -(B + B1)/M1, K/M1, B/M1],
        [0, 0, 0, 1],
        [K/M2, B/M2, -(K + K2)/M2, -(B + B2)/M2]
    ]

    C = [
        [0, 0],
        [1/M1, 0],
        [0, 0],
        [0, -1/M2]
    ]

    v0 = [
        Operable(lambda t: t),
        Operable(lambda t: t**2)
    ]
    # set up shape matrix for bounding ellipsoid for v(t)
    M = [
        [Operable(lambda t: t**2+t*16), Operable(lambda t: t**2+t*8)],
        [Operable(lambda t: t**2+t*8), Operable(lambda t: 4*t**2 + t)]
    ]

    w0 = [
        Operable(lambda t: t),
        Operable(lambda t: 2*t),
        Operable(lambda t: 0.5*t),
    ]
    #set up shape matrix for bounding ellipsoid for w(t)
    N = [
        [Operable(lambda t: 1/4), 0, 0],
        [0, Operable(lambda t: 1/9), 0],
        [0, 0, Operable(lambda t: 1)]
    ]

    # set up obesrvation equation
    G = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    y = [
        Operable(lambda t: t**2),
        Operable(lambda t: t),
        Operable(lambda t: 3*t)
    ]

    T_START = 0 # T_START - start of time
    T_END = 10  # T_END - end of time
    T_COUNT = 50  # T_COUNT - number of timestamps on [t_start, t_end]

    t_array, center, shape_matrix = solve(A, A0, P0, C, M, T_START, T_END, T_COUNT)
    plot_approximation_result(t_array, center, shape_matrix, [0, 1], 'T', 'Y1', 'Y2')


main()
