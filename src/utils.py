"""This module provides different utilities."""

import numpy as np

def project_ellipsoid_to_subspace(center, shape_matrix, initial_dimension, projection_coordinates):
    """Projects n-dimesnaional ellipsoid into 2d plane.

    shape_matrix - shape matrix of ellipsoid
    initial_dimension - dimension of ellipsoid
    projection_coordinates - numbers of coordinates that will form projection subspace.
    For e. g. if projection_coordinates=[0, 1] and dimension = 3, than projection
    onto xy plane will be computed"""
    # basis of subspace, ellipsoid is being projected
    projection_coordinates.sort()
    projection_dimension = len(projection_coordinates)

    pr_basis = [
        [1 if i == projection_coordinates[j] else 0 for j in range(projection_dimension)]
        for i in range(initial_dimension)
    ]

    # basis of complementary space
    kernel_coordinates = list(set(range(initial_dimension)) - set(projection_coordinates))
    kernel_dimension = len(kernel_coordinates)

    kernel_basis = [
        [1 if i == kernel_coordinates[j] else 0 for j in range(kernel_dimension)]
        for i in range(initial_dimension)
    ]

    new_center = np.dot(np.transpose(pr_basis), center)

    # find new shape matrix
    basis_dot_shape = np.dot(np.transpose(pr_basis), shape_matrix)
    kernel_dot_shape = np.dot(np.transpose(kernel_basis), shape_matrix)
    e11 = np.dot(basis_dot_shape, pr_basis)
    # Yt * E * Z = (Zt * E * Y)t for symmetric E
    e12 = np.dot(basis_dot_shape, kernel_basis)
    e22 = np.dot(kernel_dot_shape, kernel_basis)
    e22 = np.linalg.inv(e22)
    temp = np.dot(e12, e22)
    temp = np.dot(temp, np.transpose(e12))

    new_shape_matrix = np.subtract(e11, temp)

    return new_center, new_shape_matrix
