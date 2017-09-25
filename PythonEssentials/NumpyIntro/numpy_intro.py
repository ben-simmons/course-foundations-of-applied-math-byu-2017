# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Ben Simmons
Self Study
9/24/2017
"""
import numpy as np


def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A = np.array([[3, -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])
    return A @ B


def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    return -1 * (A @ A @ A) + 9 * (A @ A) - 15 * A


def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.triu(np.ones((7,7)))
    B = (np.ones((7,7)) * -1) + (np.triu(np.ones((7,7))) * 6)
    return (A @ B @ A).astype(np.int64)


def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    B = np.array(A)
    mask = B < 0
    B[mask] = 0
    return B


def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.array([[3, 0, 0], [3, 3, 0], [3, 3, 3]])
    C = np.diag([-2, -2, -2])
    D1 = np.hstack((np.zeros((3,3)), np.transpose(A), np.eye(3)))
    D2 = np.hstack((A, np.zeros((2,2)), np.zeros((2,3))))
    D3 = np.hstack((B, np.zeros((3,2)), C))
    D = np.vstack((D1, D2, D3))
    return D


def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    A_rowsums = A.sum(axis=1)
    B = A / np.vstack(A_rowsums)
    return B


## Array slicing example ##

# np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])

# Instead of looping through every element individually,
# shift the arrays and vectorize the multiplication.

# grid[:,:-3]
# return grid minus last 3 cols
# shape (20,17)

# grid[:,1:-2] ****
# return grid minus first 1 col and last 2 cols
# shape (20,17)

# grid[:, 2:-1]
# return grid minus first 2 cols and last 1 col
# shape (20,17)

# grid[:,3:]
# return grid minus first 3 cols
# shape (20,17)


# Project Euler problem #11: https://projecteuler.net/problem=11
def prob7(grid=None):
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    if grid is None:
        grid = np.load("grid.npy")

    hmax = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])
    vmax = np.max(grid[:-3] * grid[1:-2] * grid[2:-1] * grid[3:])
    rdmax = np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:,3:])
    ldmax = np.max(grid[:-3,3:] * grid[1:-2,2:-1] * grid[2:-1,1:-2] * grid[3:,:-3])

    return np.max(np.array([hmax, vmax, rdmax, ldmax]))


# max = 70600674 (from left diagonals)
def prob7_naive(grid=None):
    if grid is None:
        grid = np.load("grid.npy")

    num_rows = grid.shape[0]
    num_cols = grid.shape[1]
    max = -np.inf

    for i in range(num_rows):
        for j in range(num_cols):
            # vertical
            if i < num_rows - 3:
                vprod = grid[i,j] * grid[i+1,j] * grid[i+2,j] * grid[i+3,j]
                if vprod > max:
                    max = vprod
            # horizontal
            if j < num_cols - 3:
                hprod = grid[i,j] * grid[i,j+1] * grid[i,j+2] * grid[i,j+3]
                if hprod > max:
                    max = hprod
            # right diagonal
            if i < num_rows - 3 and j < num_cols - 3:
                rdprod = grid[i,j] * grid[i+1,j+1] * grid[i+2,j+2] * grid[i+3,j+3]
                if rdprod > max:
                    max = rdprod
            # left diagonal
            if i < num_rows - 3 and j > 3:
                ldprod = grid[i,j] * grid[i+1,j-1] * grid[i+2,j-2] * grid[i+3,j-3]
                if ldprod > max:
                    max = ldprod

    return max


def prob7_perf_test():
    """
    Time them sans load time.

    Slicing is about 10X faster than naive looping.

    Example:
    TIME NAIVE: 0.002044200897216797
    TIME SLICING: 0.00021886825561523438
    """
    import time
    grid = np.load("grid.npy")

    ts = time.time()
    prob7_naive(grid=grid)
    te = time.time()
    print("TIME NAIVE: {}".format(te - ts))

    ts = time.time()
    prob7(grid=grid)
    te = time.time()
    print("TIME SLICING: {}".format(te - ts))
