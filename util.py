'''
    This project is written by Anqi Ni(anqini4@gmail.com)
    according the algorithm on the paper:
    'The Split Bregman Method for L1-Regularized Problems'(2009)
    by Tom Goldstein and Stanley Osher published
    on SIAM J. IMAGING SCIENCES Vol2, No. 2, pp323-343.
    And it is For Educational Purposes Only.
    PLEASE do NOT duplicate or spread these files
    without citing the author(Anqi Ni).
'''

import numpy as np
import imageio
import matplotlib.pyplot as plt


def read_image():
    real = imageio.imread('lena_gray.bmp')
    # convert the image to numpy array
    real = np.array(real)
    real = real.astype(int)
    # add noise to the image
    noisy = imnoise(real)
    return (noisy, real)


def imnoise(l):
	noise = 3
    return l + noise * l.std() * np.random.random(l.shape)


def gradient_x(u):
    # get the height and width of a matrix
    height, width = u.shape
    # create a new empty matrix u2
    # intialize this matrix with all zeros
    u2 = np.zeros((height, width), dtype = int)
    # loop over all elements
    # for i in range(height):
    #     for j in range(width):
    #         u2[i, j] = u[i, j] - u[i - 1, j]

    # faster one
    u2[1:height, :] = u[1:height, :] - u[0:(height - 1), :]
    u2[0, :] = u[0, :] - u[height - 1, :]
    return u2


def gradient_y(u):
    # get the height and width of a matrix
    height, width = u.shape
    # create a new empty matrix u2
    # intialize this matrix with all zeros
    u2 = np.zeros((height, width), dtype = int)
    # loop over all elements
    # for i in range(height):
    #     for j in range(width):
    #         u2[i, j] = u[i, j] - u[i, j - 1]

    # faster one
    u2[:, 1:width] = u[:, 1:width] - u[:, 0:(width - 1)]
    u2[:, 0] = u[:, 0] - u[:, width - 1]
    return u2


# move the matrix upward
def up(u):
    # get the height and width of a matrix
    height, width = u.shape
    # get a copy a u to make change
    u2 = np.copy(u)
    u2[0:(height - 1), :] = u[1:height, :]
    u2[height - 1, :] = u[0, :]
    return u2


# downward
def down(u):
    # get the height and width of a matrix
    height, width = u.shape
    # get a copy a u to make change
    u2 = np.copy(u)
    u2[1:height, :] = u[0:(height - 1), :]
    u2[0, :] = u[height - 1, :]
    return u2


# leftward
def left(u):
    # get the height and width of a matrix
    height, width = u.shape
    # get a copy a u to make change
    u2 = np.copy(u)
    u2[:, 0:(width - 1)] = u[:, 1:width]
    u2[:, width - 1] = u[:, 0]
    return u2

# rightward
def right(u):
    # get the height and width of a matrix
    height, width = u.shape
    # get a copy a u to make change
    u2 = np.copy(u)
    u2[:, 1:width] = u[:, 0:(width - 1)]
    u2[:, 0] = u[:, width - 1]
    return u2


def neighbors(u):
    return up(u) + down(u) + left(u) + right(u)


def shrink(u, lambd):
    z = np.zeros(u.shape)
    u = np.sign(u) * np.maximum(np.abs(u) - (1. / lambd), z)
    return u


def norm2(u):
    sq = np.square(u)
    s = np.sum(sq)
    return s


def gaussian_method(u, f, dx, dy, bx, by, lambd, mu):
    mul1 = lambd / (mu + 4 * lambd)
    mul2 = mu / (mu + 4 * lambd)
    # print('gradient x: \n{}'.format(gradient_x(u)))
    # print('gradient y: \n{}'.format(gradient_y(u)))
    G = mul1 * (neighbors(u) + gradient_x(dx) + gradient_y(dy) -
                gradient_x(bx) - gradient_y(by)) + mul2 * f
    return G


# this is more like a gaussian-seidel method
# but it takes longer to execute
# we can use it as an alternative of gaussian_method above
def gaussian_method2(u, f, dx, dy, bx, by, lambd, mu):
    mul1 = lambd / (mu + 4 * lambd)
    mul2 = mu / (mu + 4 * lambd)
    height, width = u.shape
    ddx = gradient_x(dx)
    ddy = gradient_y(dy)
    dbx = gradient_x(bx)
    dby = gradient_y(by)
    for i in range(height):
        for j in range(width):
            if i != (height - 1) and j != (width - 1):
                u[i, j] = mul1 * (u[i - 1, j] + u[i + 1, j] +
                          u[i, j - 1] + u[i, j + 1] +
                          ddx[i, j] + ddy[i, j] -
                          dbx[i, j] - dby[i, j]) + \
                          mul2 * f[i,j]
            elif i == (height - 1) and j != (width - 1):
                u[i, j] = mul1 * (u[i - 1, j] + u[0, j] +
                          u[i, j - 1] + u[i, j + 1] +
                          ddx[i, j] + ddy[i, j] -
                          dbx[i, j] - dby[i, j]) + \
                          mul2 * f[i,j]
            elif i != (height - 1) and j == (width - 1):
                u[i, j] = mul1 * (u[i - 1, j] + u[i + 1, j] +
                          u[i, j - 1] + u[i, 0] +
                          ddx[i, j] + ddy[i, j] -
                          dbx[i, j] - dby[i, j]) + \
                          mul2 * f[i,j]
            else:  # i == (height - 1) and j == (width - 1)
                u[i, j] = mul1 * (u[i - 1, j] + u[0, j] +
                          u[i, j - 1] + u[i, 0] +
                          ddx[i, j] + ddy[i, j] -
                          dbx[i, j] - dby[i, j]) + \
                          mul2 * f[i,j]
    return u


def compute_s(u, bx, by):
    return np.sqrt(np.square(gradient_x(u) + bx) + np.square(gradient_y(u) + by))

