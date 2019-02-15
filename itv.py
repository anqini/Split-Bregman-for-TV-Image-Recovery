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
from util import *

def itv(noisy, real):
    u = noisy
    u_prev = np.zeros(u.shape)
    # initializing
    dx = np.zeros(u.shape)
    dy = np.zeros(u.shape)
    bx = np.zeros(u.shape)
    by = np.zeros(u.shape)
    lambd = 0.1
    mu = 0.05
    tol = 0.001
    epsilon = 0.0001
    iter = 0
    # terminate condition
    while np.linalg.norm(u - u_prev) / np.linalg.norm(u) > tol \
            and iter < 40:
        iter = iter + 1
        u_prev = u
        u = gaussian_method(u, noisy, dx, dy, bx, by, lambd, mu)
        s = compute_s(u, bx, by)
        # print(s)
        dx = shrink(s, lambd) * (gradient_x(u) + bx) / (s + epsilon)
        dy = shrink(s, lambd) * (gradient_y(u) + by) / (s + epsilon)
        bx = bx + (gradient_x(u) - dx)
        by = by + (gradient_y(u) - dy)
        print('converge step ratio: {:.04f}'.format(np.linalg.norm(u - u_prev) / np.linalg.norm(u)))
        # print('distance to real: {}'.format(norm2(u - real)))
    return u


if __name__ == "__main__":
    # read image
    noisy, real = read_image()
    u = itv(noisy, real)
    plt.subplot(121)
    plt.imshow(u, cmap='gray')
    plt.subplot(122)
    plt.imshow(noisy, cmap='gray')
    plt.show()




