from math import sin

from numpy import array

g = 9.81
l = 0.1


def f(r, t):
    theta = r[0]
    omega = r[1]
    ftheta = omega
    fomega = -(g / l) * sin(theta)
    return array([ftheta, fomega], float)
