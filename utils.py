import numpy as np
def ssa(angle):
    return (angle + np.pi)%(2*np.pi) - np.pi

def GCS(x1, y1, x, y, theta):
    xr, yr = x1 - x, y1 - y
    xn = xr * np.cos(theta) + -yr * np.sin(theta)
    yn = xr * np.sin(theta) + yr * np.cos(theta)
    return xn, yn

def BCS(x1, y1, x, y, theta):
    xr, yr = x1 - x, y1 - y
    xn = xr * np.cos(theta) + yr * np.sin(theta)
    yn = -xr * np.sin(theta) + yr * np.cos(theta)
    return xn, yn

def tan_inv(X,Y):
    return np.arctan2(Y, X)

def distance(X, Y):
    return np.sqrt(np.square(X) + np.square(Y))