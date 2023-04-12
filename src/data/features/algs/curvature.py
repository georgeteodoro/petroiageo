import numpy as np

def curvature_parameters(data):
    Zx, Zy = np.gradient(data, axis=(0,1))
    Zxx, Zxy = np.gradient(Zx, axis=(0,1))
    Zyy = np.gradient(Zy, axis=1)
    return [Zxx/2, Zyy/2, Zxy, Zx, Zy]

def dip_angle(params):
    return np.arctan(np.sqrt(params[3]**2 + params[4]**2))

def azimuth(params):
    return np.arctan(params[4] / params[3])

def mean_curvature(params):
    a, b, c, d, e = params
    return (a*(1+e**2)-c*d*e+b*(1+d**2))/np.power(1+d**2+e**2,3/2)

def gaussian_curvature(params):
    a, b, c, d, e = params
    return (4*a*b - c**2) / (1 + d**2 + e**2)**2

def max_curvature(params):
    km = mean_curvature(params)
    kg = gaussian_curvature(params)
    return km + np.sqrt(km**2 - kg)

def min_curvature(params):
    km = mean_curvature(params)
    kg = gaussian_curvature(params)
    return km - np.sqrt(km**2 - kg)

def most_positive_curvature(params):
    a, b, c, _, _ = params
    return a + b + np.sqrt((a-b)**2 + c**2)

def most_negative_curvature(params):
    a, b, c, _, _ = params
    return a + b - np.sqrt((a-b)**2 + c**2)

def shape_index(params):
    kmax = max_curvature(params)
    kmin = min_curvature(params)
    return 2 / np.pi * np.arctan((kmin+kmax)/(kmax-kmin))

def dip_curvature(params):
    a, b, c, d, e = params
    return 2*(a*e**2-c*d*e+b*d**2)/np.sqrt(1+d**2+e**2)/(d**2+e**2)

def contour_curvature(params):
    a, b, c, d, e = params
    return 2*(a*e**2-c*d*e+b*d**2)/np.power(1+d**2+e**2,3/2)

def curvedness(params):
    kmax = max_curvature(params)
    kmin = min_curvature(params)
    return np.sqrt((kmax**2 + kmin**2)/2)
