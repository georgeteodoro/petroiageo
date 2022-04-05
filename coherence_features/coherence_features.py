import numpy as np
import scipy.ndimage
import scipy.signal
from algs import *

files = ["NEAR"]
windows = [(5,5,11)]
algs = {"gersz", "gst"}

name = lambda w: "-".join(str(x) for x in w)
for f in files:
    seismic = np.load("dados/"+f+".npy")

    curvature = curvature_parameters(seismic)
    if "dip_angle" in algs:
        coh = dip_anlge(curvature)
        np.save("results/"+f+"_dip-angle_.npy", coh)
    if "azimuth" in algs:
        coh = azimuth(curvature)
        np.save("results/"+f+"_azimuth_.npy", coh)
    if "mean_curvature" in algs:
        coh = mean_curvature(curvature)
        np.save("results/"+f+"_mean-curvature_.npy", coh)
    if "gaussian_curvature" in algs:
        coh = gaussian_curvature(curvature)
        np.save("results/"+f+"_gaussian-curvature_.npy", coh)
    if "max_curvature" in algs:
        coh = max_curvature(curvature)
        np.save("results/"+f+"_max-curvature_.npy", coh)
    if "min_curvature" in algs:
        coh = min_curvature(curvature)
        np.save("results/"+f+"_min-curvature_.npy", coh)
    if "most_positive_curvature" in algs:
        coh = most_positive_curvature(curvature)
        np.save("results/"+f+"_most-positive-curvature_.npy", coh)
    if "most_negative_curvature" in algs:
        coh = most_negative_curvature(curvature)
        np.save("results/"+f+"_most-negative-curvature_.npy", coh)
    if "shape_index" in algs:
        coh = shape_index(curvature)
        np.save("results/"+f+"_shape-index_.npy", coh)
    if "dip_curvature" in algs:
        coh = dip_curvature(curvature)
        np.save("results/"+f+"_dip-curvature_.npy", coh)
    if "contour_curvature" in algs:
        coh = contour_curvature(curvature)
        np.save("results/"+f+"_contur-curvature_.npy", coh)
    if "curvedness" in algs:
        coh = curvedness(curvature)
        np.save("results/"+f+"_curvedness_.npy", coh)


    for w in windows:
        if "marfurt" in algs:
            coh = moving_window(seismic, w, marfurt_semblance)
            np.save("results/"+f+"_marfurt_"+name(w)+".npy", coh)
        if "gersz" in algs:
            coh = moving_window(seismic, w, gersztenkorn)
            np.save("results/"+f+"_gersz_"+name(w)+".npy", coh)
        if "gst" in algs:
            coh = gst_coherence(seismic, w, sigma=1)
            np.save("results/"+f+"_gst_"+name(w)+".npy", coh)
        if "sobel" in algs:
            coh = gersz_sobel(seismic, w)
            np.save("results/"+f+"_sobel_"+name(w)+".npy", coh)
