import numpy as np
import scipy.ndimage
import scipy.signal
from algs import *

files = ["NEAR"]
windows3D = [
        (5,5,11), (5,5,9), (5,5,7),
        (3,3,11), (3,3,9), (3,3,7)
    ]
windows1D = [3, 5, 7, 9]
algs = {"all"}

if algs == {"all"}:
    algs = {
                "dip_angle",
                "azimuth",
                "mean_curvature",
                "gaussian_curvature",
                "max_curvature",
                "min_curvature",
                "most_positive_curvature",
                "most_negative_curvature",
                "shape_index",
                "dip_curvature",
                "contour_curvature",
                "curvedness",
                "rms",
                "instFrequency",
                "envelope",
                "marfurt",
                "sobel",
                "gersz",
                "gst",
                "median",
                "mean",
                "max",
                "min",
                "sum"
        }

name = lambda w: "-".join(str(x) for x in w)
for f in files:
    seismic = np.load("dados/"+f+".npy")

    curvature = curvature_parameters(seismic)
    if "dip_angle" in algs:
        coh = dip_angle(curvature)
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
    
    for window1D in windows1D:
        if "rms" in algs:
            rmsCube = rms(window1D, seismic)
            np.save("results/"+f+"_rms-"+str(window1D)+"_.npy", rmsCube)
    
    runEnvelope = "envelope" in algs
    runInstFrequency = "instFrequency" in algs
    if runEnvelope or runInstFrequency:
        analiticCube = analiticOf(seismic)
        
        if runEnvelope:
            envelopeCube = envelopeOf(analiticCube)
            np.save("results/"+f+"_envelope_.npy", envelopeCube)

        if runInstFrequency:
            instFrequencyCube = instantaneousFrequencyOf(analiticCube)
            np.save("results/"+f+"_instantaneous-frequency_.npy", instFrequencyCube)

    for w in windows3D:
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
        if "median" in algs:
            coh = moving_window(seismic, w, np.median)
            np.save("results/"+f+"_median_"+name(w)+".npy", coh)
        if "mean" in algs:
            coh = moving_window(seismic, w, np.mean)
            np.save("results/"+f+"_mean_"+name(w)+".npy", coh)
        if "min" in algs:
            coh = moving_window(seismic, w, np.min)
            np.save("results/"+f+"_min_"+name(w)+".npy", coh)
        if "max" in algs:
            coh = moving_window(seismic, w, np.max)
            np.save("results/"+f+"_max_"+name(w)+".npy", coh)
        if "sum" in algs:
            coh = moving_window(seismic, w, np.sum)
            np.save("results/"+f+"_sum_"+name(w)+".npy", coh)
