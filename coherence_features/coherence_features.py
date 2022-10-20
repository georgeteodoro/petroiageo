import numpy as np
import scipy.ndimage
import scipy.signal
from algs import *
import pathlib

n_cpu = 1
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

def apply_and_save(func, save_path, *args, **kwargs):
    result = func(*args, **kwargs)
    np.save(save_path, result)

name = lambda w: "-".join(str(x) for x in w)
for f in files:
    data_folder = pathlib.Path("dados/")
    results_folder = pathlib.Path("results/")
    results_folder.mkdir(exist_ok=True)

    seismic = np.load(data_folder / f"{f}.npy")

    curvature = curvature_parameters(seismic)
    if "dip_angle" in algs:
        apply_and_save(dip_angle, results_folder / f"{f}_dip-angle_.npy", curvature)
    if "azimuth" in algs:
        apply_and_save(azimuth, results_folder / f"{f}_azimuth_.npy", curvature)
    if "mean_curvature" in algs:
        apply_and_save(mean_curvature, results_folder / f"{f}_mean-curvature_.npy", curvature)
    if "gaussian_curvature" in algs:
        apply_and_save(gaussian_curvature, results_folder / f"{f}_gaussian-curvature_.npy", curvature)
    if "max_curvature" in algs:
        apply_and_save(max_curvature, results_folder / f"{f}_max-curvature_.npy", curvature)
    if "min_curvature" in algs:
        apply_and_save(min_curvature, results_folder / f"{f}_min-curvature_.npy", curvature)
    if "most_positive_curvature" in algs:
        apply_and_save(most_positive_curvature, results_folder / f"{f}_most-positive-curvature_.npy", curvature)
    if "most_negative_curvature" in algs:
        apply_and_save(most_negative_curvature, results_folder / f"{f}_most-negative-curvature_.npy", curvature)
    if "shape_index" in algs:
        apply_and_save(shape_index, results_folder / f"{f}_shape-index_.npy", curvature)
    if "dip_curvature" in algs:
        apply_and_save(dip_curvature, results_folder / f"{f}_dip-curvature_.npy", curvature)
    if "contour_curvature" in algs:
        apply_and_save(contour_curvature, results_folder / f"{f}_contur-curvature_.npy", curvature)
    if "curvedness" in algs:
        apply_and_save(curvedness, results_folder / f"{f}_curvedness_.npy", curvature)
    
    for window1D in windows1D:
        if "rms" in algs:
            apply_and_save(rms, results_folder / f"{f}_rms-"+str(window1D)+"_.npy", window1D, seismic)
    
    runEnvelope = "envelope" in algs
    runInstFrequency = "instFrequency" in algs
    if runEnvelope or runInstFrequency:
        analiticCube = analiticOf(seismic)
        
        if runEnvelope:
            apply_and_save(envelopeOf, results_folder / f"{f}_envelope_.npy", analiticCube)

        if runInstFrequency:
            apply_and_save(instantaneousFrequencyOf, results_folder / f"{f}_instantaneous-frequency_.npy", analiticCube)

    for w in windows3D:
        if "marfurt" in algs:
            apply_and_save(moving_window, results_folder / f"{f}_marfurt_"+name(w)+".npy", seismic, w, marfurt_semblance, n_cpu)
        if "gersz" in algs:
            apply_and_save(moving_window, results_folder / f"{f}_gersz_"+name(w)+".npy", seismic, w, gersztenkorn, n_cpu)
        if "gst" in algs:
            apply_and_save(gst_coherence, results_folder / f"{f}_gst_"+name(w)+".npy", seismic, w, {'sigma':1})
        if "sobel" in algs:
            apply_and_save(gersz_sobel, results_folder / f"{f}_sobel_"+name(w)+".npy", seismic, w)
        if "median" in algs:
            apply_and_save(moving_window, results_folder / f"{f}_median_"+name(w)+".npy", seismic, w, np.median)
        if "mean" in algs:
            apply_and_save(moving_window, results_folder / f"{f}_mean_"+name(w)+".npy", seismic, w, np.mean, n_cpu)
        if "min" in algs:
            apply_and_save(moving_window, results_folder / f"{f}_min_"+name(w)+".npy", seismic, w, np.min, n_cpu)
        if "max" in algs:
            apply_and_save(moving_window, results_folder / f"{f}_max_"+name(w)+".npy", seismic, w, np.max, n_cpu)
        if "sum" in algs:
            apply_and_save(moving_window, results_folder / f"{f}_sum_"+name(w)+".npy", seismic, w, np.sum, n_cpu)