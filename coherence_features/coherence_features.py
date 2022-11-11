import numpy as np
import scipy.ndimage
import scipy.signal
from algs import *
import pathlib
from timeit import default_timer as timer
from datetime import datetime

def apply_and_save(func, save_path, *args, **kwargs):
    print(f"Calculando {func.__name__}")
    try:
        print(f'Começando cálculo {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        print(f"Cálculo terminou em {end-start} segundos")
    except Exception as e:
        print(f"WARNING: Exceção capturada!\n {e}")
        print(f"Dados computados por {func.__name__} não serão salvos!")
        print(f"Continuando para a próxima computação de feature")
    else:
        print(f"Salvando {save_path}")
        start = timer()
        np.save(save_path, result)
        end = timer()
        print(f"Salvo em {end-start} segundos")

def calc_curvature_features(algs, f, results_folder, seismic) -> None:
    curvature_features = [
        "dip_angle", "azimuth", "mean_curvature", "gaussian_curvature",
        "max_curvature", "min_curvature", "most_positive_curvature",
        "most_negative_curvature", "shape_index", "dip_curvature",
        "contour_curvature", "curvedness"
    ]
    #So don't calc curvature_parameters if don't have to
    if any([feature in algs for feature in curvature_features]):
        try:
            print("Calculando dados curvatura")
            start = timer()
            curvature = curvature_parameters(seismic)
            end = timer()
            print(f"Dados curvatura calculados em {end - start} segundos")
        except Exception as e:
            print(f"WARNING: Exceção capturada no cálculo dos parâmetros de curvatura!\n {e}")
            print(f"Continuando para as próximas categorias de features")
        else:
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

def calc_analitic_features(algs, f, results_folder, seismic) -> None:
    analitic_features = ["envelope", "instFrequency"]
    #So don't calc analitic cube if dont have to
    if any([feature in algs for feature in analitic_features]):
        try:
            print("Calculando cubo analítico")
            start = timer()
            analiticCube = analiticOf(seismic)
            end = timer()
            print(f"Cubo analítico calculado em {end - start} segundos")
        except Exception as e:
            print(f"WARNING: Exceção capturada no cálculo do cubo analítico!\n {e}")
            print(f"Continuando para as próximas categorias de features")
        else:
            if "envelope" in algs:
                apply_and_save(envelopeOf, results_folder / f"{f}_envelope_.npy", analiticCube)

            if "instFrequency" in algs:
                apply_and_save(instantaneousFrequencyOf, results_folder / f"{f}_instantaneous-frequency_.npy", analiticCube)

def calc_3d_window_features(n_cpu, windows_3D, algs, f, results_folder, seismic):
    for w in windows_3D:
        print(f"Window 3D: {w}")
        if "marfurt" in algs:
            print("Calculating marfurt")
            apply_and_save(moving_window, results_folder / f"{f}_marfurt_{w}.npy", seismic, w, marfurt_semblance, n_cpu)
        if "gersz" in algs:
            print("Calculating gersz")
            apply_and_save(moving_window, results_folder / f"{f}_gersz_{w}.npy", seismic, w, gersztenkorn, n_cpu)
        if "gst" in algs:
            apply_and_save(gst_coherence, results_folder / f"{f}_gst_{w}.npy", seismic, w, {'sigma':1})
        if "sobel" in algs:
            apply_and_save(gersz_sobel, results_folder / f"{f}_sobel_{w}.npy", seismic, w)
        if "median" in algs:
            print("Calculating median")
            apply_and_save(moving_window, results_folder / f"{f}_median_{w}.npy", seismic, w, np.median, n_cpu)
        if "mean" in algs:
            print("Calculating mean")
            apply_and_save(moving_window, results_folder / f"{f}_mean_{w}.npy", seismic, w, np.mean, n_cpu)
        if "min" in algs:
            print("Calculating min")
            apply_and_save(moving_window, results_folder / f"{f}_min_{w}.npy", seismic, w, np.min, n_cpu)
        if "max" in algs:
            print("Calculating max")
            apply_and_save(moving_window, results_folder / f"{f}_max_{w}.npy", seismic, w, np.max, n_cpu)
        if "sum" in algs:
            print("Calculating sum")
            apply_and_save(moving_window, results_folder / f"{f}_sum_{w}.npy", seismic, w, np.sum, n_cpu)

def calc_1d_window_features(windows_1D, algs, f, results_folder, seismic):
    for w in windows_1D:
        print(f"Window 1d: {w}")
        if "rms" in algs:
            apply_and_save(rms, results_folder / f"{f}_rms-{str(w)}_.npy", w, seismic)

def load_seismic_data(data_folder, current_file):
    data_file_path = data_folder / f"{current_file}.npy"
    if not data_file_path.exists() or not data_file_path.is_file():
        print(f"{data_file_path} não é um caminho de arquivo sísmico aceito!")
        exit(-1)
    
    seismic = None
    try:
        print("Carregando dados sísmicos")
        start = timer()
        seismic = np.load(data_file_path)
        end = timer()
        print(f"Dados sísmicos carregados em {end- start} segundos")
    except Exception as e:
        print(f"ERROR: Exceção capturada ao carregar dados sísmicos de {data_file_path}!\n {e}")
        print("Interrompendo execução do programa")
        exit(-1)
    else:  
        return seismic
    

if __name__ == "__main__":
    N_CPU = 20 #Must be a positive integer

    seismic_files = ["Franco_florin_buzios_28_09-2"]

    windows_1D = [3, 5, 7, 9]

    windows_3D = [
            #(5,5,11),
            #(5,5,9),
            #(5,5,7),
            (3,3,11),
            #(3,3,9),
            #(3,3,7)
        ]

    algs_to_run = {
            #"gersz", "marfurt", "sobel",
            "instFrequency", "envelope",
            "median", "mean", "max", "min", "sum"
    }

    if algs_to_run == {"all"}:
        algs_to_run = {
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

    data_load_folder = pathlib.Path("/petrobr/parceirosbr/petrobrasiageo/francisco.azevedo/")
    if not data_load_folder.exists() or not data_load_folder.is_dir():
        print(f" {data_load_folder} não é um caminho de pasta aceitável!")
        exit(-1)

    results_folder = pathlib.Path("results/")
    results_folder.mkdir(exist_ok=True)
    
    for current_file in seismic_files:
        seismic_data = load_seismic_data(data_load_folder, current_file)

        calc_curvature_features(algs_to_run, current_file, results_folder, seismic_data)

        calc_analitic_features(algs_to_run, current_file, results_folder, seismic_data)

        calc_1d_window_features(windows_1D, algs_to_run, current_file, results_folder, seismic_data)

        calc_3d_window_features(N_CPU, windows_3D, algs_to_run, current_file, results_folder, seismic_data)

        print(f"Fim computação para o arquivo {current_file}")