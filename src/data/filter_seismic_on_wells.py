"""
This script filters the seismic values of the target file based on the
wells x and y coordinates and a z range.
"""
import numpy as np
import pathlib

if __name__ == "__main__":
    # Como o arquivo sísmico tem medições à cada 5 metros e a profundidade
    # alvo da porosidade é entre 5651 e 5810 metros, tenho que dividir por 5
    # esse range para chegar no range correto no arquivo sísmico.
    z_start, z_end = 5651 // 5, 5810 // 5
    
    # Where to read the seismic values from
    origin_npy_file_path = pathlib.Path(
        "/petrobr/parceirosbr/petrobrasiageo/dados/Franco_florin_buzios_28_09-2.npy"
        )
    print(f"Carregando dados de {origin_npy_file_path}")
    z_filtered_np = np.load(origin_npy_file_path)[:,:,z_start : z_end + 1]
    print(f"Depois de filtrar por z, tem o shape de: {z_filtered_np.shape}")

    # Where to save the filtered values
    target_npy_file_path = pathlib.Path(
        "/petrobr/parceirosbr/petrobrasiageo/dados/Franco_florin_buzios_wells_filtered.npy"
        )

    target_x_y_coords = [
        [855, 1230], 
        [834, 1297], 
        [833, 1384], 
        [898, 1246], 
        [901, 1332]]

    resulting_np = None

    for x, y in target_x_y_coords:
        filtered_np = z_filtered_np[x, y, :]
        print(f"Filtrou np com shape: {filtered_np.shape}")
        # Create x,y,z,seismic numpy array, i.e., 
        # a (4, filtered_np.shape[-1]) shaped array

        xs = np.array([x]*filtered_np.shape[-1])
        ys = np.array([y]*filtered_np.shape[-1])
        zs = np.arange(0, filtered_np.shape[-1], 1)
        # Goes from (shape, ) to (shape, 1)
        filtered_np = filtered_np.reshape((filtered_np.shape[-1],1))
        curr_np = np.concatenate([xs, ys, filtered_np], axis=1)

        if resulting_np is None:
            resulting_np = curr_np
        else:
            resulting_np = np.concatenate([resulting_np, curr_np])
        
        print(f"Novo np resultante de shape {resulting_np.shape}")
    
    print(f"Tentando salvar em {target_npy_file_path}")
    np.save(target_npy_file_path, resulting_np)