"""
This script filters the seismic values of the target file based on the
wells x and y coordinates and a z range.
"""
import numpy as np
import pathlib

# Every seismic point z wise has a resolution of
SEISMIC_FILE_RESOLUTION = 5

if __name__ == "__main__":
    # Como o arquivo sísmico tem medições à cada 5 metros e a profundidade
    # alvo da porosidade é entre TARGET_Z_MIN e TARGET_Z_MAX metros, tenho que dividir por
    # SEISMIC_FILE_RESOLUTION esse range para chegar no range correto no arquivo sísmico.
    # Por causa disso, pode ser que o valor de profundidade dos extremos seja diferente
    # do TARGET_Z_MIN e TARGET_Z_MAX informados
    TARGET_Z_MIN = 5473
    TARGET_Z_MAX = 5859
    z_start, z_end = TARGET_Z_MIN // SEISMIC_FILE_RESOLUTION, TARGET_Z_MAX // SEISMIC_FILE_RESOLUTION
    print(f"Intervalo de dados original pretendido: [{z_start}, {z_end}], {z_end-z_start+1} valores por poço.")
 
    # Where to read the seismic values from
    origin_npy_file_path = pathlib.Path(
        "/petrobr/parceirosbr/petrobrasiageo/dados/Franco_florin_buzios_28_09-2.npy"
        )
    print(f"Carregando dados de {origin_npy_file_path}")
    z_filtered_np = np.load(origin_npy_file_path)
    print(f"Shape do arquivo sísmico original: {z_filtered_np.shape}")
    max_z_to_filter_in_seismic = z_filtered_np.shape[-1]
    if max_z_to_filter_in_seismic < z_end+1:
        print(f"O arquivo sísmico tem z máximo de {max_z_to_filter_in_seismic}" +
            f" mas o z_end alvo é de {z_end+1}. Filtrando pelo mínimo.")
    
    z_end_to_filter = min(z_filtered_np.shape[-1], z_end+1)
    print(f"Filtrando em z no intervalo [{z_start},{z_end_to_filter}), {z_end_to_filter-z_start} valores por poço.")
    z_filtered_np = z_filtered_np[:,:,z_start : z_end_to_filter]
    print(f"Depois de filtrar por z, tem o shape de: {z_filtered_np.shape}")

    # Where to save the filtered values
    target_npy_file_path = pathlib.Path(
        "/petrobr/parceirosbr/petrobrasiageo/dados/Franco_florin_buzios_wells_filtered_area_3.npy"
        )

    target_x_y_coords = [[415, 392],[477, 329]]
    
    n_wells = len(target_x_y_coords)
    print(f"{n_wells} poços a serem filtrados nas coordenadas: {target_x_y_coords}")
    resulting_np = None

    for well_id, (x, y) in enumerate(target_x_y_coords):
        filtered_np = z_filtered_np[x, y, :]
        print(f"Poço {well_id+1}/{n_wells}. Filtrou np com shape: {filtered_np.shape}")
        # Create x,y,z,seismic numpy array, i.e., 
        # a (4, filtered_np.shape[-1]) shaped array
        n_values = filtered_np.shape[-1]
        xs = np.array([x]*n_values).reshape(n_values, 1)
        ys = np.array([y]*filtered_np.shape[-1]).reshape(n_values, 1)
        # The zs are in the original depth, thats why we multiply
        # it back with SEISMIC_FILE_RESOLUTION
        zs = np.arange(z_start, z_end_to_filter, 1).reshape(n_values, 1) * SEISMIC_FILE_RESOLUTION
        # Goes from (shape, ) to (shape, 1)
        filtered_np = filtered_np.reshape(n_values, 1)
        curr_np = np.concatenate([xs, ys, zs, filtered_np], axis=1)

        if resulting_np is None:
            resulting_np = curr_np
        else:
            resulting_np = np.concatenate([resulting_np, curr_np])
        
        print(f"Novo np resultante de shape {resulting_np.shape}")
    
    print(f"Tentando salvar em {target_npy_file_path}")
    try:
        np.save(target_npy_file_path, resulting_np)
    except Exception as e:
        print("Erro ao salvar:\n"+e)
    else:
        print("Salvo com sucesso")
