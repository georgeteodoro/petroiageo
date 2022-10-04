from math import ceil


# def filter_h5_all_clusters(d_h5, f):
#     chunks = d_h5.chunks
#     x_shape = d_h5.shape[0]
#     y_shape = d_h5.shape[1]
#     z_shape = d_h5.shape[2]

#     out = []

#     # Iterate on all coordinates
#     for c_x in range(ceil(x_shape / chunks[0])):
#         x_i = c_x * chunks[0]
#         x_o = min((c_x + 1) * chunks[0], x_shape)
#         for c_y in range(ceil(y_shape / chunks[1])):
#             y_i = c_y * chunks[1]
#             y_o = min((c_y + 1) * chunks[1], y_shape)
#             for c_z in range(ceil(z_shape / chunks[2])):
#                 z_i = c_z * chunks[2]
#                 z_o = min((c_z + 1) * chunks[2], z_shape)

#                 d_np = d_h5[x_i:x_o, y_i:y_o, z_i:z_o]

#                 out = out + d_np[f(d_np)].tolist()

#     return out

def fold_h5_all_clusters(d_h5, f, out_0):
    chunks = d_h5.chunks
    x_shape = d_h5.shape[0]
    y_shape = d_h5.shape[1]
    z_shape = d_h5.shape[2]

    out = out_0

    # Iterate on all coordinates
    for c_x in range(ceil(x_shape / chunks[0])):
        x_i = c_x * chunks[0]
        x_o = min((c_x + 1) * chunks[0], x_shape)
        for c_y in range(ceil(y_shape / chunks[1])):
            y_i = c_y * chunks[1]
            y_o = min((c_y + 1) * chunks[1], y_shape)
            for c_z in range(ceil(z_shape / chunks[2])):
                z_i = c_z * chunks[2]
                z_o = min((c_z + 1) * chunks[2], z_shape)

                d_np = d_h5[x_i:x_o, y_i:y_o, z_i:z_o]

                out = out + f(d_np)

    return out
