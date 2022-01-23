import pandas as pd
import numpy as np


def prepare_header(window):
    # window = 3
    out_list = list()
    for i in range(-window, window + 1):
        for j in range(-window, window + 1):
            for z in range(-window, window + 1):
                out_list.append(f"near {i} {j} {z},")
                out_list.append(f"mid {i} {j} {z},")
                out_list.append(f"far {i} {j} {z},")
                out_list.append(f"ufar {i} {j} {z},")
                out_list.append(f"gersz {i} {j} {z},")
                out_list.append(f"gst {i} {j} {z},")
    out_list.append("well,real,X,Y,depth,phi,rho,vp,vs")

    return "".join(out_list)


if __name__ == '__main__':
    prepare_header(3)