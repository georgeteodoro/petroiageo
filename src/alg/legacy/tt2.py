import pandas as pd
import numpy as np

w = 2
for i in range(-w, w + 1):
    for j in range(-w, w + 1):
        for z in range(-w, w + 1):
            print("near", i, j, z, end=",")
            print("mid", i, j, z, end=",")
            print("far", i, j, z, end=",")
            print("ufar", i, j, z, end=",")
print("well,real,X,Y,depth,phi,rho,vp,vs")
