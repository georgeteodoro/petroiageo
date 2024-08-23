# Instruções para cálculo de *features*

## Passo 1 - Calcular as *features*
Para calcular as *features* sobre um cubo sísmico, deve-se executar o seguinte comando: `make_features.py -o OUTPUT -w1 WINDOW1D -w3 WINDOW3D WINDOW3D WINDOW3D -n_cpu N_CPU FILENAME ALGORITHM [ALGORITHM ...]`. Esse script `make_features.py` é responsável por calcular as *features* sobre o cubo sísmico. Mais informações sobre cada parâmetro podem ser vistas com o comando `python3 make_features.py -h`.

Basicamente, `FILENAME` é o caminho para o arquivo sísmico do qual devemos computar as *features*; `N_CPU` é o número de CPUs que poderão ser utilizadas para calcular as *features* em paralelo; `OUTPUT` é o caminho da **pasta** que deverá receber os arquivos de *features* gerados; `WINDOW1D` é o tamanho da janela 1D a ser utilizada pelas *features* que precisam desse parâmetro; a sequência de 3 `WINDOW3D` é referente às três dimensões da uma janela 3D que será necessária para algumas *features*; a sequência de `ALGORITHM` é uma sequência de nomes de *features* a serem computadas separados por espaço. A lista completa de *features* que podem ser passadas é:

### *Features* de curvatura
- dip_angle
- azimuth
- mean_curvature
- gaussian_curvature
- max_curvature
- min_curvature
- most_positive_curvature
- most_negative_curvature
- shape_index
- dip_curvature
- contour_curvature
- curvedness

### *Features* analíticas
- envelope
- instFrequency

### *Features* com janela 3D
- marfurt
- gersz
- gst
- sobel
- median
- mean
- min
- max
- sum

### *Feature* com janela 1D
- rms

Obs: Passar a lista de algoritmos como um elemento igual a all irá computar todas as features disponíveis.

## Passo 2 - Atualizar o formato do arquivo
Os arquivos de *features* gerados no passo anterior estão no formato `.npy`. Devemos alterar esse formato para `.h5`. Para isso, execute seguinte comando: `seismic_data3_hdf5.py -f FEATURE_DIR -o OUTPUT_DIR --large MULT_FACTOR --config CONFIG_FILE_PATH`. Mais informações sobre cada parâmetro podem ser vistas com o comando `python3 make_features.py -h`.

Basicamente, `FEATURE_DIR` é o caminho da pasta que recebeu os arquivos de *features* gerados no passo anterior; `OUTPUT_DIR` é o caminho da pasta que deverá receber os novos arquivos de *features*; o parâmetro `large MULT_FACTOR` é opcional; `CONFIG_FILE_PATH` é o caminho para o arquivo de configuração `YAML` do algoritmo. Nesse ponto do *pipeline*, só precisamos das seção `wells` desse arquivo de configuração que indica as coordenadas dos poços na área alvo e o `window`.

**TODO**: Explicar para que serve o parâmetro window do arquivo de configuração e explicar o parâmetro MULT_FACTOR dos argumentos.
