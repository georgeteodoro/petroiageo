# Instruções de preparação de dados

## Passo 1 - Preparação de dados

Prepare os dados crus seguindo os passos descritos em `data_prep/README.md`

## Passo 2 - Computação de features
Compute as features seguindo os passos descritos em `features/README.md`

## Passo 3: Geração do cubo de porosidade inicial
Agora, deve-se gerar o cubo de porosidade que será preenchido pelo algoritmo. Inicialmente, ele terá apenas as porosidades ao longo dos poços da área. Para tal, deve-se executar o seguinte comando: `wells_data3_hdf5.py [-h] -f FEAT_FILE_PATH -p POROSITY_FILE -p_col POR_COL -o HDF5_FILE [--large MULT_FACTOR]`. Mais informações sobre cada parâmetro podem ser vistas com o comando `python3 wells_data3_hdf5.py -h`.

Basicamente, o `FEAT_FILE_PATH` é o caminho para algum arquivo de feature no formato h5 desse cubo; `POROSITY_FILE` é o caminho para o arquivo de porosidade que possui informações das coordenadas e porosidades para cada ponto de poço. Esse arquivo foi gerado durante o último passo do Passo 1; `POR_COL` é o nome da coluna de `POROSITY_FILE` na qual deve-se levar em consideração para preencher inicialmente o cubo. Isso se deve ao fato de poderem haver vários tipos de porosidades; `HDF5_FILE` caminho de saída do cubo (deve terminar com a extensão `h5`). O argumento opcional `--large MULT_FACTOR` pode ser passado como forma de aumentar os dados ao replicar poços de forma aleatória pelo cubo em que `MULT_FACTOR` é um número inteiro.