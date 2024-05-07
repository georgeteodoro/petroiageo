
# Objetivo
Os códigos dessa pasta se referem à preparação dos dados de porosidade distribuídos em vários poços para serem servidos ao algoritmo. 

## Arquivos necessários
É esperado que existam os seguintes arquivos de antemão:

- Arquivo csv com nome e coordenada de poços (`$WELLS_INFO_PATH`): Esse arquivo possui informações sobre os poços contidos no cubo sísmico como nome e coordenadas x e y relativas ao cubo sísmico. Deve ter as colunas: "Well, x_coord, y_coord, min_z, max_z". Well é simplesmente o nome do poço. x_coord e y_coord devem ser valores inteiros não negativos. min_z e max_z podem ser valores de ponto flutuante com `min_z < max_z`.

- Arquivos de porosidades ao longo dos poços: Esses arquivos possuem as medições das diferentes porosidades ao longo dos poços. Dessa forma, possuem a profundidade e medição de até três porosidades diferentes (densidade, sônica e neutrônica). Para o arquivo referente a cada poço, deve ter as seguintes colunas em ordem: "depth, neutron_por, density_por, sonic_por". Todas essas colunas podem ser valores de ponto flutuante. É esperado que esses arquivos não possuam linha de cabeçalho com os nomes das colunas. É esperado que os valores estejam separados por um espaço ' ' e não vírgula. 

- Áreas alvo e coordenadas (`$AREAS_COORDS_PATH`): Arquivo csv que delimita áreas de interesse no cubo sísmico baseado na concentração de poços. Criado a partir da análise da distribuição dos poços na área sísmica. De ter as seguintes colunas: "area, max_x, min_x, min_y, max_y". area é a identificação da área, podendo ser de qualquer tipo. As outras colunas devem ser valores inteiros não negativos com `min_x < max_x` e `min_y < max_y`

- Cubo sísmico (`$SEISMIC_FILE`): Arquivo npy de três dimensões com as medidas sísmicas com resolução de `SEISMIC_RESOLUTION` metros na dimensão de profundidade. Deve-se saber qual é o valor da resolução `SEISMIC_RESOLUTION`. É esperado que esse valor seja um valor inteiro positivo.

## Agregação dos dados de porosidade por área

Primeiramente, devemos agregar os arquivos de porosidade dos poços que residem dentro de uma mesma área alvo definida no arquivo de áreas alvo. Isso pode ser feito executando o seguinte script:
`python3 agg_area_por_files.py --target_area TARGET_AREA --areas_coords_path AREAS_COORDS_PATH --wells_info_path WELLS_INFO_PATH --wells_por_dir_path WELLS_POR_DIR_PATH --final_df_dir_path FINAL_DF_DIR_PATH`

A explicação de cada um desses parâmetros pode ser visto com:`python3 agg_area_por_files.py -h`. Mas, basicamente, TARGET_AREA é o nome da área alvo de onde agregar os arquivos de porosidade (se for igual a -1, executa para todas as áreas alvo), AREAS_COORDS_PATH é o caminho para o arquivo que possui as áreas e suas coordenadas, WELLS_INFO_PATH é o caminho para o arquivo que possui informações sobre cada poço, WELLS_POR_DIR_PATH é o caminho para a pasta que possui os arquivos com as porosidades de cada poço e FINAL_DF_DIR_PATH é o caminho para a pasta que receberá os arquivos agregados.
## Filtrar e tratar porosidades
Aqui, será realizado a filtragem por intervalo de profundidade, a interpolação e possível agregação dos dados de porosidade de uma área. Isso pode ser feito executando o seguinte script:
`python3 treat_porosity_file.py --min_depth MIN_DEPTH --max_depth MAX_DEPTH --por_file POR_FILE --target_por_file TARGET_POR_FILE --agg_strat {M_O_M,NONE,M_O_R} --rolling_w ROLLING_W`

A explicação para cada uma das opções pode ser vista ao executar `python3 treat_porosity_file.py -h`. Mas, basicamente, MIN_DEPTH e MAX_DEPTH definem o intervalo de profundidade onde deverão ser filtrados os dados de porosidade de cada poço, POR_FILE é o caminho do arquivo que possui os dados de porosidade da área alvo produzido pelo script anterior, TARGET_POR_FILE é o caminho alvo do arquivo csv a ser gerado, agg_strat deve receber uma de três opções {M_O_M, NONE, M_O_R} onde NONE não realiza nenhuma agregação sobre os dados de porosidade, M_O_M gera pontos para cada intervalo de metro de profundidade baseado na média dos pontos nesse intervalo e M_O_R aplica uma média móvel de acordo com a janela indicada em ROLLING_W. É esperado que ROLLING_W seja um valor inteiro positivo se utilizado.
## Junção dos dados de porosidade com a sísmica
Aqui, será realizado o casamento entre os dados de porosidade tratados com o cubo sísmico. O cubo sísmico será filtrado de acordo com a área delimitada por uma área alvo e também será filtrado pelo intervalo de profundidade de acordo com a presença de dados de porosidade para essa área. Além disso, os valores sísmicos serão interpolados para que casem com as medições de porosidade. Isso pode ser feito executando o seguinte script:
`python3 filter_and_merge_seismic_por.py --por_file POR_FILE --area_info_file AREA_INFO_FILE --target_area TARGET_AREA --seismic_file SEISMIC_FILE --seismic_resolution SEISMIC_RESOLUTION --start_seismic_depth START_SEISMIC_DEPTH --target_seismic_file TARGET_SEISMIC_FILE --target_merge_file TARGET_MERGE_FILE`

A explicação de cada parâmetro pode ser vista executando `python3 filter_and_merge_seismic_por.py -h`. Mas, basicamente, POR_FILE é o caminho para o arquivo de tratado gerado pelo script anterior, AREA_INFO_FILE é o caminho para o arquivo que possui informações sobre cada área alvo, TARGET_AREA é o nome/identificador da área alvo, SEISMIC_FILE é o caminho para o arquivo sísmico `.npy`, SEISMIC_RESOLUTION é o valor da resolução sísmica no eixo z, START_SEISMIC_DEPTH é a profundidade inicial real em metros presente no arquivo sísmico, TARGET_SEISMIC_FILE é o caminho alvo do arquivo sísmico tratado a ser gerado e TARGET_MERGE_FILE é o caminho alvo do arquivo csv que contém as porosidades e a sísmica relacionada para cada poço da área alvo.  

