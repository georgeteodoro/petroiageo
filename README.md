# Geral

A solução de aprendizado invertido consiste em uma aplicação *Python3* que expande o hipercubo de pontos, gerando novos valores de porosidade. A aplicação tem suporte a execução serializada (1 core de CPU) ou com paralelismo local e distribuído via MPI.

# Versionamento

Para facilitar o processo de resolução de eventuais problemas de código é pedido que apenas tags sejam usadas. Embora a branch *main* esteja atualizada na grande maioria das vezes, isso não é grarantido a todo momento, principalmente em momentos de entregas, onde existem atualizações.

# Requisitos - Libs

As seguintes libs são necessárias para execução em ambiente linux:
 - python3
 - python3-pip
 - python3-dev
 - mpi
 - libopenmpi-dev

A aplicação foi testada com o OpenMPI, porém podem ser usadas outras implementações do MPI.

## Requisitos - Libs - HDF5 e h5py

Para a gestão de arquivos grandes por meio de execução Out-of-Core são usadas as libs hdf5 e h5py. Para permitir seu uso em conjunto ao MPI é necessária a compilação manual dessas duas ferramentas. Para a instalação dessas libs não e necessário acesso *root*, exceto para os comandos *pip3*. Porém é possível fazê-los usando um ambiente virtual, como *anaconda* ou *venv*, sem permissões *root*.

A compilação do HDF5 pode ser feita da seguinte forma:

    git clone https://github.com/HDFGroup/hdf5.git
    cd hdf5
    git checkout hdf5-1_12_2-3-rc1
    autoconf
    ./configure --enable-parallel --enable-shared --prefix=<HDF5_PATH>/build
    make -j8
    make install

A compilação/instalação do h5py, a ser feita após o HDF5, sendo feita usando os comandos:

    pip3 uninstall mpi4py
    pip3 install mpi4py==3.1.3
    git clone https://github.com/h5py/h5py.git
    cd h5py
    git checkout 3.8.0
    pip3 install wheel Cython==3.0.0a11 numpy
    export CC=mpicc; export HDF5_MPI="ON"; export HDF5_DIR="<HDF5_PATH>/build"; pip3 install --no-build-isolation .

O h5py pode ter problemas de incompatibilidade com o mpi4py. Para evitar problemas certificar que a versão do mpi4py seja a 3.1.3. Isso é feito nos comando acima, onde uma versão previamente instalada do mpi4py é removida, sendo instalada a versão esperada.

# Requisitos - Dados

A execução da aplicação usa dois tipos de dados: dados sísmicos e dados reais de porosidade de poços. Os dados reais de poço são pequenos, e por isso já estão disponíveis nesse repositório privado. Os dados sísmicos são maiores e assim não constam no repositório. Esses arquivos *numpy* devem ser colocados no diretório **./data/**, contido na raiz deste projeto. Abaixo temos a lista de arquivos necessários para execução da aplicação que devem ser colocados no diretório **./data**:
 - porosity-canal.npy
 - features/NEAR.npy
 - features/MID.npy
 - features/FAR.npy
 - features/UFAR.npy

Os dados sísmicos (*NEAR.npy*, *MID.npy*, *FAR.npy* e *UFAR.npy*) são arquivos *numpy*, contendo uma matriz de *(434,646,251)* pontos (dimensão do hipercubo), com valores do tipo *numpy.float64*.

## Geração de arquivos h5

Os dados *.npy* deverão ser convertidos para o formato *.h5*. Isso é feito por meio dos seguintes comandos, executados a partir da raiz do projeto:

    cd src/data
    python3 wells_data3_hdf5.py
    cd features
    python3 seismic_data3_hdf5.py

Esses comandos acima gerarão arquivos *.h5* que deverão estar nos diretórios abaixo mostrados:
 - ./data/POV/porosity-canal.h5
 - ./data/POV/features/h5_features/NEAR.npy
 - ./data/POV/features/h5_features/MID.npy
 - ./data/POV/features/h5_features/FAR.npy
 - ./data/POV/features/h5_features/UFAR.npy

Os scripts *Python* já geram os dados nos locais corretos, porém é importante verificar a existência dos mesmos.

# Dependências *Python*

A aplicação usa diversas libs *python* para sua execução. Todas elas (exceto h5py, já instalada anteriormente) podem ser facilmente instaladas por meio do seguinte comando:

    pip3 install -r requirements

Vale ressaltar que essas dependências devem ser instaladas somente após as libs Linux necessárias.

Caso não tenha acesso *root*, é possível realizar essa instalação em um ambiente virtual suportado pelo ambiente de execução, como *anaconda* ou *venv*.

# Execução

Existem duas formas de executar a aplicação, serializada com um único processo ou com paralelismo. Dado o custo computacional envolvido é recomendado usar a execução serializada apenas para validar a instalação da aplicação. A execução serializada é feita por meio do comando a seguir a partir do diretório **./src/alg** desse repositório:

    python3 main.py --config config.yaml [params]

Para execução paralela basta executar o seguinte comando para gerar N processos paralelos:

    mpirun -np N --tag-output --bind-to core python3 main.py --config config.yaml [params]

Esses N processos paralelos são distribuídos pelos recursos disponíveis. Então, para uma única máquina com 10 núcleos e N=20, 20 processos são inicializados nesta. Se houverem duas máquinas configuradas em ambiente distribuído, haverão 10 processos por máquina. Não é recomendado usar mais processos por máquina do que núcleos de CPU que esta possui.

Outro ponto de atenção é que essa aplicação consome uma grande quantidade de memória, e dessa forma valores muito grandes de N podem resultar em execuções interrompidas por falta de memória.

Os dados de estimação de porosidade são salvos em **./data/POV/porosity-canal.h5**. Dados relativos à performance da aplicação (e.g., tempos de execução) e à performance do modelo gerado (e.g., acurácia do modelo) são visto na saída padrão do sistema (*stdout*).

# Parâmetros de Execução

Nessa versão da aplicação existem parâmetros de entrada que podem ser usados para configurar como a geração de dados pode ser feita. A aplicação também conta com um parâmetro de "ajuda": *--help* ou *-h*. Abaixo alguns parâmetros disponíveis:
 - *--it*: Iteração inicial a ser executada. Exemplo, se tiver o valor 3, é esperado que as iterações 1 e 2 tenham sido concluídas, com os resultados no arquivo de porosidade, sendo assim executada a iteração 3 em diante. Nota: a primeira iteração é a iteração 1.
 - *--nits*: Define o número de iterações que serão realizadas, incluindo a iteração inicial. Caso a aplicação não consiga terminar sua execução (e.g., foi cancelada por uso excessivo de memória ou por *timeout* em *clusters*), os dados parciais dessa execução são **PERDIDOS**. Para evitar tal problema em ambientes de *clusters* é recomendado a execução de poucas iterações por vez, realizando o backup dos arquivos de porosidade a cada termino.
 - *--nf*: Número de características a serem usadas, a partir das características base disponíveis em **./data/POV/features/h5_features**. Se não definido, todas as características base disponíveis são usadas. 
 - *--nfs*: Número de característcas selecionadas por iteração. Por padrão 10 características são usadas para gerar um modelo estimador de porosidade. Porém esse valor pode ser reduzido para diminuir o tempo de execução.

Além disso, um arquivo de configuração (*config.yaml*) é usado para configurar a aplicação a nível mais fino. Caso seja passado um parâmetro via argumento python que já tenha um valor definido em *config.yaml*, o valor da config é sobreescrito, sendo mantido o valor passado via parâmetro.

# Exemplo

Abaixo temos um exemplo rápido usando apenas 1 arquivo, selecionando apenas 1 melhor característica, para 1 iteração:

    python3 main.py --config config.yaml --nits 1 --nf 1 --nsf 1

A saída do comando acima será:

    loading seismic
    loading porosity
    [gen_expanded_points][it0] Expanding points on ring 0
    [PROFILING][expand][it0][chunk0-time] 5.728178262710571
    [PROFILING][expand][it0][chunk1-time] 6.6458210945129395
    [PROFILING][expand][it0][chunk0-time] 5.793959140777588
    [PROFILING][expand][it0][chunk1-time] 5.810708999633789
    [PROFILING][expand][it0][chunk0-time] 5.762479782104492
    [PROFILING][expand][it0][chunk0-time] 5.958775043487549
    [PROFILING][expand][it0][chunk0-time] 5.750436782836914
    [PROFILING][expand][it0][chunk0-time] 5.760072231292725
    [PROFILING][expand][it0][chunk0-time] 5.853853464126587
    [PROFILING][expand][it0][chunk0-time] 5.790080308914185
    [PROFILING][expand][it0][ran-chunks-time] 58.85436511039734
    [PROFILING][expand][it0][chunks-ran] 1 2
    ============== NEED TO AUTOMATE TMP_LIST CHUNK_SIZE
    [get_features_sets][('MID', -3, -3, -3)] insert_feature_time: 0.0773627758026123
    [get_features_sets][('MID', -3, -3, -3)] train_time: 0.4123711585998535
    [get_features_sets][('MID', -3, -3, -3)] error: 0.05177651273487806
    [get_features_sets][it0] Tested features ['x', 'y', 'z', ('MID', -3, -3, -3)] with error 0.05177651273487806
    [get_features_sets][it0] full_it_time: 0.48976802825927734
    [get_features_sets][('MID', -3, -3, -3)] commit_feature_time: 0.0767519474029541
    [get_features_sets] full_time: 3.45944881439209


# Execução Via Container

Uma outra opção de execução é por meio do Dockerfile presente neste repositório. Usando o container é possível executar a aplicação sem precisar se preocupar com os módulos/requisitos a serem baixados, e sem precisar de acesso *root*. Dois comandos são necessários (executando a partir da raiz deste repositório):

    podman build --tag app .
    podman run -v <GIT_PATH>/data:/home/petroiageo/data:Z app

O prímeiro comando gera a imagem do conteiner a ser executada. No segundo comando é necessário colocar o caminho absoluto deste repositório (GIT_PATH). É possível passar parâmetros da aplicação normalmente via CLI para a aplicação. Os dados sismícos e de porosidade reais devem estar presentes no diretório **./data** no sistema *host*. Assim, não são necessários movimentos de dados (cópias) entre a imagem do container e o sistema host. Esse exemplo foi testado com a aplicação *podman* para criação e execução de containers, porém tratando-se de um Dockerfile, pode ser executada usando o software de containers de sua preferência.
