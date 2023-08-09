Para a execução local do POV é necessário:

* Instalar as dependencias do sistema (no meu caso gawk)
    * sudo apt-get install gawk

* Dependências existentes do python estão listadas no arquivo `requirements` 
    * segyio está listado mas não é utilizado ainda

* Instalar as dependencias do python 
    * pip install -r requirements ou pip3 install -r requirements

Após a refatoração é necessário colocar os arquivos com os dados no diretório `dados` e então executar o exec.
* scripts
    * dados
        * near.npy
        * mid.npy
        * far.npy
        * ufar.npy
        * etc

Os arquivos .npy acima estão disponíveis, atualmente, na [página do professor Adriano](https://homepages.dcc.ufmg.br/~adrianov/pov/). Não estão aqui no GitLab pois são muito grandes e o GitLab não aceita.

**Dependendo do sistema (como no meu caso, onde existe o python 2.7 e o python 3.xx), alterar no script exec de python para python3**

* Executar o código na versão original:
    * bash exec
teste

## Geração de Features
Segue uma breve descrição de como computar as features de coerência e curvatura para um dado arquivo com extensão ```.npy```. É necessario o uso de python com o seguintes pacotes instalados:

- Numpy
- Scipy
- joblib (apenas caso queira usar mais de um núcleo nos algoritmos que usam janelas 3D)

Em seguida, basta chamar o programa ```make_features.py``` para um arquivo de entrada ```arquivo.npy``` com uma feature desejada, nesse caso duas features, ```gersz``` e ```mean_curvature```:

```bash
python3 src/data/features/make_features.py arquivo.npy gersz mean_curvature --window3d 3 3 5
```

O primeiro algoritmo, ```gersz```, possui um parâmetro, o tamanho ```window3d``` da janela deslizante. Ele possui um valor padrão mas posso modificá-lo facilmente. Abaixo temos todos os parâmetros disponíveis:
|Parâmetro|Descrição|
|:---|:---|
|```filename```|Arquivo ```.npy``` de entrada.|
|```algorithm```|Qual o algoritmo a ser usado.|
|```-o```, ```--output```|Pasta de destino do arquivo gerado. Por padrão é o diretório atual.|
|```-w1```, ```--window1d```|Tamanho da janela unidimensional para o RMS. Por padrão é ```5```.|
|```-w3```, ```--window3d```|Tamanho da janela tridimensional para algoritmos de janela deslizante. Por padrão é ```3 3 9```.|
|```-n_cpu```|Número de núcleos disponíveis para o cálculo das features que usam janela deslizante 3D. Por padrão é apenas ```1``` núcleo.|

As features que podem ser calculadas são as seguintes:

|Nome|Descrição|
|:---|:--|
|```dip_angle```|Ângulo do maior declive|
|```azimuth```|Ângulo de azimute|
|```mean_curvature```|Curvatura média|
|```gaussian_curvature```|Curvatura Gaussiana|
|```max_curvature```|Maior curvatura em módulo|
|```min_curvature```|Menor curvatura em módulo|
|```most_positive_curvature```|Maior curvatura|
|```most_negative_curvature```|Menor curvatura|
|```shape_index```|Índice de forma|
|```dip_curvature```|Curvatura na direção de maior declive|
|```contour_curvature```|Curvatura da curva de nível|
|```curvedness```|Intensidade de curvatura|
|```instFrequency```|Frequência instantânea|
|```envelope```|Extremos de um sinal oscilante|
|```rms```|Raiz quadrada da média dos valores ao quadrado dentro de uma janela 1D|
|```marfurt```|Soma da matriz de covariância dividida pelo seu traço, em uma janela deslizante 3D|
|```sobel```|Filtro sobel 3D sobre o ```gersz```|
|```gersz```|Autovalor da primeira componente de um PCA na janela 3D|
|```gst```|*Gradient Structure Tensor*|
|```median```|Mediana de uma janela deslizante 3D|
|```mean```|Média de uma janela deslizante 3D|
|```max```|Máximo de uma janela deslizante 3D|
|```min```|Mínimo de uma janela deslizante 3D|
|```sum```|Soma de uma janela deslizante 3D|