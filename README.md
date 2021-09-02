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

**Dependendo do sistema (como no meu caso, onde existe o python 2.7 e o python 3.xx), alterar no script exec de python para python3**

* Executar o código na versão original:
    * bash exec
