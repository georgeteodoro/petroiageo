Para a execução local do POV é necessário:

* Instalar as dependencias do sistema (no meu caso gawk)
    * sudo apt-get install gawk


* Instalar as dependencias do python3 
    * pip install numpy ou pip3 install numpy
    * pip install pandas ou pip3 install pandas 
    * pip install segyio ou pip3 install segyio (import não utilizado até o momento)
    * pip install scipy ou pip3 install scipy
    * pip install scikit-learn ou pip3 install scikit-learn

**Dependendo do sistema (como no meu caso, onde existe o python 2.7 e o python 3.xx), alterar no script exec de python para python3**

* Executar o código na versão original:
    * bash exec

Após a refatoração é necessário colocar os arquivos com os dados no diretório `dados` e então executar o exec.
* scripts
    * dados
        * near.npy
        * mid.npy
        * far.npy
        * ufar.npy
        * etc
