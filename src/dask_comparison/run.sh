#module load anaconda3.2023.09-0
#conda activate /snfs2/willianjunior/git/petroiageo/src/dask_comparison/dask-env

source /snfs2/willianjunior/git/petro-env/bin/activate

#python3 -m pip install dask distributed

#python3 -m pip list

python3 -u run_dask.py
