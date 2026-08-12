#module load anaconda3.2023.09-0
#source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate /snfs2/willianjunior/git/petroiageo/src/dask_comparison/dask-env

source /snfs2/willianjunior/git/petro-env/bin/activate


dask worker tcp://127.0.0.1:8786
