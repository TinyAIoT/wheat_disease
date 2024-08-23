# Wheat Diseas Detection

## data 
Y_split already in /data. 
X_split can be downloaded from sciebo in Use cases/Landwirtschaft/edge_impulse_train_data

## create conda env ( cd to edgeimpulse_copy)

## On Palma

salloc --nodes 1 --cpus-per-task 8 -t 03:00:00 --mem 32G --gres gpu:1 --partition gpuexpress

module purge
module load palma/2021a Miniconda3/4.9.2
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda deactivate

conda env create -f ./wheat_dis_env.yaml -n ENV_NAME

conda activate ENV_NAME

## train.py : add full paths 
python ./train.py --data-directory "./data" --out-directory "./output"


## finding akida
akida is imported in resources/libraries/ei_tensorflow/brainchip/edge_learning.py and model.py

module load palma/2023a  GCCcore/12.3.0 Python/3.11.3    kein akida

module load palma/2021a  GCCcore/10.3.0 Python/3.9.5     akida versions: 2.0.0, 2.0.1, 2.0.2, 2.0.3, 2.0.4, 2.0.5, 2.1.0,
2.1.1, 2.1.2, 2.1.3, 2.1.4, 2.1.5, 2.1.6, 2.2.0, 2.2.1, 2.2.2

module load palma/2022a  GCCcore/11.3.0 Python/3.10.4    kein akida
module load palma/2022b  GCCcore/12.2.0 Python/3.10.8    kein akida