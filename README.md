# Wheat Diseas Detection

## from Scratch in /tensor_flow_wheat

-data found in sciebo in /Use cases/Landwirtschaft/wheat_disease_raw_data

cd ./tensor_flow_wheat

conda env create -f ./tf_2.13_env.yml -n ENV_NAME

conda activate ENV_NAME

python ./train_keras_sequential.py ( with set flags)

### Using Pama bash files

#### Train Keras model

1. edit train_keras.sh (uses train_keras_sequential.py)
2. sbatch path/to/bashfiles_palma/train_keras.sh

#### Test Keras model

1. edit test_keras.sh (uses test_keras_model.py)
2. sbatch path/to/bashfiles_palma/test_keras.sh

#### Convert Keras to tflite

1. edit convert_to_tflite.sh (uses convert_to_tflite.py)
2. sbatch path/to/bashfiles_palma/convert_to_tflite.sh

------------------------------------------------------------------------------------------------
## Edge Impulse Copy
### data 
Y_split already in /data. 
X_split can be downloaded from sciebo in Use cases/Landwirtschaft/edge_impulse_train_data

### create conda env ( cd to edgeimpulse_copy)

### On Palma

salloc --nodes 1 --cpus-per-task 8 -t 03:00:00 --mem 32G --gres gpu:1 --partition gpuexpress

module purge
module load palma/2021a Miniconda3/4.9.2
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda deactivate

conda env create -f ./wheat_dis_env.yaml -n ENV_NAME

conda activate ENV_NAME