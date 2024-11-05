#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=gpuv100

#SBATCH --mem=24GB

#SBATCH --time=0-00:30:00

#SBATCH --job-name=testing_keras

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/output/wheat_det/test/test_keras%j.log

#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
WORK=/scratch/tmp/kwundram/
# test data path
test_data="$WORK"/tiny_ai/test_data/Test_data
# enter all model names that need to be tested
model_names=("29.10.2024/mobn_v2_80_lr_0.0015_bs_80_t_29.10.2024-09:34:39_dim_800")
image_dim=800
batch_size=80

for model in ${model_names[@]}; do
    # keras model path
    keras_model="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/keras_models/"$model"/model.keras
    # test 
    python "$wd"/test_keras_model.py --keras_savepath "$keras_model" --model_name "$model" --testdata_path "$test_data" --image_dim $image_dim --batch_size $batch_size
done
