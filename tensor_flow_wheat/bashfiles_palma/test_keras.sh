#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=normal

#SBATCH --mem=16GB

#SBATCH --time=0-01:00:00

#SBATCH --job-name=testing_keras

#SBATCH --mail-type=ALL
#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
# test data path
test_data="$WORK"/tiny_ai/test_data/Test_data
# enter all model names that need to be tested
model_names=("mobn_v2_2_lr_0.0011")

for model in ${model_names[@]}; do
    # keras model path
    keras_model="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/keras_models/"$model"/model.keras
    # test 
    python "$wd"/test_keras_model.py --keras_savepath "$keras_model" --model_name "$model" --testdata_path "$test_data" --batch_size 120
done
