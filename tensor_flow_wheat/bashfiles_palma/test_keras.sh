#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=normal

#SBATCH --mem=16GB

#SBATCH --time=0-01:00:00

#SBATCH --job-name=testing_keras

#SBATCH --output=/scratch/tmp/kwundram/output/wheat_det/testing/test_keras

#SBATCH --mail-type=ALL

#SBATCH --mail-user=kwundram@uni-muenster.de

#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd=/scratch/tmp/kwundram/tiny_ai/wheat_repo/wheat_disease/tensor_flow_wheat/
# test data path
test_data=/scratch/tmp/kwundram/tiny_ai/test_data/Test_data
# enter all model names that need to be tested
model_names=("mobn_v2_150_lr_0.0011" "mobn_v2_150_lr_0.002" "mobn_v2_150_lr_0.0025" "mobn_v2_150_lr_0.01")

for model in ${model_names[@]}; do
    # keras model path
    keras_model=/scratch/tmp/kwundram/tiny_ai/wheat_repo/wheat_disease/tensor_flow_wheat/keras_models/"$model"/model.keras
    # test 
    python "$wd"/test_keras_model.py --keras_savepath "$keras_model" --testdata_path "$test_data" --batch_size 120
done
