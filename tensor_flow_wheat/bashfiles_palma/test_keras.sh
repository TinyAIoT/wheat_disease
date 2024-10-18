#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=normal

#SBATCH --mem=16GB

#SBATCH --time=0-01:00:00

#SBATCH --job-name=testing_keras

#SBATCH --output=/scratch/tmp/b_kari02/output/wheat_det/testing/test_keras

#SBATCH --mail-type=ALL

#SBATCH --mail-user=b_kari02@uni-muenster.de

#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd=/scratch/tmp/b_kari02/wheat_disease/tensor_flow_wheat/
# keras model path
keras_model=/scratch/tmp/b_kari02/wheat_disease/tensor_flow_wheat/keras_models/mobn_v2_60Epochs/model.keras
# test data path
test_data=/scratch/tmp/b_kari02/data/test_data

# test 
python "$wd"/test_keras_model.py --keras_savepath "$keras_model" --testdata_path "$test_data" --batch_size 120
