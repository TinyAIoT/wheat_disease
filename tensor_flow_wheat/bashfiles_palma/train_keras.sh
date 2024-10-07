#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=normal

#SBATCH --mem=16GB

#SBATCH --time=0-01:00:00

#SBATCH --job-name=training

#SBATCH --output=/scratch/tmp/kwundram/output/wheat_det/training/train_keras

#SBATCH --mail-type=ALL

#SBATCH --mail-user=kwundram@uni-muenster.de

#load modules with available GPU support (this is an example, modify to your needs!)
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd=/scratch/tmp/kwundram/tiny_ai/wheat_repo/wheat_disease/tensor_flow_wheat/
# training data path
training_data=/scratch/tmp/kwundram/tiny_ai/data/ds4_with_combined
# name given to model
model_name=mobn_v2_60Epochs

# test 
python "$wd"/train_keras_sequential.py --data_folder "$training_data" --model_name "$model_name" --batch_size 120 --epochs 60 --learning_rate 0.001

