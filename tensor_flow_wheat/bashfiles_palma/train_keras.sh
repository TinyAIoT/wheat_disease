#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=normal

#SBATCH --mem=16GB

#SBATCH --time=0-01:00:00

#SBATCH --job-name=training

#SBATCH --output=/scratch/tmp/b_kari02/output/wheat_det/training/train_keras

#SBATCH --mail-type=ALL

#SBATCH --mail-user=b.karic@uni-muenster.de

#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd=/scratch/tmp/b_kari02/wheat_disease/tensor_flow_wheat/
# training data path
training_data=/scratch/tmp/b_kari02/data/ds4_with_combined
# epochs and learning rate
epochs=60
lr=0.0001
# name given to model
model_name=mobn_v2_"$epochs"_lr_"$lr"
# test 
python "$wd"/train_keras_sequential.py --data_folder "$training_data" --model_name "$model_name" --batch_size 120 --epochs $epochs --learning_rate $lr

