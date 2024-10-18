#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=10

#SBATCH --partition=normal

#SBATCH --mem=16GB

#SBATCH --time=0-02:00:00

#SBATCH --job-name=training_cross_val

#SBATCH --mail-type=ALL

#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
 # training data path
training_data="$WORK"/tiny_ai/data/ds4_with_combined
# epochs and learning rate
epochs=2
lr=0.0011
# Number of folds
num_folds=6
# name given to model
model_name=mobn_v2_"$epochs"_lr_"$lr"_nf_"$num_folds"
# min delta and patience for early stopping
min_d=0.001
patience=10

# test 
python "$wd"/train_keras_crossv.py --data_folder "$training_data" --pt_weights "$weights" --model_name "$model_name" --batch_size 120 --epochs $epochs --learning_rate $lr --min_delta $min_d --patience $patience --num_folds $num_folds

