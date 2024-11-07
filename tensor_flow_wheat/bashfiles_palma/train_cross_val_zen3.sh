#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=10

#SBATCH --partition=gpu2080

#SBATCH --gres=gpu:4

#SBATCH --mem=64GB

#SBATCH --time=0-02:00:00

#SBATCH --job-name=training_cross_val_z3

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/output/wheat_det/training/train_cross_val_z3%j.log

#load modules 
module purge
module load palma/2022a  GCC/11.3.0  OpenMPI/4.1.4
module load TensorFlow/2.11.0
module load scikit-learn/1.1.2

TF_GPU_ALLOCATOR=cuda_malloc_async
# place of code in palma
WORK=/scratch/tmp/kwundram/
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
 # training data path
#training_data="$WORK"/tiny_ai/data/ds4_with_combined
training_data="$WORK"/tiny_ai/data/long_2023_999
save_path="$WORK"/tiny_ai/results/
time=`date +%d.%m.%Y-%H:%M:%S`
# epochs and learning rate
epochs=30
lr=0.0020
# Number of folds
num_folds=3
image_dim=480
batch_size=60
# name given to model
model_name=mobn_v2_"$epochs"_lr_"$lr"_nf_"$num_folds"_t_"$time"_dim_"$image_dim"
# min delta and patience for early stopping
zen3=True
min_d=0.001
patience=10

# test 
python "$wd"/train_keras_crossv.py --data_folder "$training_data" --save_path "$save_path" --pt_weights "$weights" --model_name "$model_name" --batch_size "$batch_size" --epochs $epochs --learning_rate $lr --min_delta $min_d --patience $patience --num_folds $num_folds --image_dim "$image_dim" --zen3 $zen3

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
mv "$WORK"/output/wheat_det/training/train_cross_val_z3"$SLURM_JOB_ID".log "$WORK"/output/wheat_det/training/"$model_name".log
