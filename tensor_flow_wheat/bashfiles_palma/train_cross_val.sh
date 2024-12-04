#!/bin/bash -l

#SBATCH --nodes=1

#SBATCH --tasks-per-node=1

#SBATCH --gpus-per-node=2

#SBATCH --partition=gpuexpress

#SBATCH --mem=30GB

#SBATCH --time=0-01:00:00

#SBATCH --job-name=training_cross_val

#SBATCH --mail-type=ALL

#SBATCH --mail-user=b_kari02@uni-muenster.de

#SBATCH --output /scratch/tmp/%u/output/wheat_det/training/train_cross_val_%j.log

#load modules 
module purge
ml palma/2022a  GCC/11.3.0 OpenMPI/4.1.4
ml TensorFlow/2.11.0-CUDA-11.7.0
# module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
# module load TensorFlow/2.13.0
module load scikit-learn/1.1.2

# place of code in palma
WORK=/scratch/tmp/b_kari02/
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
 # training data path
#training_data="$WORK"/tiny_ai/data/ds4_with_combined
training_data="$WORK"/tiny_ai/data/long_2023_999
save_path="$WORK"/tiny_ai/results/
time=`date +%d.%m.%Y-%H:%M:%S`
# epochs and learning rate
epochs=10
lr=0.0015
batch_size=80
# Number of folds
num_folds=3
# Image dimension
image_dim=800
# name given to model
model_name=mobn_v2_"$epochs"_lr_"$lr"_nf_"$num_folds"_t_"$time"_dim_"$image_dim"
# min delta and patience for early stopping
min_d=0.0005
patience=10

# test 
python "$wd"/train_keras_crossv.py --data_folder "$training_data" --save_path "$save_path" --pt_weights "$weights" --model_name "$model_name" --batch_size $batch_size --epochs $epochs --learning_rate $lr --min_delta $min_d --patience $patience --num_folds $num_folds

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
mv "$WORK"/output/wheat_det/training/train_cross_val_"$SLURM_JOB_ID".log "$WORK"/output/wheat_det/training/"$model_name".log
