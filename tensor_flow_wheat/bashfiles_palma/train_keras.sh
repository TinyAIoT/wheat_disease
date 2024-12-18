#!/bin/bash -l
#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=normal
# try express, normal , long
#SBATCH --mem=30GB

#SBATCH --time=0-02:30:00

#SBATCH --job-name=training

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/output/wheat_det/training/train_keras_%j.log
# sbatch /home/k/kwundram/tiny_ai_home/wheat_disease/tensor_flow_wheat/bashfiles_palma/train_keras.sh
#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
# training data path
#training_data="$WORK"/tiny_ai/data/ds4_with_combined
training_data="$WORK"/tiny_ai/data/long_2023_999
save_path="$WORK"/tiny_ai/results/
time=`date +%d.%m.%Y_%H-%M-%S`
# epochs and learning rate
epochs=80
lr=0.0018
# 
batch_size=160
test_model=True
# image_dim for MobileNetv2 : [96, 128, 160, 182, 224]
image_dim=224
# used base model : [mobile_net_v3_s,mobile_net_v3_l,mobile_net_v2,effnet_v2_b3]
base_model=mobile_net_v2
weights="imagenet"
#include_top=false
# name given to model
model_name="$base_model"_"$epochs"_lr_"$lr"_bs_"$batch_size"_t_"$time"_dim_"$image_dim"
# min delta and patience for early stopping
min_d=0.0005
patience=10
# number of classes
num_classes=5
#
dropout=0.1
# sbatch /home/k/kwundram/tiny_ai_home/wheat_disease/tensor_flow_wheat/bashfiles_palma/train_keras.sh
python "$wd"/train_keras_sequential.py --data_folder "$training_data" --dropout $dropout --base_model $base_model --num_classes $num_classes --save_path "$save_path" --pt_weights "$weights" --model_name "$model_name" --image_dim "$image_dim" --batch_size $batch_size --epochs $epochs --learning_rate $lr --min_delta $min_d --patience $patience --test_model $test_model

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
mv "$WORK"/output/wheat_det/training/train_keras_"$SLURM_JOB_ID".log "$WORK"/output/wheat_det/training/"$model_name".log
