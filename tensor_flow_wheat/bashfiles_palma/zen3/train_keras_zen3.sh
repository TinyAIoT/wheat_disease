#!/bin/bash -l
#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=gpua100
# try gpuv100, gpu2080, gpua100
#SBATCH --mem=64GB

#SBATCH --time=0-02:00:00

#SBATCH --job-name=training_keras_seq__zen3

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/output/wheat_det/training/train_keras_%j.log

#load modules 
module purge
module load palma/2022a  GCC/11.3.0  OpenMPI/4.1.4
module load TensorFlow/2.11.0
module load scikit-learn/1.1.2

# place of code in palma
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
# training data path
#training_data="$WORK"/tiny_ai/data/ds4_with_combined
training_data="$WORK"/tiny_ai/data/long_2023_999
save_path="$WORK"/tiny_ai/results/
time=`date +%d.%m.%Y_%H-%M-%S`
# epochs and learning rate
epochs=11
lr=0.001
# 
batch_size=80
test_model=True
# image_dim for MobileNetv2 : [96, 128, 160, 182, 224]
# [320, 480, 540, 700]
image_dim=320
# used base model : [mobile_net_v3_s,mobile_net_v3_l,mobile_net_v2,effnet_v2_b3]
base_model=mobile_net_v3_s
include_top=True
# name given to model
model_name="$base_model"_"$epochs"_lr_"$lr"_bs_"$batch_size"_t_"$time"_dim_"$image_dim"
# min delta and patience for early stopping
min_d=0.0005
patience=10
# number of classes
num_classes=5
zen3=True
# dropout 
dropout=0.3
export TF_GPU_ALLOCATOR=cuda_malloc_async
echo $TF_GPU_ALLOCATOR
# test 
python "$wd"/train_keras_sequential.py --data_folder "$training_data" --base_model $base_model --dropout $dropout --num_classes $num_classes --save_path "$save_path" --pt_weights "$weights" --model_name "$model_name" --image_dim "$image_dim" --batch_size $batch_size --epochs $epochs --learning_rate $lr --min_delta $min_d --patience $patience --test_model $test_model --zen3 $zen3

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
mv "$WORK"/output/wheat_det/training/train_keras_"$SLURM_JOB_ID".log "$WORK"/output/wheat_det/training/"$model_name".log
# sbatch /home/k/kwundram/tiny_ai_home/wheat_disease/tensor_flow_wheat/bashfiles_palma/zen3/train_keras_zen3.sh