#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=gpua100
# zen3 : try gpu2080, gpua100
#gpuexpress, gpu3090, gputitanrtx, gpuhgx

#SBATCH --gres=gpu:1

# max mem/node : 120 GB
#SBATCH --mem=128GB

#SBATCH --time=0-02:30:00

#SBATCH --job-name=training_cross_val_z3

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/output/wheat_det/training/train_cross_val_z3%j.log


# ZEN3 module load
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
# save path for 
save_path="$WORK"/tiny_ai/results/
time=`date +%d.%m.%Y_%H-%M-%S`
# epochs and learning rate
epochs=3
lr=0.01
# number of classes
num_classes=5
# Number of folds
num_folds=5
# test ds size
test_ds_size=0.2
# image_dim for MobileNet: [96, 128, 160, 182, 224]
# if no weights path given also other image dimensions can be used. e.g.: [320,480,640,800]
image_dim=224
# decrease image dim or batch_size when OOM (Out of Memory)
batch_size=40
# used base model : [mobile_net_v3_s,mobile_net_v3_l,mobile_net_v2,effnet_v2_b3]
base_model=mobile_net_v3_s
include_top=0
# change weights accordingly to the base model
#weights="$WORK"/transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
# name given to model
weights="imagenet"
model_name="$base_model"_"$epochs"_lr_"$lr"_nf_"$num_folds"_t_"$time"_dim_"$image_dim"
# using zen3 partition ?
zen3=True

export TF_GPU_ALLOCATOR=cuda_malloc_async
echo $TF_GPU_ALLOCATOR
# min delta and patience for early stopping
min_d=0.0001
patience=10

# run code with flags
# sbatch --mail-user "kwundram@uni-muenster.de" $HOME/tiny_ai_home/wheat_disease/tensor_flow_wheat/bashfiles_palma/zen3/train_cross_val_zen3.sh

python "$wd"/train_keras_crossv.py --data_folder "$training_data" --test_ds_size $test_ds_size --include_top $include_top --base_model $base_model --num_classes $num_classes  --save_path "$save_path" --pt_weights "$weights" --model_name "$model_name" --batch_size "$batch_size" --epochs $epochs --learning_rate $lr --min_delta $min_d --patience $patience --num_folds $num_folds --image_dim "$image_dim" --zen3 $zen3

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
mv "$WORK"/output/wheat_det/training/train_cross_val_z3"$SLURM_JOB_ID".log "$WORK"/output/wheat_det/training/"$model_name".log
