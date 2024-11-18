#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=10

#SBATCH --partition=normal

#SBATCH --mem=64GB

#SBATCH --time=0-02:00:00

#SBATCH --job-name=training_cross_val

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/output/wheat_det/training/train_cross_val_%j.log

#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
WORK=/scratch/tmp/kwundram/
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
 # training data path
#training_data="$WORK"/tiny_ai/data/ds4_with_combined
training_data="$WORK"/tiny_ai/data/long_2023_999
save_path="$WORK"/tiny_ai/results/


time=`date +%d.%m.%Y-%H:%M:%S`
# epochs and learning rate
epochs=20
lr=0.0015
# Number of folds
num_folds=2
# image_dim for MobileNetv2 : [96, 128, 160, 182, 224]
image_dim=320
# used base model : [mobile_net_v3,mobile_net_v2]
base_model=mobile_net_v3
# change weights accordingly to the base model
weights="$WORK"/transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5

# name given to model
model_name="$base_model"_"$epochs"_lr_"$lr"_nf_"$num_folds"_t_"$time"_dim_"$image_dim"
# min delta and patience for early stopping
min_d=0.001
patience=5

# test 
python "$wd"/train_keras_crossv.py --data_folder "$training_data" --save_path "$save_path" --pt_weights "$weights" --model_name "$model_name" --batch_size 120 --epochs $epochs --learning_rate $lr --min_delta $min_d --patience $patience --num_folds $num_folds --image_dim "$image_dim"

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
mv "$WORK"/output/wheat_det/training/train_cross_val_"$SLURM_JOB_ID".log "$WORK"/output/wheat_det/training/"$model_name".log
