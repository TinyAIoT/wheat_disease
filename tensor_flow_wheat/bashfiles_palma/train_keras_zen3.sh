#!/bin/bash -l
#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=gpua100

#SBATCH --gres=gpu:4
# max mem/node : 240 GB
#SBATCH --mem=128GB

#SBATCH --time=0-02:00:00

#SBATCH --job-name=training_keras_seq__zen3

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/output/wheat_det/training/train_keras_%j.log


# ZEN3 module load
#load modules 
module purge
module load palma/2022a  GCC/11.3.0  OpenMPI/4.1.4
module load TensorFlow/2.11.0
module load scikit-learn/1.1.2

# place of code in palma
WORK=/scratch/tmp/b_kari02/
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/

#training_data="$WORK"/tiny_ai/data/ds4_with_combined
# training data path
training_data="$WORK"/tiny_ai/data/long_2023_999
# save path for keras models and checkpoints
save_path="$WORK"/tiny_ai/results/
time=`date +%d.%m.%Y-%H:%M:%S`
# epochs and learning rate
epochs=10
batch_size=32
lr=0.0005
# if true model will also be tested after training
test_model=True
# image_dim for MobileNetv2 : [96, 128, 160, 182, 224]
image_dim=224
# name given to model
model_name=mobn_v2_"$epochs"_lr_"$lr"_bs_"$batch_size"_t_"$time"_dim_"$image_dim"
# min delta and patience for early stopping
min_d=0.0
patience=10
# test 
python "$wd"/train_keras_sequential.py --data_folder "$training_data" --save_path "$save_path" --pt_weights "$weights" --model_name "$model_name" --image_dim "$image_dim" --batch_size $batch_size --epochs $epochs --learning_rate $lr --min_delta $min_d --patience $patience --test_model $test_model

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
mv "$WORK"/output/wheat_det/training/train_keras_"$SLURM_JOB_ID".log "$WORK"/output/wheat_det/training/"$model_name".log
