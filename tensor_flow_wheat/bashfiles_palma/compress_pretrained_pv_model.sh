#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=10

#SBATCH --partition=gpu2080
# try: normal ,express, long

#SBATCH --mem=200GB

#SBATCH --time=0-1:00:00

#SBATCH --job-name=model_compression

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/output/wheat_det/training/compress_%j.log

#load modules 
module purge

#skylake config
# module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
# # module load TensorFlow/2.13.0
# module load scikit-learn/1.3.1
# module load matplotlib/3.7.2

#zen3 config
module load palma/2022a  GCC/11.3.0  OpenMPI/4.1.4
module load scikit-learn/1.1.2

pip install --user --upgrade tensorflow-model-optimization
pip install --user tensorflow_datasets
pip install --user tensorflow==2.18.0
pip install --user tf_keras==2.18.0 
# place of code in palma
wd="$HOME"/tiny_ai_home/wheat_disease/tensor_flow_wheat/
 # training data path
#training_data="$WORK"/tiny_ai/data/ds4_with_combined
training_data="$WORK"/tiny_ai/data/plant_village/color
input_model="$HOME"/tiny_ai_home/wheat_disease/models/plant_disease_detection.keras
# save path for 
save_path="$WORK"/tiny_ai/results/
time=`date +%d.%m.%Y_%H-%M-%S`
# image_dim for MobileNet: [96, 128, 160, 182, 224]
image_dim=224
# decrease image dim or batch_size when OOM (Out of Memory)
batch_size=32
# using zen3 partition ?
zen3=true
test_ds_size=0.01

# run code with flags
# sbatch --mail-user "kwundram@uni-muenster.de" $HOME/tiny_ai_home/wheat_disease/tensor_flow_wheat/bashfiles_palma/train_cross_val.sh
# --include_top $include_top
python "$wd"/compress_pt_pv.py --time "$time" --save_path "$save_path" --training_data "$training_data" --input_model "$input_model" --batch_size "$batch_size" --image_dim "$image_dim" --zen3 $zen3 --test_ds_size "$test_ds_size"

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
# mv "$WORK"/output/wheat_det/training/compress_"$SLURM_JOB_ID".log "$WORK"/output/wheat_det/training/compressed_"$model_name".log
