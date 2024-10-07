#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=normal

#SBATCH --mem=16GB

#SBATCH --time=0-01:00:00

#SBATCH --job-name=convert_to_tflite

#SBATCH --output=/scratch/tmp/kwundram/output/wheat_det/converting/convert_to_tflite

#SBATCH --mail-type=ALL

#SBATCH --mail-user=kwundram@uni-muenster.de

#load modules with available GPU support (this is an example, modify to your needs!)
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1

# place of code in palma
wd=/scratch/tmp/kwundram/tiny_ai/wheat_repo/wheat_disease/tensor_flow_wheat/
# path where keras model is saved. ends with .keras
keras_savepath=/scratch/tmp/kwundram/tiny_ai/wheat_repo/wheat_disease/tensor_flow_wheat/keras_models/mobile_net_v2_80Epochs/model.keras
# name given to tf lite model
tfl_model_name=tfl_80Epochs

# test 
python "$wd"/convert_to_tflite.py --keras_savepath "$keras_savepath" --tf_lite_modelname "$tfl_model_name"
