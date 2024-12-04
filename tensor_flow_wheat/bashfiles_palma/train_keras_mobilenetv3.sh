#!/bin/bash

#SBATCH --nodes=1

#SBATCH --tasks-per-node=8

#SBATCH --partition=express

#SBATCH --mem=16GB

#SBATCH --time=0-01:00:00

#SBATCH --job-name=test_train_script

#SBATCH --output=/scratch/tmp/b_kari02/output/wheat_det/training/train_keras_%j.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=b.karic@uni-muenster.de

#load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1
module load matplotlib/3.7.2

# place of code in palma
wd=/scratch/tmp/b_kari02/wheat_disease/tensor_flow_wheat/
time=`date +%Y%m%d-%H%M%S`
outpath=/scratch/tmp/b_kari02/output/wheat_det/training_${time} 
mkdir ${outpath}
mv /scratch/tmp/b_kari02/output/wheat_det/training/train_keras_$SLURM_JOB_ID.log /scratch/tmp/b_kari02/output/wheat_det/training_${time}/output.log

# test 
python "$wd"/train_keras_mobilenetv3.py --model_name "MobileNetV3_WheatDiseaseModel" --time "$time" --out "$outpath"

