# Wheat Diseas Detection



## create conda env
cd ./
conda env create -f ./wheat_dis_env.yaml -n ENV_NAME



## train.py
python ./train.py --data-directory "./data" --out-directory "./output"
