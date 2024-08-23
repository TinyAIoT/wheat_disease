# Wheat Diseas Detection

## data 
Y_split already in /data. 
X_split can be downloaded from sciebo in Use cases/Landwirtschaft/edge_impulse_train_data


## create conda env ( cd to edgeimpulse_copy)


conda env create -f ./wheat_dis_env.yaml -n ENV_NAME



## train.py
python ./train.py --data-directory "./data" --out-directory "./output"
