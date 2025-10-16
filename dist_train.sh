config_dir=configs/accident_anticipation/top
config_name=top_cap

current_datetime=$(date +"%Y%m%d_%H%M%S")
mkdir -p codes/$config_name/$current_datetime

find accident_anticipation -name "*.py" -exec cp --parents {} codes/$config_name/$current_datetime/ \;

find $config_dir -name $config_name.py -exec cp --parents {} codes/$config_name/$current_datetime/ \;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 tools/dist_train.sh $config_dir/$config_name.py 8
