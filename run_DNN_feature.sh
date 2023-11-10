#!/bin/bash

# Define two arrays
type_filter=("none" "BPF_20_200" "none" "BPF_20_200" "none" "BPF_20_200" "none" "BPF_20_200" "none" "BPF_20_200")
type_norm=("none" "none" "mvc" "mvc" "min_max" "min_max" "standardization" "standardization" "mu_law_1" "mu_law_1")

for (( i=0; i<${#type_filter[@]}; i++ )); do
    
    # echo ${type_filter[$i]} ${type_norm[$i]}

    log_name=$(date +%Y%m%d_%H%M)_main_develop_DB2
    mkdir -p Results/$log_name

    CUDA_VISIBLE_DEVICES=3 \
    python3 main_develop_DB2.py \
    --subject_list 1,2,3 \
    --exercise_list 1,2,3 \
    --window_size_sec 0.2  \
    --window_step_sec 0.1 \
    --num_epoch 3000 \
    --batch_size 512 \
    --model_type DNN_feature \
    --lr 0.001 \
    --en_train \
    --feat_extract \
    --type_filter ${type_filter[$i]} \
    --type_norm  ${type_norm[$i]} \
    --log_name $log_name \
    | tee Results/$log_name/record.log

    # --en_train \
    # --feat_extract \
    # --subject_list 1 \
    # --load_dataset \


done

