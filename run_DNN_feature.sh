#!/bin/bash
model_name=DNN_feature

# Define two arrays
type_filter=("none")
type_norm=("standardization")
# type_filter=("none" "BPF_20_200" "none" "BPF_20_200" )
# type_norm=("mu_law_16" "mu_law_16" "mu_law_64" "mu_law_64" )

for (( i=0; i<${#type_filter[@]}; i++ )); do
    
    # echo ${type_filter[$i]} ${type_norm[$i]}

    current_time=$(date +%Y%m%d_%H%M)
    log_name="${model_name}_${current_time}"
    mkdir -p Results/$log_name

    CUDA_VISIBLE_DEVICES=0 \
    python3 main_intra_subject.py \
    --exercise_list 1,2,3 \
    --window_size_sec 0.2  \
    --window_step_sec 0.1 \
    --num_epoch 1000 \
    --batch_size 1024 \
    --model_type $model_name \
    --pretrain_model_PATH Results/DNN_feature_20231113_2348/DNN_feature.pth \
    --lr 0.001 \
    --dropout 0.5 \
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

