# timestamp=$(date +%Y%m%d_%H%M)
model_name=ViT_TNet
current_time=$(date +%Y%m%d_%H%M)
log_name="${model_name}_${current_time}"
mkdir -p Results/$log_name

CUDA_VISIBLE_DEVICES=1 \
python3 main_develop_DB2.py \
--subject_list 1 \
--exercise_list 1,2,3 \
--window_size_sec 0.2  \
--window_step_sec 0.1 \
--num_epoch 3000 \
--batch_size 1024 \
--model_type $model_name \
--lr 0.001 \
--en_train \
--log_name $log_name \
| tee Results/$log_name/record.log

# --en_train \
# --feat_extract \
# --subject_list 1 \
# --load_dataset \