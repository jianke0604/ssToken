
model_name_or_path=$1
train_data=$2
BATCH_SIZE_PER_GPU=$3
NUM_GPUS=$4
model_path=$5
data_prop=$6
token_select_pattern=$7
random_seed=${8:-42}
ratio=${9:-0.5}
run_name=${10}
train_data_tag=$(basename "$train_data" .json)


echo "*** Training ${model_name_or_path} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, 2 gradient accumulation steps ***"
echo "*** Training data path: ${train_data} ***"
echo "*** Selected data proportion: ${data_prop} ***"
echo "*** Random deed: ${random_seed} ***"


accelerate launch \
    --num_machines 1 \
    --mixed_precision bf16 \
    --num_processes $NUM_GPUS \
    --config_file fsdp_configs/fsdp_config.yaml \
    --main_process_port 29501 \
    scripts/finetune.py \
    --model_name_or_path $model_name_or_path \
    --gradient_checkpointing \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name $model_name_or_path \
    --train_file $train_data \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps 2\
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ${model_path}/lora_${run_name}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --train_data_tag $run_name \
    --token_select_pattern $token_select_pattern \
    --data_prop $data_prop \
    --ratio $ratio \
    --with_prompt_token False \
    --seed $random_seed \
    --reduce_loss selection

python scripts/merge_lora.py \
    --base_model_name_or_path $model_name_or_path \
    --lora_model_name_or_path ${model_path}/lora_${run_name}/ \
    --output_dir ${model_path}/lora_merged_${run_name}/ \
    --save_tokenizer \
    --use_fast_tokenizer 

rm -rf ${model_path}/lora_${run_name}