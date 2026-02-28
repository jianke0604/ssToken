export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=2



base_model="meta-llama/Llama-3.2-3B"
data_prop=0.6


## path 
result_path="eval_results"
model_path=$1


    ###########################################
    ############## tydiqa eval ################
    ###########################################
CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
    --data_dir eval_data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 200 \
    --max_context_length 512 \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $model_path \
    --eval_batch_size 5
