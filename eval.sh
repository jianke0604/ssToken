export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1         
export HF_DATASETS_CACHE="path_to_your_hf_cache"  

HF_MODEL_PATH=$1
TASK_NAME=$2

declare -A task_2_numfewshot=(
    ["truthfulqa"]=0
    ["agieval"]=0
    ["mmlu"]=5
    ["triviaqa"]=0
    ["winogrande"]=0
    ["hellaswag"]=0
    ["arc_easy"]=0
    ["arc_challenge"]=0
    ["logiqa"]=0
)

if [[ "$TASK_NAME" == "all" ]]; then
    ALL_TASKS=("winogrande" "arc_challenge" "arc_easy" "hellaswag" "logiqa" "mmlu" "truthfulqa" "triviaqa" "agieval")
    
    EVAL_DIR="$HF_MODEL_PATH/evaluate"
    mkdir -p "$EVAL_DIR"
    echo "Created evaluation directory: $EVAL_DIR"
    
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
    echo "Available GPUs: $GPU_COUNT"
    
    for i in "${!ALL_TASKS[@]}"; do
        task="${ALL_TASKS[$i]}"
        gpu_id=$((i % GPU_COUNT))
        num_fewshot=${task_2_numfewshot[$task]}
        
        echo "Starting evaluation for task: $task on GPU: $gpu_id"
        
        (
            lm_eval --model hf \
              --model_args pretrained=$HF_MODEL_PATH,parallelize=False \
              --tasks $task \
              --device cuda:$gpu_id \
              --batch_size auto \
              --num_fewshot $num_fewshot \
              > "$EVAL_DIR/${task}.log" 2>&1
            
            echo "Completed evaluation for task: $task"
        ) &
        
        if (( (i + 1) % GPU_COUNT == 0 )); then
            wait
        fi
    done
    
    wait
    echo "All evaluations completed. Results saved in: $EVAL_DIR"
    
else
    if [[ -z "${task_2_numfewshot[$TASK_NAME]}" ]]; then
        echo "Warning: Task '$TASK_NAME' not found in task_2_numfewshot. Using default num_fewshot=0"
        NUM_FEWSHOT=0
    else
        NUM_FEWSHOT=${task_2_numfewshot[$TASK_NAME]}
    fi

    lm_eval --model hf \
      --model_args pretrained=$HF_MODEL_PATH,parallelize=False \
      --tasks $TASK_NAME \
      --device cuda:0 \
      --batch_size auto \
      --num_fewshot $NUM_FEWSHOT \

fi