# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


cluster_root_path=YOUR_ROOT_PATH
root_data_path="./datasets"

base_model="meta-llama/Llama-3.2-3B" #"meta-llama/Llama-3.1-8B" "Qwen/Qwen-2.5-7B" "Qwen/Qwen-2.5-14B" 
token_select_pattern="default" #'random' 'tokencleaning'
BATCH_SIZE_PER_GPU=3
random_seed=42
ratio=$1
data_prop=$2

model_path="./model/Llama-3.2-3B"

train_data_tag="ds2-50k-full"
train_data="${root_data_path}/${train_data_tag}.json"
run_name=$3

# note: need to use bash_src/calculate_loss.sh to calculate the loss of base model first

echo "start finetuning..."
bash_src/finetune.sh "$model_path" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop" "$token_select_pattern" "$random_seed" "$ratio" "$run_name"


