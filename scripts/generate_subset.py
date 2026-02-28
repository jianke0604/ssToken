import json
from datasets import load_dataset
import os

def main():
    data_path = './datasets'
    subset_size = 5
    train_dataset_name = "ds2-50k-test"
    local_json_path = "./datasets/ds2-50k-full.json"
    
    try:
        if os.path.exists(local_json_path):
            dataset = load_dataset('json', data_files=local_json_path)['train']
            print(f"Loaded dataset from local JSON: {local_json_path}")
        else:
            raise FileNotFoundError(f"Local JSON file not found: {local_json_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    data_size = len(dataset) // subset_size
    print(f"Total dataset size: {len(dataset)}")
    print(f"Creating {subset_size} subsets with {data_size} samples each")
    
    for i in range(subset_size):
        selected_indices = [idx for idx in range(data_size * i, data_size * (i + 1))]
        subset = dataset.select(selected_indices)
        output_file = os.path.join(data_path, f"{train_dataset_name}_{i}.json")
        subset.to_json(output_file)
        print(f"Saved subset {i} to {output_file}")

if __name__ == "__main__":
    main()