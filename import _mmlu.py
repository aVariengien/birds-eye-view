# %%
from datasets import load_dataset
import json
from tqdm import tqdm

def export_mmlu_to_json(output_file: str, splits = ['test']):
    """Possible splits are ['train', 'test', 'validation', 'dev', 'auxiliary_train']"""
    # Load the dataset
    ds = load_dataset("cais/mmlu", "all")

    # Prepare the data for JSON export
    json_data = []

    # Process each split
    for split in splits:
        if split not in ds:
            print(f"Split '{split}' not found in the dataset. Skipping.")
            continue

        split_data = ds[split]

        for i, item in enumerate(tqdm(split_data, desc=f"Processing {split} split")):
            chunk = {
                "text": f"Question: {item['question']}<br>"
                        f"A: {item['choices'][0]}<br>"
                        f"B: {item['choices'][1]}<br>"
                        f"C: {item['choices'][2]}<br>"
                        f"D: {item['choices'][3]}<br>"
                        f"Answer: {item['choices'][item['answer']]}\n",
                "attribs": {
                    "subject": item['subject'],
                    "split": split,
                    "index": i,
                    "doc_position": i / len(split_data),
                    "correct_answer": item['choices'][item['answer']],
                    "answer_index": item['answer']
                }
            }
            json_data.append(chunk)

    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"MMLU dataset exported to {output_file}")
    print(f"Total number of questions exported: {len(json_data)}")

# Usage
export_mmlu_to_json("data/mmlu_dataset_all_splits.json")
# %%
