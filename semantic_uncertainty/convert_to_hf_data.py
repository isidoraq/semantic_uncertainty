import json
import os
import re
from datasets import Dataset, DatasetDict
from typing import List, Dict, Tuple


def extract_language_pair(file_name: str) -> Tuple[str, str]:
    match = re.search(r"\.([a-z]{2})-([a-z]{2})\.", file_name)
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"


def process_entry(entry: Dict, source_lang: str, target_lang: str) -> Dict:
    if (
        "id" not in entry
        or "model_input" not in entry
        or "model_output_text" not in entry
        or "hard_labels" not in entry
    ):
        return {
            "id": entry.get("id", "unknown"),
            "question": entry.get("model_input", entry.get("input", "")),
            "context": entry.get("model_output_text", entry.get("output", "")),
            "answers": {"text": [], "answer_start": []},
            "source_language": source_lang,
            "target_language": target_lang,
        }

    return {
        "id": entry["id"],
        "question": entry["model_input"],
        "context": entry["model_output_text"],
        "answers": {
            "text": [
                entry["model_output_text"][start:end]
                for start, end in entry["hard_labels"]
            ],
            "answer_start": [start for start, _ in entry["hard_labels"]],
        },
        "source_language": source_lang,
        "target_language": target_lang,
    }


def load_data(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def convert_to_squad_format(
    data: List[Dict], source_lang: str, target_lang: str
) -> Dict:
    squad_format = {"version": "v2.0", "data": []}

    for entry in data:
        processed = process_entry(entry, source_lang, target_lang)

        article = {
            "title": f"Article {processed['id']}",
            "paragraphs": [
                {
                    "context": processed["context"],
                    "qas": [
                        {
                            "id": processed["id"],
                            "question": processed["question"],
                            "answers": [
                                {"text": text, "answer_start": start}
                                for text, start in zip(
                                    processed["answers"]["text"],
                                    processed["answers"]["answer_start"],
                                )
                            ],
                            "is_impossible": len(processed["answers"]["text"]) == 0,
                        }
                    ],
                }
            ],
        }
        squad_format["data"].append(article)

    return squad_format


def convert_to_hf_dataset(
    squad_data: Dict, source_lang: str, target_lang: str
) -> Dataset:
    flattened_data = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                flattened_data.append(
                    {
                        "id": qa["id"],
                        "title": article["title"],
                        "context": context,
                        "question": qa["question"],
                        "answers": {
                            "text": [answer["text"] for answer in qa["answers"]],
                            "answer_start": [
                                answer["answer_start"] for answer in qa["answers"]
                            ],
                        },
                        "source_language": source_lang,
                        "target_language": target_lang,
                    }
                )
    return Dataset.from_list(flattened_data)


def process_file(file_path: str, output_dir: str):
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")

    source_lang, target_lang = extract_language_pair(file_name)
    print(f"Detected language pair: {source_lang}-{target_lang}")

    try:
        data = load_data(file_path)
        squad_format_data = convert_to_squad_format(data, source_lang, target_lang)
        hf_dataset = convert_to_hf_dataset(squad_format_data, source_lang, target_lang)

        # Save the dataset
        dataset_output_dir = os.path.join(output_dir, file_name.split(".")[0])
        hf_dataset.save_to_disk(dataset_output_dir)

        # Save the SQuAD format JSON file
        squad_output_path = os.path.join(output_dir, f"{file_name.split('.')[0]}.json")
        with open(squad_output_path, "w", encoding="utf-8") as f:
            json.dump(squad_format_data, f, ensure_ascii=False, indent=2)

        print(
            f"Processed {file_name}. Dataset saved to {dataset_output_dir}, SQuAD format saved to {squad_output_path}"
        )

        return hf_dataset
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return None


# Usage
input_folder = "/home/ubuntu/semeval-2025-task3/val_data_test"
output_folder = "/home/ubuntu/semantic_uncertainty/semantic_uncertainty/data"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process all .jsonl files in the input folder
datasets = {}
for file_name in os.listdir(input_folder):
    if file_name.endswith(".jsonl"):
        file_path = os.path.join(input_folder, file_name)
        dataset_name = file_name.split(".")[0]
        dataset = process_file(file_path, output_folder)
        if dataset is not None:
            datasets[dataset_name] = dataset

# Combine all successfully processed datasets into a DatasetDict
if datasets:
    combined_dataset = DatasetDict(datasets)

    # Save the combined dataset
    combined_output_dir = os.path.join(output_folder, "combined_dataset")
    combined_dataset.save_to_disk(combined_output_dir)

    print(f"All datasets processed. Combined dataset saved to {combined_output_dir}")

    # Print some information about the combined dataset
    print(combined_dataset)
    for dataset_name, dataset in combined_dataset.items():
        print(f"\nDataset: {dataset_name}")
        print(dataset)
        print(dataset[0])  # Print the first example of each dataset
else:
    print("No datasets were successfully processed.")
