"""Data Loading Utilities."""

import os
import json
import hashlib
import datasets
from uncertainty.utils import utils

args = utils.get_parser().parse_args()
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets


def load_ds(dataset_name, seed, add_options=None):
    """Load dataset."""
    user = os.environ["USER"]

    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == "svamp":
        dataset = datasets.load_dataset("ChilleD/SVAMP")
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            "question": x["Question"],
            "context": x["Body"],
            "type": x["Type"],
            "equation": x["Equation"],
            "id": x["ID"],
            "answers": {"text": [str(x["Answer"])]},
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "nq":
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16))

        reformat = lambda x: {
            "question": x["question"] + "?",
            "answers": {"text": x["answer"]},
            "context": "",
            "id": md5hash(str(x["question"])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset("TimoImhof/TriviaQA-in-SQuAD-format")[
            "unmodified"
        ]
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        scratch_dir = os.getenv("SCRATCH_DIR", ".")
        path = f"{scratch_dir}/{user}/semantic_uncertainty/data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {"question": [], "answers": [], "id": []}

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question["exact_answer"], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question["exact_answer"]
                    ]
                else:
                    exact_answers = [question["exact_answer"]]

                dataset_dict["answers"].append(
                    {
                        "text": exact_answers,
                        "answer_start": [0] * len(question["exact_answer"]),
                    }
                )
            else:
                dataset_dict["answers"].append(
                    {"text": question["ideal_answer"], "answer_start": [0]}
                )
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)

        # Split into training and validation set.
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    elif "mushroom" in dataset_name:
        # Load the custom dataset we created, treating all data as validation
        custom_dataset_path = f"/home/ubuntu/semantic_uncertainty/semantic_uncertainty/data/combined_dataset/{args.dataset}"

        try:
            # Load the dataset
            combined_dataset = load_from_disk(custom_dataset_path)

            # Check if it's already a Dataset object
            if isinstance(combined_dataset, Dataset):
                validation_dataset = combined_dataset
            else:
                # If it's a DatasetDict, concatenate all splits
                validation_dataset = datasets.concatenate_datasets(
                    list(combined_dataset.values())
                )

            # Treat all data as validation data
            train_dataset = None

            # Ensure the dataset has the required fields
            required_fields = ["id", "question", "context", "answers"]
            for field in required_fields:
                if field not in validation_dataset.features:
                    raise ValueError(
                        f"Custom dataset is missing required field: {field}"
                    )

            print(
                f"Loaded custom dataset with {len(validation_dataset)} validation examples"
            )

            # Print some information about the dataset
            print("Dataset features:", validation_dataset.features)
            print("First example:", validation_dataset[0])

        except Exception as e:
            print(f"Error loading custom dataset: {str(e)}")
            raise

    else:
        raise ValueError

    return train_dataset, validation_dataset
