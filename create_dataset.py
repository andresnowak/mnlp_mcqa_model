from datasets import load_dataset, concatenate_datasets, DatasetDict
import argparse
from huggingface_hub import login

from src.utils import load_config
from src.env_secrets import HF_TOKEN

def join_datasets(config):
    datasets = config["datasets"]

    datasets_to_combine = {}
    for dataset_info in datasets:
        print(f"Loading dataset: {dataset_info['name']}")

        dataset = load_dataset(
            dataset_info["name"],
            dataset_info.get(
                "subset", None
            ),  # Optional subset (e.g., "mnli" for "glue")
            split=dataset_info.get(
                "split", "train"
            ),  # Default to "train" if not specified
        )

        # Rename columns if specified in config
        if "column_mapping" in dataset_info:
            for old_col, new_col in dataset_info["column_mapping"].items():
                dataset = dataset.rename_column(old_col, new_col)

        # Select only specified columns (if given)
        if "columns_to_keep" in dataset_info:
            dataset = dataset.select_columns(dataset_info["columns_to_keep"])

        datasets_to_combine[dataset_info["subset_name"]] = dataset

    # Combine all datasets
    # combined_dataset = concatenate_datasets(DatasetDict(datasets_to_combine))

    # Shuffle if specified
    # if config.get("shuffle", False):
    #     combined_dataset = combined_dataset.shuffle(seed=config.get("seed", 42))

    return DatasetDict(datasets_to_combine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join datasets from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="config/IF_datasets_join.yml",
        help="Path to the configuration YAML file.",
    )

    login(token=HF_TOKEN)

    args = parser.parse_args()
    config = load_config(args.config)

    combined_dataset = join_datasets(config)

    # Save or push to Hub
    if config.get("push_to_hub", False):
        combined_dataset.push_to_hub(config["hub_dataset_name"])
        print(f"Dataset uploaded to Hugging Face Hub: {config['hub_dataset_name']}")
    else:
        output_path = config.get("output_path", "combined_dataset")
        combined_dataset.save_to_disk(output_path)
        print(f"Dataset saved locally at: {output_path}")