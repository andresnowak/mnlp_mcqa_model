
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk, Dataset
import argparse
from huggingface_hub import login
import time
import re
import random
from tqdm import tqdm

from src.utils import load_config

def join_datasets(config):

    datasets_to_combine = {}
    dataset_splits = []

    for dataset_info in tqdm(config["datasets"], desc="Processing datasets"):
        for split in dataset_info["config"]:
            dataset = load_dataset(
                dataset_info["name"],
                dataset_info["subset_name"],
                split=split,
            )

            def add_none_column(example, name):
                example[name] = None
                return example
            
            def add_messages(example, name):
                example[name] = [{}]

            if dataset_info["name"] == "andresnowak/MNLP_MCQA_dataset":
                dataset = dataset.map(lambda x: add_messages(x, "messages"))
            elif dataset_info["name"] == "andresnowak/Instruction-finetuning-mixture-mnlp":
                dataset = dataset.rename_column("source", "dataset")
                dataset = dataset.map(lambda x: add_none_column(x, "question"))
                dataset = dataset.map(lambda x: add_none_column(x, "choices"))
                dataset = dataset.map(lambda x: add_none_column(x, "answer"))
                dataset = dataset.map(lambda x: add_none_column(x, "context"))

            datasets_to_combine[f"{dataset_info['subset_name']}|{split}"] = dataset
            dataset_splits.append(split)


    return datasets_to_combine, dataset_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join datasets from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="config/MCQA_join_with_IF_datasets.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--hub-dataset-name",  # Fixed: hyphen instead of underscore for CLI convention
        type=str,
        default=None,  # Explicit None instead of empty default
        help="Override the dataset name for Hugging Face Hub",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    combined_dataset, dataset_splits = join_datasets(config)

    # Override config with CLI argument if provided
    if args.hub_dataset_name is not None:
        config["hub_dataset_name"] = args.hub_dataset_name

    # Save or push to Hub
    if config.get("push_to_hub", False):
        for dataset_tuple, split in zip(combined_dataset.items(), dataset_splits):
            name_combined, dataset = dataset_tuple
            name, split = name_combined.split("|")
            print(f"Pushing {name} dataset, split: {split}")
            dataset.push_to_hub(
                config["hub_dataset_name"], split=split, config_name=name
            )
            time.sleep(10)

        print(f"Dataset uploaded to Hugging Face Hub: {config['hub_dataset_name']}")
    else:
        output_path = config.get("output_path", "combined_dataset")
        combined_dataset.save_to_disk(output_path)
        print(f"Dataset saved locally at: {output_path}")
