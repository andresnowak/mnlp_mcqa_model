from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
import argparse
from huggingface_hub import login
import time

from src.utils import load_config
from src.env_secrets import HF_TOKEN


# This one is specfic to the allenai sft mixture
def join_datasets(config):
    dataset_info = config["datasets"][0]
    dataset_mnlp_info = config["datasets"][1]

    dataset = load_dataset(
        dataset_info["name"],
        split=dataset_info.get("split", "train"),
    )
    dataset_mnlp = load_from_disk(dataset_mnlp_info["name"])
    sources_to_exclude = dataset_info["sources_to_exclude"] # because list is small it is faster than set
    filtered_dataset = dataset.filter(lambda x: x["source"] not in sources_to_exclude)
    print("Size after excluding sources: ", len(filtered_dataset))
    filtered_dataset = filtered_dataset.filter(lambda x: len(x["messages"]) == 2) # we only want instruction - answer
    print("Final size: ", len(filtered_dataset))

    sources = dataset.unique("source")

    datasets_to_combine = {}

    for src_col, target_col in dataset_info["column_mapping"].items():
        # Create dataset for this column mapping
        dataset_to_combine = filtered_dataset.filter(
            lambda x: x["source"] == src_col,
        )

        if len(dataset_to_combine) != 0:
            datasets_to_combine[target_col] = dataset_to_combine

    datasets_to_combine[dataset_mnlp_info["subset_name"]] = dataset_mnlp

    return datasets_to_combine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join datasets from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="config/IF_datasets_join_2.yaml",
        help="Path to the configuration YAML file.",
    )

    login(token=HF_TOKEN)

    args = parser.parse_args()
    config = load_config(args.config)

    combined_dataset = join_datasets(config)

    # Save or push to Hub
    if config.get("push_to_hub", False):
        for name, dataset in combined_dataset.items():
            dataset.push_to_hub(
                config["hub_dataset_name"], split="train", config_name=name
            )
            time.sleep(10)
        concatenate_datasets(list(combined_dataset.values())).push_to_hub(
            config["hub_dataset_name"], split="train", config_name="all"
        )
        print(f"Dataset uploaded to Hugging Face Hub: {config['hub_dataset_name']}")
    else:
        output_path = config.get("output_path", "combined_dataset")
        combined_dataset.save_to_disk(output_path)
        print(f"Dataset saved locally at: {output_path}")