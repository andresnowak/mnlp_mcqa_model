
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk, Dataset
import argparse
from huggingface_hub import login
import time
import re
import random
from tqdm import tqdm

from src.utils import load_config


def MMLU_train_auxiliar(data: Dataset, data_info: dict, stem_subsets: list[str]):
    mmlu_auxiliary_data = []
    int_to_char_ans = {0: "A", 1: "B", 2: "C", 3: "D"}
    for cnt, data_point in enumerate(data):
        if data_point["task"] not in stem_subsets:
            continue
        mmlu_auxiliary_data.append(
            {
                "dataset": data_info["name"],
                "id": f"{data_info['subset_name']}_{cnt}",
                "question": data_point["question"],
                "choices": data_point["choices"],
                "answer": int_to_char_ans[data_point["answer"]],
                "context": None,
            }
        )

    return mmlu_auxiliary_data

def MMLU(data: Dataset, data_info: dict, stem_subsets: list[str]):
    mmlu_validation_data = []

    int_to_char_ans = {0: "A", 1: "B", 2: "C", 3: "D"}

    mmlu_validation_data = []
    for cnt, data_point in enumerate(data):
        if data_point["subject"] not in stem_subsets:
            continue
        mmlu_validation_data.append(
            {
                "dataset": data_info["name"],
                "id": f"{data_info['subset_name']}_{cnt}",
                "question": data_point["question"],
                "choices": data_point["choices"],
                "answer": int_to_char_ans[data_point["answer"]],
                "context": None,
            }
        )

    return mmlu_validation_data


def MMLU_10_choices(data: Dataset, data_info: dict):
    mmlu_validation_data = []

    for cnt, data_point in enumerate(data):
        mmlu_validation_data.append(
            {
                "dataset": data_info["name"],
                "id": f"{data_info['subset_name']}_{cnt}",
                "question": data_point["question"],
                "choices": data_point["10_choices"],
                "answer": data_point["answer_10_choices_letter"],
                "context": None,
            }
        )

    return mmlu_validation_data


def ai2_arc(data: Dataset, data_info: dict):
    arc_easy_data_validation = []

    answer_map = {"1": "A", "2": "B", "3": "C", "4": "D", "A": "A", "B": "B", "C": "C", "D": "D", "E": "E"} # some choices have level 1 to 4 and some A to E

    for data_point in data:
        arc_easy_data_validation.append(
            {
                "dataset": data_info["name"],
                "id": f"{data_info['subset_name']}_{data_point['id']}",
                "question": data_point["question"],
                "choices": data_point["choices"]["text"],
                "answer": answer_map[data_point["answerKey"]],
                "context": None,
            }
        )

    return arc_easy_data_validation


def science_qa(data: Dataset, data_info: dict):
    int_to_char_ans = {0: "A", 1: "B", 2: "C", 3: "D"}

    scienceqa_data = []

    for cnt, data_point in enumerate(data):
        if data_point["image"] is not None or data_point["subject"] != "natural science" or data_point["task"] != "closed choice":
            continue
        scienceqa_data.append({
            "dataset": data_info["name"],
            "id": f"{data_info['subset_name']}_{cnt}",
            "question": data_point["question"],
            "choices": data_point["choices"],
            "answer": int_to_char_ans[data_point["answer"]],
            "context": None,
        })

    return scienceqa_data


def math_qa(data: Dataset, data_info: dict):
    mathqa_data = []

    char_to_char_ans = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}

    def extract_choices(choices_str):
        matches = re.findall(r"[a-e]\s*\)\s*([^,]+)", choices_str)
        # Clean up whitespace and dots
        res = [m.strip().replace(" .", ".").replace(" ,", ",") for m in matches]
        return res


    for cnt, data_point in enumerate(data):
        mathqa_data.append(
            {
                "dataset": data_info["name"],
                "id": f"{data_info['subset_name']}_{cnt}",
                "question": data_point["Problem"],
                "choices": extract_choices(data_point["options"]),
                "answer": char_to_char_ans[data_point["correct"]],
                "context": data_point["Rationale"],
            }
        )

    return mathqa_data


def openbook_qa(data: Dataset, data_info: dict):
    openbook_qa_data = []

    for cnt, data_point in enumerate(data):
        openbook_qa_data.append(
            {
                "dataset": data_info["name"],
                "id": f"{data_info['subset_name']}_{cnt}",
                "question": data_point["question_stem"],
                "choices": data_point["choices"]["text"],
                "answer": data_point["answerKey"],
                "context": data_point["fact1"],
            }
        )

    return openbook_qa_data


def sciq(data: Dataset, data_info: dict):
    sciq_data = []

    for cnt, data_point in enumerate(data):
        choices = [data_point[f"distractor{i + 1}"] for i in range(3)]
        correct = data_point["correct_answer"]
        choices.append(correct)

        # Shuffle choices and find new index of the correct answer
        random.shuffle(choices)
        correct_index = choices.index(correct)
        letter_map = ['A', 'B', 'C', 'D']
        correct_letter = letter_map[correct_index]

        sciq_data.append(
            {
                "dataset": data_info["name"],
                "id": f"{data_info['subset_name']}_{cnt}",
                "question": data_point["question"],
                "choices": choices,
                "answer": correct_letter,
                "context": data_point["support"]
            }
        )

    return sciq_data


def med_mcqa(data: Dataset, data_info: dict):
    med_mcqa_data = []

    for cnt, data_point in enumerate(data):
        if data_point["choice_type"] != "single" or data_point["cop"] == -1:
            continue

        choices = [data_point[f"op{i}"] for i in ["a", "b", "c", "d"]]

        answer_number = data_point["cop"]
        letter_map = ["A", "B", "C", "D"]
        answer = letter_map[answer_number]

        med_mcqa_data.append(
            {
                "dataset": data_info["name"],
                "id": f"{data_info['subset_name']}_{cnt}",
                "question": data_point["question"],
                "choices": choices,
                "answer": answer,
                "context": None, # the exp column is about the explanation of solution
            }
        )

    return med_mcqa_data

def join_datasets(config):

    datasets_to_combine = {}
    dataset_splits = []

    for dataset_info in tqdm(config["datasets"], desc="Processing datasets"):
        split = dataset_info["split"]
        dataset = load_dataset(
            dataset_info["name"],
            dataset_info["config"],
            split=split,
        )

        data = []

        if dataset_info["name"] == "kz919/mmlu-auxiliary-train-auto-labelled":
            data = MMLU_train_auxiliar(dataset, dataset_info, config["general"]["stem_subsets"])
        elif dataset_info["name"] == "cais/mmlu":
            data = MMLU(dataset, dataset_info, config["general"]["stem_subsets"])
        elif dataset_info["name"] == "andresnowak/mmlu-auxiliary-train-10-choices":
            data = MMLU_10_choices(dataset, dataset_info)
        elif dataset_info["name"] == "allenai/ai2_arc":
            data = ai2_arc(dataset, dataset_info)
        elif dataset_info["name"] == "derek-thomas/ScienceQA":
            data = science_qa(dataset, dataset_info)
        elif dataset_info["name"] == "allenai/math_qa":
            data = math_qa(dataset, dataset_info)
        elif dataset_info["name"] == "allenai/openbookqa":
            data = openbook_qa(dataset, dataset_info)
        elif dataset_info["name"] == "allenai/sciq":
            data = sciq(dataset, dataset_info)
        elif dataset_info["name"] == "openlifescienceai/medmcqa":
            data = med_mcqa(dataset, dataset_info)


        data = Dataset.from_list(data)

        if "size" in dataset_info:
            data = data.shuffle(seed=42).select(
                range(dataset_info["size"])
            )

        datasets_to_combine[f"{dataset_info['subset_name']}|{split}"] = data
        dataset_splits.append(split)


    return datasets_to_combine, dataset_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join datasets from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="config/MCQA_datasets_join_2.yaml",
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
                config["hub_dataset_name"],
                split=split,
                config_name=name,
            )
            time.sleep(10)

        for split_type in ["train", "validation", "test"]:
            datasets_for_split = [
                dataset for dataset, split in zip(combined_dataset.values(), dataset_splits)
                if split == split_type
            ]

            concatenate_datasets(datasets_for_split).push_to_hub(
                config["hub_dataset_name"], split=split_type, config_name="all"
            )

        print(f"Dataset uploaded to Hugging Face Hub: {config['hub_dataset_name']}")
    else:
        output_path = config.get("output_path", "combined_dataset")
        combined_dataset.save_to_disk(output_path)
        print(f"Dataset saved locally at: {output_path}")
