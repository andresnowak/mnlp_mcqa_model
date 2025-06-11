import pandas as pd
import re
import difflib
from difflib import SequenceMatcher
from datasets import Dataset
import numpy as np
import argparse

# ===== NEW: Add CLI argument setup =====
parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", type=str, default="andresnowak/mmlu-auxiliary-train-10-choices", help="HF repo ID")
args = parser.parse_args()

df = pd.read_json("generated_mmlu_choices.jsonl", lines=True)
print(df.head())
print(len(df))

def normalize(s):
    return s.strip().replace("’", "'").replace("“", '"').replace("”", '"') # As GPT because of its training can like to use unicode a lot

def extract_choices(text):
    return [normalize(match[1]) for match in re.findall(r"([A-J])\.\s*(.+)", text)]

df["10_choices"] = df["response"].apply(extract_choices)

print(df["10_choices"])

assert all(df["10_choices"].apply(lambda x: len(x) == 10)), (
    "Some rows don't have exactly 10 choices."
)

# Extract original choices A–D from the original choices list in the dataset
df["original_choices"] = df["choices"].apply(
    lambda x: [choice.strip() for choice in x[:4]]
)

# Extract the first 4 from generated choices
df["generated_A_to_D"] = df["10_choices"].apply(lambda x: x[:4])


def similarity_percent_seqmatch_list(list1, list2, threshold=80):
    # Returns True if all choices have similarity > threshold
    for a, b in zip(list1, list2):
        sim = SequenceMatcher(None, a.strip(), b.strip()).ratio() * 100
        if sim <= threshold:
            return False
    return True


# Apply per row to get boolean column:
df["A_to_D_match"] = df.apply(
    lambda row: similarity_percent_seqmatch_list(
        row["original_choices"], row["generated_A_to_D"]
    ),
    axis=1,
)

# Optional: show mismatches
mismatches = df[~df["A_to_D_match"]][
    ["original_choices", "generated_A_to_D"]
]
print("Mismatches")
print(mismatches.head())

print(mismatches.iloc[0]["original_choices"])
print()
print(mismatches.iloc[0]["generated_A_to_D"])

print(mismatches.iloc[0]["original_choices"] == mismatches.iloc[0]["generated_A_to_D"])

# Summary: how many mismatched?
num_mismatched = (~df["A_to_D_match"]).sum()
print(f"Mismatches: {num_mismatched} out of {len(df)}")


def show_differences(list1, list2):
    for i, (a, b) in enumerate(zip(list1, list2)):
        if a != b:
            print(f"\n🔍 Difference at index {i}:")
            print("ORIGINAL: ", repr(a))
            print("GENERATED:", repr(b))
            print("CHAR DIFF:")
            for line in difflib.ndiff([a], [b]):
                print(line)

show_differences(mismatches.iloc[0]["original_choices"], mismatches.iloc[0]["generated_A_to_D"])

def similarity_percent_seqmatch(s1, s2):
    for i, (a, b) in enumerate(zip(s1, s2)):
        print(SequenceMatcher(None, a.strip(), b.strip()).ratio() * 100)


similarity_percent_seqmatch(
    mismatches.iloc[0]["original_choices"], mismatches.iloc[0]["generated_A_to_D"]
)

# ----- Create huggingface dataset -----

# Step 1: Filter matching rows without resetting index
df_matched = df[df["A_to_D_match"]].copy()

# Step 2: Use original index to create the dataset ID
df_matched["dataset"] = df_matched.index.map(
    lambda i: f"kz919-mmlu-auxiliary-train-auto-labelled_{i}"
)


# Step 3: Create permuted choices and track correct answer
def permute_choices(row):
    choices = row["10_choices"]
    correct_answer = row["correct_answer"]

    # Get current index of correct answer
    original_index = int(correct_answer) # the answers here are numbers (index) not letters

    # Create permutation
    rng = np.random.default_rng()
    perm = rng.permutation(len(choices))
    permuted_choices = [choices[i] for i in perm]

    # Find new position of correct answer
    new_index = np.where(perm == original_index)[0][0]

    return pd.Series(
        {"10_choices_permuted": permuted_choices, "answer_10_choices": new_index}
    )


# Apply permutation and create new columns
df_matched[["10_choices", "answer_10_choices"]] = df_matched.apply(
    permute_choices, axis=1
)

# Step 4: Select final columns
df_matched = df_matched[
    [
        "dataset",
        "task",
        "question",
        "choices",
        "correct_answer",
        "10_choices",
        "answer_10_choices",
    ]
]

print(df_matched["answer_10_choices"].unique())

df_matched["answer_10_choices_letter"] = df_matched["answer_10_choices"].apply(
    lambda x: chr(65 + x)  # 0->A, 1->B, ..., 9->J
)
# Rename columns
df_matched.rename(columns={"correct_answer": "answer"}, inplace=True)

# Step 5: Convert to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df_matched, preserve_index=False)

hf_dataset.push_to_hub(args.repo_id, config_name="default", split="train")