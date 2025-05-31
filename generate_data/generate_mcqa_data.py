# %%
import gpt_wrapper
from gpt_wrapper.chat import Chat
from datasets import load_dataset

from secrets_env import PARROT_API_BASE, PARROT_API_KEY

# %%
gpt_wrapper.api_base = PARROT_API_BASE
gpt_wrapper.api_key = PARROT_API_KEY

# %%
Chat.budget()

# %%
stem_subjects = [
      "abstract_algebra",
      "anatomy",
      "astronomy",
      "college_biology",
      "college_chemistry",
      "college_computer_science",
      "college_mathematics",
      "college_physics",
      "computer_security",
      "conceptual_physics",
      "electrical_engineering",
      "elementary_mathematics",
      "high_school_biology",
      "high_school_chemistry",
      "high_school_computer_science",
      "high_school_mathematics",
      "high_school_physics",
      "high_school_statistics",
      "machine_learning",
  ]

mmlu_dataset = load_dataset(
    "kz919/mmlu-auxiliary-train-auto-labelled"
)

def only_stem_subjects(example):
    return example["task"] in stem_subjects


# Filter the dataset to only include STEM subjects
stem_dataset = mmlu_dataset.filter(only_stem_subjects)
stem_dataset = stem_dataset["train"]

# %%
print(stem_dataset)

# %%
chat_general = Chat.create("Test chat")
model_args = {
    "temperature": 0.8,  # default is 0.7
    "max_tokens": 1000,  # default is 100
    "top_p": 0.8,  # default is 0.9
    "presence_penalty": 0.1,  # default is 0.1
    "frequency_penalty": 0.1,  # default is 0.1
}

# %%
prompt_template = """### Prompt: Extend MMLU Multiple-Choice Options

You will be given a multiple-choice question from the MMLU dataset, along with four answer choices labeled A through D. Your task is to generate **six additional plausible but incorrect answer choices** (E through J) that are consistent in style, topic, and level of difficulty with the original choices. 

**Do not change the original question or the first four answer choices (A–D)** or their order.

Return the **full list of ten answer choices (A–J)**, keeping the original A–D unchanged, and appending the six new distractors (E–J). Use the exact format below:

A. original_choice_A  
B. original_choice_B  
C. original_choice_C  
D. original_choice_D  
E. new_choice_E  
F. new_choice_F  
G. new_choice_G  
H. new_choice_H  
I. new_choice_I  
J. new_choice_J

**Guidelines for new choices (E–J):**
- Must be plausible but clearly incorrect.
- Should match the grammar and tone of the original options.
- Must not duplicate or closely paraphrase any existing option.
- Should require similar domain knowledge or reasoning as the original options.

**Do not output anything other than the final 10 choices in the specified format.**

---

### Question:
{question}

A. {choice_a}  
B. {choice_b}  
C. {choice_c}  
D. {choice_d}
"""


# %%
# Optional: Choose a subset for testing
subset = stem_dataset.select(range(10))

# Now loop through the dataset and format each prompt
for example in subset:
    question = example["question"]
    filled_prompt = prompt_template.format(
        question=question,
        choice_a=example["choices"][0],
        choice_b=example["choices"][1],
        choice_c=example["choices"][2],
        choice_d=example["choices"][3],
    )

    # Send `filled_prompt` to your model here
    # response = model.generate(filled_prompt) or openai.ChatCompletion.create(...)

    print("===== Prompt =====")
    print(filled_prompt)
    print("\n===== Model Output Placeholder =====\n")
    response = chat_general.ask(filled_prompt, model_args=model_args)
    print(response)


# %%
responses = []

# %%
# Optional: Choose a subset for testing

# Now loop through the dataset and format each prompt
for index, example in enumerate(stem_dataset):
    question = example["question"]
    filled_prompt = prompt_template.format(
        question=question,
        choice_a=example["choices"][0],
        choice_b=example["choices"][1],
        choice_c=example["choices"][2],
        choice_d=example["choices"][3],
    )

    chat_new = Chat.create(f"Test chat {index}")
    response = chat_new.ask(filled_prompt, model_args=model_args)
    responses.append(response)
    print(Chat.budget())

# %%
import json
from pathlib import Path
from tqdm import tqdm

output_path = Path("generated_mmlu_choices.jsonl")
responses = []

# Resume support: load existing responses (if resuming)
if output_path.exists():
    with open(output_path, "r") as f:
        completed = [json.loads(line) for line in f]
        completed_ids = set(item["index"] for item in completed)
else:
    completed = []
    completed_ids = set()

# Main loop
for index, example in tqdm(enumerate(stem_dataset), total=len(stem_dataset)):
    if index in completed_ids:
        continue  # Skip already processed

    question = example["question"]
    choices = example["choices"]

    filled_prompt = prompt_template.format(
        question=question,
        choice_a=choices[0],
        choice_b=choices[1],
        choice_c=choices[2],
        choice_d=choices[3],
    )

    try:
        chat = Chat.create(f"Test chat {index}")
        response = chat.ask(filled_prompt, model_args=model_args)
    except Exception as e:
        print(f"[Error at index {index}]: {e}")
        continue  # Skip or retry logic can be added here

    # Store result
    record = {
        "index": index,
        "question": question,
        "choices": choices,
        "filled_prompt": filled_prompt,
        "response": str(response),
        "task": example["task"],
    }
    responses.append(record)

    # Save to JSONL incrementally
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(Chat.budget())  # Optional: print budget status


# %%



