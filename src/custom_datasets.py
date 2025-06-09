from torch.utils.data import Dataset
import random

LETTER_INDICES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

random.seed(42)

class MCQADatasetClassification(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        answer_letter = ex["answer"].strip().upper()
        correct_index = ord(answer_letter) - ord("A")

        prompt = (
            "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"
            f"{ex['question']}\n"
            + "".join(
                [
                    f"{key}. {choice}\n"
                    for key, choice in zip(LETTER_INDICES, ex["choices"])
                ]
            )
            + "Answer:"
        )

        return {
            "prompt": prompt,
            "options": [
                f" {letter}" for letter in LETTER_INDICES[: len(ex["choices"])]
            ],
            "correct_idx": correct_index,
            "dataset": ex["dataset"],
        }


class MCQADatasetClassification_2(Dataset):
    def __init__(self, data, template_files, jinja_env):
        self.data = data
        self.jinja_env = jinja_env
        self.template_files = template_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        answer_letter = ex["answer"].strip().upper()
        correct_index = ord(answer_letter) - ord("A")


        chosen_template = random.choice(self.template_files)
        template = self.jinja_env.get_template(chosen_template)

        choices_list = [
            f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, ex["choices"])
        ]

        prompt = template.render(question=ex['question'], choices_list=choices_list)

        return {
            "prompt": prompt,
            "options": [
                f" {letter}" for letter in LETTER_INDICES[: len(ex["choices"])]
            ],
            "correct_idx": correct_index,
            "dataset": ex["dataset"],
        }
