from torch.utils.data import Dataset

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

class MCQADatasetClassification(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
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
