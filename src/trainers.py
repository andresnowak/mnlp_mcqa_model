from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import Trainer, PreTrainedModel, PreTrainedTokenizerBase
import torch
import torch.nn.functional as F

class MCQATrainer(Trainer):
    def compute_loss(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        For each prompt we run the model once, grab the final (next‐token)
        logits, index into the letter‐token IDs, and compute a CE loss.
        """
        self.tokenizer = tokenizer
        device = model.device
        prompts = inputs["prompt"]  # List[str]
        correct_idxs = inputs["correct_idx"]  # List[int]
        all_options = inputs["options"]  # List[List[str]]

        batch_logits = []
        losses = []

        # Pre‐tokenize all option‐letters to single token IDs
        option_token_ids = [
            [tokenizer(opt, add_special_tokens=False).input_ids[0] for opt in opts]
            for opts in all_options
        ]

        for prompt, opt_ids, target in zip(prompts, option_token_ids, correct_idxs):
            # 1) encode prompt
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=2048,
            ).to(device)

            # 2) single forward pass
            outputs = model(**enc)
            # outputs.logits: [1, seq_len, vocab_size]
            last_logits = outputs.logits[:, -1, :]  # [1, V]

            # 3) pick out the logits for our option tokens → [1, num_opts], so in this case letters (A, B, etc...)
            opt_logits = last_logits[:, opt_ids]  # [1, num_opts]
            batch_logits.append(opt_logits.squeeze(0))  # [num_opts]

            # 4) CE against the correct index
            tgt = torch.tensor([target], device=device)
            losses.append(F.cross_entropy(opt_logits, tgt))

            del enc, outputs, last_logits, opt_logits, tgt
            torch.cuda.empty_cache()

        loss = torch.stack(losses).mean()
        logits = torch.stack(batch_logits)  # [batch, num_opts]

        return (loss, logits) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        dataloader_params = {"batch_size": self.args.train_batch_size, "collate_fn": self.data_collator}
        return DataLoader(self.train_dataset, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        dataloader_params = {"batch_size": self.args.eval_batch_size, "collate_fn": self.data_collator}
        return DataLoader(eval_dataset, **dataloader_params)

    def evaluate(self, ignore_keys=None):
        model = self.model
        model.eval()
        dataloader = self.get_eval_dataloader(self.eval_dataset)
        device = model.device

        # track per‐dataset stats
        correct_by_ds = {}
        total_by_ds = {}

        # overall stats
        overall_correct = 0
        overall_total = 0

        with torch.inference_mode(), torch.amp.autocast("cuda"):
            for batch in dataloader:
                prompts = batch["prompt"]
                options = batch["options"]
                correct_idxs = batch["correct_idx"]
                datasets = batch["dataset"]

                for i in range(len(prompts)):
                    ds_name = datasets[i]
                    prompt = prompts[i]
                    opts = options[i]
                    target = correct_idxs[i]

                    # ensure counters exist
                    if ds_name not in correct_by_ds:
                        correct_by_ds[ds_name] = 0
                        total_by_ds[ds_name] = 0

                    # score each option by negative NLL
                    scores = []
                    for opt in opts:
                        enc = self.tokenizer(
                            prompt + opt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=2048,
                        ).to(device)
                        labels = enc["input_ids"].clone()
                        out = model(**enc, labels=labels)
                        nll = out.loss * labels.size(1)
                        scores.append(-nll.item())
                        del enc, labels, out
                        torch.cuda.empty_cache()

                    pred = int(torch.argmax(torch.tensor(scores)))

                    # update stats
                    is_correct = pred == target
                    correct_by_ds[ds_name] += int(is_correct)
                    total_by_ds[ds_name] += 1
                    overall_correct += int(is_correct)
                    overall_total += 1

        # compute accuracies
        acc_by_ds = {ds: correct_by_ds[ds] / total_by_ds[ds] for ds in correct_by_ds}
        overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

        # return as metrics dict
        metrics = {"accuracy": overall_acc}
        metrics.update({f"accuracy_{ds}": acc for ds, acc in acc_by_ds.items()})
        return metrics