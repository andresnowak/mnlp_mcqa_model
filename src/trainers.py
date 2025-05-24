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
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        For each prompt we run the model once, grab the final (next‐token)
        logits, index into the letter‐token IDs, and compute a CE loss.
        """
        device = model.device
        prompts = inputs["prompt"]  # List[str]
        correct_idxs = inputs["correct_idx"]  # List[int]
        all_options = inputs["options"]  # List[List[str]]

        batch_logits = []
        losses = []

        # Pre‐tokenize all option‐letters to single token IDs
        option_token_ids = [
            [self.tokenizer(opt, add_special_tokens=False).input_ids[0] for opt in opts]
            for opts in all_options
        ]

        # 1) Batch encode prompts
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=2048,
        ).to(device)

        # 2) Forward pass over the full batch
        outputs = model(**enc)  # logits: [B, seq_len, V]
        last_logits = outputs.logits[:, -1, :]  # [B, V]

        # 3) Convert option_token_ids to tensor [B, O]
        opt_ids_tensor = torch.tensor(option_token_ids, device=device)  # [B, O]

        # 4) Gather logits at option token positions
        # This gives [B, O]
        opt_logits = torch.gather(
            last_logits.unsqueeze(1).expand(-1, opt_ids_tensor.shape[1], -1),  # [B, O, V]
            2,
            opt_ids_tensor.unsqueeze(2),  # [B, O, 1]
        ).squeeze(-1)  # -> [B, O]

        # 5) Cross-entropy loss per row
        targets = torch.tensor(correct_idxs, device=device)  # [B]
        loss = F.cross_entropy(opt_logits, targets)  # this will average over batch by default

        return (loss, opt_logits) if return_outputs else loss

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