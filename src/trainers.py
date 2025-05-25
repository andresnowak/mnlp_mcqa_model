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
        # Get device and dtype from trainer args
        device = model.device
        dtype = torch.bfloat16 if self.args.bf16 else (
            torch.float16 if self.args.fp16 else torch.float32
        )
        
        # Input validation (unchanged)
        prompts = inputs["prompt"]
        correct_idxs = inputs["correct_idx"]
        all_options = inputs["options"]
        
        # 1. Tokenize options and create mask
        option_token_ids = [
            [self.tokenizer(opt, add_special_tokens=False).input_ids[0] for opt in opts]
            for opts in all_options
        ]
        
        # 2. Create padded tensors
        max_options = max(len(ids) for ids in option_token_ids)
        opt_ids_tensor = torch.full(
            (len(prompts), max_options),
            fill_value=0,  # padding index
            device=device,
            dtype=torch.long  # Must remain long for indexing
        )
        opt_mask = torch.zeros(
            (len(prompts), max_options),
            device=device,
            dtype=torch.bool
        )
        
        for i, ids in enumerate(option_token_ids):
            opt_ids_tensor[i, :len(ids)] = torch.tensor(ids, device=device)
            opt_mask[i, :len(ids)] = True

        # 3. Forward pass with autocast
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=dtype,
            enabled=self.args.fp16 or self.args.bf16  # Respect trainer's AMP setting
        ):
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            outputs = model(**enc)
            last_logits = outputs.logits[:, -1, :]  # [B, V]
            
            # Advanced indexing
            batch_idx = torch.arange(len(prompts), device=device)[:, None]
            opt_logits = last_logits[batch_idx, opt_ids_tensor]  # [B, max_O]
            opt_logits = opt_logits.masked_fill(~opt_mask, -float('inf'))

        # 4. Loss computation (PyTorch handles dtype internally)
        loss = F.cross_entropy(
            opt_logits,  # dtype already handled by autocast
            torch.tensor(correct_idxs, device=device),
            ignore_index=-100
        )
        
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