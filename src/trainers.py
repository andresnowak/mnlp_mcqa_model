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
        prompts = inputs["prompt"]
        correct_idxs = inputs["correct_idx"]  # List[int]
        all_options = inputs["options"]  # List[List[str]]

        # Check for empty options

        assert all(len(opts) > 0 for opts in all_options), "Empty options list found"

        # Verify correct indices are within bounds
        for idx, (opts, target) in enumerate(zip(all_options, correct_idxs)):
            assert 0 <= target < len(opts), (
                f"Invalid target {target} for {len(opts)} options in sample {idx}"
            )
        
        # 1. Tokenize all options and create mask
        option_token_ids = []
        option_mask = []
        for opts in all_options:
            ids = [self.tokenizer(opt, add_special_tokens=False).input_ids[0] for opt in opts]
            option_token_ids.append(ids)
            option_mask.append([1] * len(ids))  # 1 = real option
        
        # 2. Pad options and masks
        max_options = max(len(x) for x in option_token_ids)
        padded_ids = [ids + [0]*(max_options - len(ids)) for ids in option_token_ids]
        padded_mask = [mask + [0]*(max_options - len(mask)) for mask in option_mask]
        
        # Convert to tensors
        opt_ids_tensor = torch.tensor(padded_ids, device=device)  # [B, max_O]
        opt_mask = torch.tensor(padded_mask, dtype=torch.bool, device=device)  # [B, max_O]
        
        # 3. Forward pass (batched)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        outputs = model(**enc)
        last_logits = outputs.logits[:, -1, :]  # [B, V]
        
        # 4. Extract option logits WITHOUT gather
        # Index directly using advanced indexing
        batch_idx = torch.arange(len(prompts), device=device)[:, None]  # [B, 1]
        opt_logits = last_logits[batch_idx, opt_ids_tensor]  # [B, max_O]
        
        # 5. Apply mask by setting invalid options to -inf
        opt_logits = opt_logits.masked_fill(~opt_mask, -float('inf'))
        
        # 6. Compute loss (automatically ignores -inf)
        loss = F.cross_entropy(
            opt_logits,
            torch.tensor(correct_idxs, device=device),
            ignore_index=-100  # Redundant with -inf but safer
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