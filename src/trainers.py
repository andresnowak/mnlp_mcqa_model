from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import Trainer, PreTrainedModel, PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
import numpy as np
from trl import SFTTrainer
from tqdm import tqdm

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
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)
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

    def evaluate(self, ignore_keys=None, metric_key_prefix="eval"):
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

        with torch.inference_mode(), torch.amp.autocast("cuda"), torch.no_grad():
            batch_pbar = tqdm(
                dataloader, desc="Evaluating", leave=False
            )
            for batch in batch_pbar:
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
        metrics = {f"{metric_key_prefix}_accuracy": overall_acc}
        metrics.update({f"{metric_key_prefix}_accuracy_{ds}": acc for ds, acc in acc_by_ds.items()})
        self.log(metrics)
        model.train()
        return metrics
    
# ------ Instruction finetuning trainer -------

# Note simple evaluation method where we grab the last logit and we only see the logits of the choice letters, and then we just do argmax and compare if chosen token is the same as the correct answer for accuracy
def evaluate_mmlu_accuracy(
    model, tokenizer, mmlu_datasets, max_length=2048, metric_key_prefix="eval"
):
    """
    Evaluate model on MMLU datasets
    """
    model.eval()
    results = {}
    choice_tokens = [
        tokenizer.encode(choice, add_special_tokens=False)[0]
        for choice in ["A", "B", "C", "D"]
    ]

    subject_pbar = tqdm(mmlu_datasets.items(), desc="MMLU Subjects", leave=False)
    for subject, dataset in subject_pbar:
        subject_pbar.set_postfix({"subject": subject[:20] + "..."})
        correct = 0
        total = 0

        example_pbar = tqdm(dataset, desc="Evaluating", leave=False)
        for example in example_pbar:
            question = example["question"]
            choices = example["choices"]
            correct_answer = example["answer"]  # This should be 0, 1, 2, or 3

            # Format the question
            def format_mmlu_question(question, choices):
                """Format MMLU question for evaluation"""
                choice_labels = ["A", "B", "C", "D"]
                formatted_choices = "\n".join(
                    [f"{label}. {choice}" for label, choice in zip(choice_labels, choices)]
                )

                prompt = f"""Question: {question}\n{formatted_choices}\nAnswer: """
                return prompt

            prompt = format_mmlu_question(question, choices)

            # Tokenize and get model prediction
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length - 1
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]  # Get last token logits, 
                # NOTE: Even if the model would output something else, the idea is to see how possible is it that after Answer: the model would output A, B, C, D. Like maybe in the latent space the model still has the idea of what it has to choose, it i just first generating something before it answers (like if its planning what it has to say, and already has an idea of the answer)

                # Get probabilities for A, B, C, D tokens
                choice_logits = logits[choice_tokens]
                predicted_choice = torch.argmax(choice_logits).item()

                if predicted_choice == correct_answer:
                    correct += 1
                total += 1
    
            example_pbar.set_postfix(
                {"acc": f"{correct / total:.1%}" if total else "0%"}
            )

        accuracy = correct / total if total > 0 else 0
        results[f"{metric_key_prefix}_mmlu_{subject}_accuracy"] = accuracy
        subject_pbar.set_postfix({"curr_acc": f"{accuracy:.1%}"})

    # Calculate overall MMLU accuracy
    if results:
        overall_accuracy = np.mean(list(results.values()))
        results[f"{metric_key_prefix}_mmlu_overall_accuracy"] = overall_accuracy
        # logger.info(f"Overall MMLU accuracy: {overall_accuracy:.3f}")

    model.train()
    return results


class IFSFTTrainer(SFTTrainer):
    """Custom trainer that includes MMLU evaluation"""

    def __init__(
        self, *args, mmlu_datasets=None, eval_dataset_name="validation", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mmlu_datasets = mmlu_datasets or {}
        self.eval_dataset_name = eval_dataset_name

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to include MMLU evaluation
        """
        # Standard evaluation on training dataset split
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # MMLU evaluation if datasets are provided
        if self.mmlu_datasets:
            mmlu_results = evaluate_mmlu_accuracy(
                self.model, self.tokenizer, self.mmlu_datasets
            )

        eval_results.update(mmlu_results)
        self.log(mmlu_results)

        # Return all results - trainer will handle logging automatically
        return eval_results