import hydra
from omegaconf import DictConfig
import wandb
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTConfig, SFTTrainer
import torch
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@hydra.main(config_path="config", config_name="reasoning-config")
def train(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=dict(cfg))

    # Load dataset and tokenizer
    raw_datasets = load_dataset(cfg.dataset_name, cfg.dataset_config_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=2048)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

    # Load model
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        eval_strategy="steps",
        logging_dir=cfg.logging_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        report_to=cfg.report_to,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    train()
