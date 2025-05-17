import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig
import torch
from datasets import load_dataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@hydra.main(config_path="config", config_name="IF-config_sweep.yml", version_base="1.1")
def train(cfg: DictConfig):
    # Initialize wandb (ensure no legacy-service warnings)
    wandb.init(
        project=cfg.wandb.project, 
        name=cfg.wandb.name,  
        config=OmegaConf.to_container(cfg, resolve=True),  # export all cfg to wandb)
    )

    # Override with sweep parameters
    if wandb.config:
        cfg.training.learning_rate = wandb.config["training"]["learning_rate"]
        cfg.training.per_device_train_batch_size = wandb.config["training"][
            "per_device_train_batch_size"
        ]
        cfg.training.num_train_epochs = wandb.config["training"]["num_train_epochs"]
        cfg.training.weight_decay = wandb.config["training"]["weight_decay"]

    # Load dataset with subset for sweeps
    raw_train_datasets = load_dataset(
        cfg.dataset[0].name,
        cfg.dataset[0].config,
        split={
            "train": f"train[:{cfg.dataset[0].samples}]"
            if cfg.dataset[0].get("samples")
            else "train",
        },
    )

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )

    # Tokenization with instruction formatting
    def format_instruction(example):
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

    def tokenize_function(examples):
        texts = [format_instruction(ex) for ex in examples]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=2048,
            return_tensors="pt",
        )

    tokenized_datasets = raw_train_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_train_datasets["train"].column_names,
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ).to(device)

    # Training setup
    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        learning_rate=float(cfg.training.learning_rate),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        num_train_epochs=cfg.training.num_train_epochs,
        weight_decay=cfg.training.weight_decay,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        report_to=cfg.training.report_to,
        save_strategy="steps",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    train()
