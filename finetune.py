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
from torch.optim.lr_scheduler import LinearLR
import logging
import transformers
import sys
import datasets
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger = logging.getLogger(__name__)


def format_instruction(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def tokenize_function(examples, tokenizer):
    texts = [format_instruction(ex) for ex in examples]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt",
    )


def get_wandb_id(cfg):
    wandb_id_path = os.path.join(cfg.training.output_dir, "wandb_run_id.txt")

    if os.path.exists(wandb_id_path):
        with open(wandb_id_path, "r") as f:
            wandb_id = f.read().strip()
        resume_mode = "must"
    else:
        wandb_id = None
        resume_mode = "allow"

    return wandb_id, resume_mode


@hydra.main(config_path="config", config_name="IF-config_sweep.yml", version_base="1.1")
def train(cfg: DictConfig):
    # Resume from checkpoint
    # Look for a latest checkpoint in the output directory
    last_checkpoint = None
    if os.path.isdir(cfg.training.output_dir):
        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = get_last_checkpoint(cfg.training.output_dir)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = 1
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Initialize wandb (ensure no legacy-service warnings)
    wandb_id = get_wandb_id(cfg)
    run = wandb.init(
        id=wandb_id[0],
        resume=wandb_id[1],
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),  # export all cfg to wandb)
    )
    wandb_id_path = os.path.join(cfg.training.output_dir, "wandb_run_id.txt")
    if not os.path.exists(wandb_id_path):
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        with open(wandb_id_path, "w") as f:
            f.write(run.id)

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
    split = raw_train_datasets.train_test_split(test_size=0.05)

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )

    # Tokenization with instruction formatting

    # tokenized_datasets = raw_train_datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     remove_columns=raw_train_datasets["train"].column_names,
    # )

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
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=50,
        report_to=cfg.training.report_to,
        save_strategy="steps",
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        lr_scheduler_type="linear",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        # dataset_text_field="text",
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    wandb.finish()


if __name__ == "__main__":
    train()
