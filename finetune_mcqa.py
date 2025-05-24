import unsloth
from unsloth import FastLanguageModel
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
from datasets import load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LinearLR
import logging
import transformers
import sys
import datasets
import os

from src.trainers import MCQATrainer
from src.custom_datasets import MCQADatasetClassification

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Available gpus {torch.cuda.device_count()}")
logger = logging.getLogger(__name__)


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


def mcqa_collatefn(batch):
    return {
        "prompt": [item["prompt"] for item in batch],
        "options": [item["options"] for item in batch],
        "correct_idx": [item["correct_idx"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
    }


@hydra.main(config_path="config", config_name="MCQA-config.yaml", version_base="1.1")
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

    raw_train_dataset = concatenate_datasets(
        [
            load_dataset(
                dataset_info["name"],
                dataset_info["subset_name"],
                split=dataset_info["config"],
            )
            for dataset_info in cfg.dataset_train
        ]
    )

    raw_val_dataset = concatenate_datasets(
        [
            load_dataset(
                dataset_info["name"],
                dataset_info["subset_name"],
                split=dataset_info["config"],
            )
            for dataset_info in cfg.dataset_validation
        ]
    )

    raw_train_dataset = raw_train_dataset.shuffle(seed=cfg.environment.seed)
    raw_val_dataset = raw_val_dataset.shuffle(seed=cfg.environment.seed)

    train_dataset = MCQADatasetClassification(raw_train_dataset, tokenizer)
    val_dataset = MCQADatasetClassification(raw_val_dataset, tokenizer)

    # Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="flash_attention_2",
        load_in_4bit=False,
        load_in_8bit=False,
    )
    # model = model.to(device) # the model is already passed to the device
    # It seems by default the model with unsloth doesn't have require grad = true, only when using lora it seems
    for param in model.parameters():
        param.requires_grad = True

    # Tokenizer setup
    # tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"  # Critical for Flash Attention compatibility (It seems Qwen3 Flash attention needs this <pad> value, instead of value <pad>)
    tokenizer.max_length = 2048

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
        warmup_ratio=cfg.training.warmup_ratio,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=10,
        report_to=cfg.training.report_to,
        save_strategy="steps",
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        lr_scheduler_type="linear",
        seed=cfg.environment.seed,
        push_to_hub=True,
        hub_model_id=cfg.model.hub_model_id,
    )

    trainer = MCQATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=mcqa_collatefn,
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    wandb.finish()

    # Push final model

    trainer.push_to_hub()


if __name__ == "__main__":
    train()
