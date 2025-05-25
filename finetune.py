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
from unsloth import unsloth_train
from datetime import datetime

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Available gpus {torch.cuda.device_count()}")
logger = logging.getLogger(__name__)

# ------------------------

def load_mmlu_datasets(name="cais/mmlu", split="test", subjects):
    """
    Load MMLU evaluation datasets
    subjects: List of MMLU subjects to evaluate on, or None for all
    num_samples_per_subject: Number of samples per subject for evaluation
    """

    mmlu_datasets = {}
    for subject in subjects:
        try:
            dataset = load_dataset(name, subject, split=split)
            mmlu_datasets[subject] = dataset
            logger.info(f"Loaded {len(dataset)} samples for MMLU subject: {subject}")
        except Exception as e:
            logger.warning(f"Failed to load MMLU subject {subject}: {e}")

    return mmlu_datasets


def format_chat_messages(messages, tokenizer):
    formatted_text = ""
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")

        if role == "user":
            formatted_text += f"{content}\n\n"
        elif role == "assistant":
            formatted_text += f"{content}\n\n"
        else:
            raise ValueError
        formatted_text += (
            tokenizer.eos_token
        )  # add eos_token so it doesn't go on forever

    return formatted_text.strip()


def tokenize_chat_function(examples, tokenizer):
    """
    Tokenize chat-based examples where each example has a 'messages' field
    containing a list of message dictionaries.
    """
    texts = [
        format_chat_messages(messages, tokenizer) for messages in examples["messages"]
    ]

    return {"text": texts}


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


@hydra.main(config_path="config", config_name="IF-config.yml", version_base="1.1")
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
        name=f"{cfg.wandb.name}_{datetime.now().strftime('%Y-%m-%d')}",
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


    # Load datasets
    raw_train_datasets = concatenate_datasets(
        [
            load_dataset(dataset["name"], dataset["config"], split="train")
            .shuffle(cfg.environment.seed)
            .take(int(dataset["size"] * 100))
            for dataset in cfg.dataset
        ]
    ).shuffle(seed=cfg.environment.seed)

    # Then select the number of samples you want from the shuffled dataset
    # if cfg.dataset[0].get("samples"):
    #     raw_train_datasets = raw_train_datasets.select(range(cfg.dataset[0].samples))

    # Tokenization with instruction formatting
    tokenized_dataset = raw_train_datasets.map(
        lambda x: tokenize_chat_function(x, tokenizer),
        batched=True,
        num_proc=4,
    )
    split = tokenized_dataset.train_test_split(test_size=0.05)

    ## log
    total_batch_size = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(split['train'])}")
    logger.info(f"  Num Epochs = {cfg.training.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {cfg.training.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}"
    )

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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # trainer.train(resume_from_checkpoint=last_checkpoint)
    unsloth_train(
        trainer, resume_from_checkpoint=last_checkpoint
    )  # use unsloth to have the fix of the gradient accumulation
    wandb.finish()

    # Push final model
    trainer.push_to_hub()


if __name__ == "__main__":
    train()
