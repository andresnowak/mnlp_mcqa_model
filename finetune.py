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

import logging
import transformers
import sys
import datasets
import os
from unsloth import unsloth_train
from datetime import datetime
import random
from jinja2 import Environment, FileSystemLoader

from src.trainers import IFSFTTrainer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Available gpus {torch.cuda.device_count()}")
logger = logging.getLogger(__name__)

# Templates
template_dir = f"{os.getcwd()}/templates/IF" # seems jinja wants the absolute path
template_files = [f for f in os.listdir(template_dir) if f.endswith('.jinja')]
jinja_env = Environment(loader=FileSystemLoader(template_dir))

# ------------------------

def load_mmlu_datasets(name="cais/mmlu", split="test", subjects=["abstract_algebra"]):
    """
    Load MMLU evaluation datasets
    subjects: List of MMLU subjects to evaluate on, or None for all
    num_samples_per_subject: Number of samples per subject for evaluation
    """

    # so as to be able to see more finegrained evaluation (based on each type of subject)
    mmlu_datasets = {}
    for subject in subjects:
        try:
            dataset = load_dataset(name, subject, split=split)
            mmlu_datasets[subject] = dataset
            logger.info(f"Loaded {len(dataset)} samples for MMLU subject: {subject}")
        except Exception as e:
            logger.warning(f"Failed to load MMLU subject {subject}: {e}")

    return mmlu_datasets


def format_chat_messages(messages, tokenizer, use_template=True):
    instruction = ""
    answer = ""

    assert(len(messages) == 2)
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")

        if role == "user":
            instruction = content
        elif role == "assistant":
            answer = content
        else:
            raise ValueError

    chosen_template = random.choice(template_files)
    template = jinja_env.get_template(chosen_template)

    if use_template:
        formatted_text = template.render(instruction=instruction, answer=answer)
    else:
        formatted_text = f"{instruction}\n\n{answer}"

    formatted_text += tokenizer.eos_token # add eos_token so it doesn't go on forever

    return formatted_text


def tokenize_chat_function(examples, tokenizer):
    """
    Tokenize chat-based examples where each example has a 'messages' field
    containing a list of message dictionaries.
    """
    texts = [
        format_chat_messages(messages, tokenizer) for messages in examples["messages"]
    ]

    return {"text": texts}

    # text_tokenized = tokenizer(
    #     texts,
    #     max_length=2048,
    #     truncation=True,
    #     padding="longest",
    #     return_tensors="pt",
    #     add_special_tokens=True, 
    # )
    # NOTE: did this change so we always have eos token at the end
    # eos_token_id = tokenizer.eos_token_id
    # pad_token_id = tokenizer.pad_token_id
    
    # # Skip adjustment if no EOS token
    # if eos_token_id is None:
    #     return text_tokenized
    
    # # Calculate sequence lengths (number of non-pad tokens)
    # input_ids = text_tokenized["input_ids"]
    # non_pad_mask = input_ids != pad_token_id
    # seq_lengths = non_pad_mask.sum(dim=1)  # Vectorized computation
    
    # # Create indices for last tokens [batch_size, 2]
    # batch_indices = torch.arange(input_ids.size(0))
    # last_token_indices = seq_lengths - 1  # Position of last non-pad token
    
    # # Only modify sequences with at least 1 token
    # valid_seqs = seq_lengths > 0
    # batch_indices = batch_indices[valid_seqs]
    # last_token_indices = last_token_indices[valid_seqs]
    
    # # Vectorized assignment
    # input_ids[batch_indices, last_token_indices] = eos_token_id

    # input_ids = text_tokenized["input_ids"]  # (batch_size, seq_len)
    # eos_token_id = tokenizer.eos_token_id
    # last_token = input_ids[:, -1]  # last token in each sequence
    
    # # Only replace if it's not already eos
    # needs_replace = last_token != eos_token_id
    # input_ids[needs_replace, -1] = eos_token_id
    
    # text_tokenized["input_ids"] = input_ids
        
    # return text_tokenized


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


@hydra.main(config_path="config", config_name="IF-config_2.yml", version_base="1.1")
def train(cfg: DictConfig):
    random.seed(cfg.environment.seed)

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
    log_level = logging.INFO
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
        # attn_implementation="flash_attention_2",
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=True,  # this is necessary to activate gradiendts and do upcast in some layers
    )
    # Just because full_finetuning has problems with different versions of torch and unsloth and triton
    # model = model.to(device) # the model is already passed to the deviceAdd commentMore actions
    # It seems by default the model with unsloth doesn't have require grad = true, only when using lora it seems


    # Tokenizer setup
    # tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"  # Critical for Flash Attention compatibility (It seems Qwen3 Flash attention needs this <pad> value, instead of value <pad>)
    tokenizer.max_length = 2048


    # ---- Load training dataset ----- 
    dataset_list = []
    for dataset in cfg.dataset:
        # Load the full dataset first
        full_dataset = load_dataset(dataset["name"], dataset["config"], split="train")
        dataset_length = len(full_dataset)

        # Calculate number of samples to take (percentage of total)
        num_samples = int(dataset["size"] * dataset_length)
        logger.info(
            f"Taking {num_samples} samples ({dataset['size'] * 100:.1f}%) from {dataset['name']} (total: {dataset_length})"
        )

        # Shuffle and take the specified percentage
        sampled_dataset = full_dataset.shuffle(cfg.environment.seed).select(
            range(min(num_samples, dataset_length))
        )
        dataset_list.append(sampled_dataset)

    # filter examples, this will have more than 2048 tokens
    def filter_long_examples(example):
        # Format the messages to calculate total length
        formatted_text = format_chat_messages(example["messages"], tokenizer, cfg.environment.use_template)
        return len(formatted_text) <= 15_000

    raw_train_datasets = concatenate_datasets(dataset_list).shuffle(
        seed=cfg.environment.seed
    ).filter(filter_long_examples, num_proc=10)

    # Tokenization with instruction formatting
    tokenized_dataset = raw_train_datasets.map(
        lambda x: tokenize_chat_function(x, tokenizer),
        batched=True,
        num_proc=30,
    )
    split = tokenized_dataset.train_test_split(test_size=0.05)

    # load mmlu
    mmlu_datasets = load_mmlu_datasets(
        cfg.dataset_evaluation[0].name, cfg.dataset_evaluation[0].config, cfg.dataset_evaluation[0].subjects
    )

    # ---- log -----
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
        eval_steps=300,
        logging_steps=10,
        report_to=cfg.training.report_to,
        save_strategy="steps",
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        lr_scheduler_type="linear",
        seed=cfg.environment.seed,
        push_to_hub=cfg.training.push_to_hub,
        hub_model_id=cfg.model.hub_model_id,
    )

    trainer = IFSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        mmlu_datasets=mmlu_datasets,
        eval_dataset_name="training_validation_split",  # Name for your training data validation split
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        max_seq_length=2048,
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
