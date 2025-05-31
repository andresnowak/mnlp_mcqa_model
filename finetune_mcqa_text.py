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
import random
from jinja2 import Environment, FileSystemLoader, ChoiceLoader

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

# Define paths to your template folders
template_dir = f"{os.getcwd()}/templates/MCQA" # seems jinja wants the absolute path
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


def format_mcqa_questions(question, choices, tokenizer):
    formatted_text = ""

    chosen_template = random.choice(template_files)
    template = jinja_env.get_template(chosen_template)

    question_val = question
    choices = [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    formatted_text = template.render(question=question_val, choices_list=choices)

    # formatted_text += (
    #     tokenizer.eos_token
    # )

    return formatted_text

def format_mcqa_answer(answer, choices, tokenizer):
    pos = ord(answer) -  ord("A")

    completion = f"{answer}. {choices[pos]}{tokenizer.eos_token}"  # add eos_token so it doesn't go on forever

    return completion


def tokenize_mcqa_with_labels(examples, tokenizer):
    """
    Create input_ids and labels where labels are -100 for prompt tokens.
    This gives precise control over which tokens contribute to loss.
    """
    input_ids_list = []
    completion_mask_list = []
    
    for question, choices, answer in zip(examples["question"], examples["choices"], examples["answer"]):
        # Get prompt and completion
        prompt = format_mcqa_questions(question, choices, tokenizer)
        completion = format_mcqa_answer(
            answer, choices, tokenizer
        )
        
        # Tokenize separately to know the lengths
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False, max_length=2048)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
        
        # Combine tokens
        input_ids = prompt_tokens + completion_tokens
        
        # Create labels: -100 for prompt tokens, actual tokens for completion

        # Create attention mask (1 for all real tokens)
        completion_mask = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    
        
        input_ids_list.append(input_ids)
        completion_mask_list.append(completion_mask)
    
    return {
        "input_ids": input_ids_list,
        "completion_mask": completion_mask_list
    }


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


@hydra.main(config_path="config", config_name="MCQA-text_config.yaml", version_base="1.1")
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


    # ---- Load training dataset ----- 
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

    # Tokenization with instruction formatting
    train_dataset = raw_train_dataset.map(
        lambda x: tokenize_mcqa_with_labels(x, tokenizer),
        batched=True,
        num_proc=30,
    )
    val_dataset = raw_val_dataset.map(
        lambda x: tokenize_mcqa_with_labels(x, tokenizer),
        batched=True,
        num_proc=30,
    )

    mmlu_datasets = load_mmlu_datasets(
        cfg.dataset_mmlu[0].name, cfg.dataset_mmlu[0].config, cfg.dataset_mmlu[0].subjects
    )

    # ---- log -----
    total_batch_size = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
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
        max_seq_length=2048,
    )

    trainer = IFSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # dataset_text_field="text",
        mmlu_datasets=mmlu_datasets,
        eval_dataset_name="training_validation_split", 
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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
