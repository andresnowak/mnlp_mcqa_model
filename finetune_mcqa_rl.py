# import unsloth
# from unsloth import FastLanguageModel
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from trl import GRPOConfig, GRPOTrainer
import torch
from datasets import load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LinearLR
import logging
import transformers
import sys
import datasets
import os
# from unsloth import unsloth_train
from datetime import datetime
import random
from jinja2 import Environment, FileSystemLoader, ChoiceLoader
import re

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
template_dir = f"{os.getcwd()}/templates/MCQA"  # seems jinja wants the absolute path
template_files = [f for f in os.listdir(template_dir) if f.endswith(".jinja")]
jinja_env = Environment(loader=FileSystemLoader(template_dir))

# ------------------------


def load_mmlu_datasets(name="cais/mmlu", split="test", subjects=["abstract_algebra"]):
    """Load MMLU evaluation datasets"""
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
    """Format multiple-choice questions using Jinja templates"""
    chosen_template = random.choice(template_files)
    template = jinja_env.get_template(chosen_template)
    question_val = question
    choices = [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    formatted_text = template.render(question=question_val, choices_list=choices)
    return formatted_text


def format_mcqa_answer(answer, choices, tokenizer):
    """Format the answer for multiple-choice questions"""
    pos = ord(answer) - ord("A")
    completion = f"{answer}. {choices[pos]}{tokenizer.eos_token}"
    return completion


def tokenize_mcqa_with_labels(examples, tokenizer):
    """Tokenize MCQA examples with proper labels for training"""
    prompts_list = []
    completion_list = []
    choices_list = []
    answer_list = []

    for question, choices, answer in zip(
        examples["question"], examples["choices"], examples["answer"]
    ):
        prompt = format_mcqa_questions(question, choices, tokenizer)
        completion = format_mcqa_answer(answer, choices, tokenizer)

        prompts_list.append(prompt)
        completion_list.append(completion)
        choices_list.append(choices)
        answer_list.append(answer)

    return {
            "prompt": prompts_list,
            "completion": completion_list,
            "choices": choices_list,
            "correct_answer_letter": answer_list
        }



def extract_predicted_answer(output_text):
    """Extract the predicted answer (A, B, C, or D) from model output"""
    # Look for the first occurrence of A., B., C., or D. in the output
    match = re.search(r"\b([A-D])\.", output_text)
    if match:
        return match.group(1)
    # Fallback: look for standalone A, B, C, or D
    match = re.search(r"\b([A-D])\b", output_text)
    return match.group(1) if match else None


def mcqa_reward_function(prompts, completions, choices, correct_answer_letter, **kwargs):
    """Compute rewards for MCQA responses"""
    rewards = []

    for completion, choice, correct_answer in zip(completions, choices, correct_answer_letter):
        predicted_answer = extract_predicted_answer(completion)
        if predicted_answer and predicted_answer in ["A", "B", "C", "D"]:
            reward = 1.0 if predicted_answer == correct_answer else -1.0
        else:
            # Penalize invalid responses
            reward = -1.0

        rewards.append(reward)

    return rewards


def get_wandb_id(cfg):
    """Get or create wandb run ID"""
    wandb_id_path = os.path.join(cfg.training.output_dir, "wandb_run_id.txt")
    if os.path.exists(wandb_id_path):
        with open(wandb_id_path, "r") as f:
            wandb_id = f.read().strip()
        resume_mode = "must"
    else:
        wandb_id = None
        resume_mode = "allow"
    return wandb_id, resume_mode


@hydra.main(
    config_path="config", config_name="MCQA-RL.yaml", version_base="1.1"
)
def train(cfg: DictConfig):
    """Main training function"""
    random.seed(cfg.environment.seed)

    # Resume from checkpoint
    last_checkpoint = None
    if os.path.isdir(cfg.training.output_dir):
        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = get_last_checkpoint(cfg.training.output_dir)

    # Setup logging
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

    # Initialize wandb
    wandb_id = get_wandb_id(cfg)
    run = wandb.init(
        id=wandb_id[0],
        resume=wandb_id[1],
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}_{datetime.now().strftime('%Y-%m-%d')}",
        config=OmegaConf.to_container(cfg, resolve=True),
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

    # Load model
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     cfg.model.name,
    #     dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    #     attn_implementation="flash_attention_2",
    #     load_in_4bit=False,
    #     load_in_8bit=False,
    # )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        # attn_implementation="flash_attention_2"
        )

    # Enable gradient computation
    for param in model.parameters():
        param.requires_grad = True

    # Tokenizer setup
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"
    tokenizer.max_length = 2048

    # Load datasets
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

    # Tokenize datasets
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

    # Log training info
    total_batch_size = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.training.num_train_epochs}")
    logger.info(f"  Batch size per device = {cfg.training.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}"
    )

    # Training configuration
    training_args = GRPOConfig(
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
        eval_steps=3000,
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
        max_completion_length=cfg.training.completion_length,
        num_generations=cfg.training.num_generations,
        beta=cfg.training.beta
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_processing_classes=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_funcs=mcqa_reward_function,
    )

    # Start training
    trainer.train(resume_from_checkpoint=last_checkpoint)
    # unsloth_train(trainer, resume_from_checkpoint=last_checkpoint)
    wandb.finish()

    # Push final model to hub
    trainer.push_to_hub()


if __name__ == "__main__":
    train()
