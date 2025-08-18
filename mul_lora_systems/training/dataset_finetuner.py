import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, AutoTokenizer, TrainingArguments, Trainer,
    get_cosine_schedule_with_warmup, AdamW
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import argparse
import logging
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetFinetunConfig:
    base_model_path: str = "."
    output_dir: str = "mul_lora_systems/loras/dataset_finetuned/mmrs_chatearth_lora"
    config_path: str = "mul_lora_systems/configs/training_config.json"
    
    mmrs_data_path: str = "/data/remote_sensing/MMRS-1M"
    chatearth_data_path: str = "/data/remote_sensing/ChatEarthNet"
    
    lora_r: int = 64
    lora_alpha: int = 128
    
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_length: int = 2048


class AbstractDataset(Dataset):
    def __init__(self, data_path: str, name: str, expected_structure: str, specs: str):
        self.data_path = data_path
        self.name = name
        self.expected_structure = expected_structure
        self.specs = specs
        self._validate_path()

    def _validate_path(self):
        logger.info(f"Checking {self.name} dataset path: {self.data_path}")
        if not os.path.exists(self.data_path):
            error_message = (
                f"{self.name} dataset not found at: {self.data_path}\n\n"
                f"Please ensure the dataset is downloaded and placed in the correct path.\n"
                f"Expected directory structure:\n{self.expected_structure}\n"
                f"Dataset specifications:\n{self.specs}"
            )
            raise FileNotFoundError(error_message)
        logger.info(f"{self.name} path check passed.")

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset not loaded successfully, cannot get item.")


class MMRS1MDataset(AbstractDataset):
    def __init__(self, data_path: str):
        name = "MMRS-1M"
        expected_structure = (
            f"  {data_path}/\n"
            f"  ├── images/\n"
            f"  │   ├── mmrs_000001.jpg\n"
            f"  │   └── ...\n"
            f"  ├── annotations/\n"
            f"  │   ├── train.json\n"
            f"  │   └── ...\n"
        )
        specs = (
            "  - Total samples: 1,000,000\n"
            "  - Tasks: Image Captioning, Visual QA, Visual Grounding, etc."
        )
        super().__init__(data_path, name, expected_structure, specs)


class ChatEarthNetDataset(AbstractDataset):
    def __init__(self, data_path: str):
        name = "ChatEarthNet"
        expected_structure = (
            f"  {data_path}/\n"
            f"  ├── images/\n"
            f"  │   ├── chatearth_000001.tif\n"
            f"  │   └── ...\n"
            f"  ├── qa_pairs/\n"
            f"  │   ├── train_qa.json\n"
            f"  │   └── ...\n"
        )
        specs = (
            "  - Total samples: 200,000\n"
            "  - Tasks: Visual QA, Scene Reasoning, etc."
        )
        super().__init__(data_path, name, expected_structure, specs)


class DatasetFinetuner:
    def __init__(self, config: DatasetFinetunConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup_model_and_tokenizer(self):
        logger.info("Loading base model and tokenizer...")
        if os.path.exists(self.config.base_model_path):
            self.model = AutoModel.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            if torch.cuda.is_available():
                self.model.to("cuda")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True
            )
            logger.info("Base model and tokenizer loaded successfully.")
        else:
            raise FileNotFoundError(f"Base model path not found: {self.config.base_model_path}")

    def setup_lora_config(self) -> LoraConfig:
        logger.info("Configuring LoRA parameters...")
        target_modules = [
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ]
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        logger.info(f"LoRA configuration complete: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        return lora_config

    def train(self):
        logger.info("Starting dataset finetuning process...")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.setup_model_and_tokenizer()
        
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA applied to the model.")

        logger.info("Preparing to load training dataset...")
        train_dataset_mmrs = MMRS1MDataset(self.config.mmrs_data_path)
        
        logger.info("Dataset loaded successfully, preparing to configure trainer...")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=500,
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset_mmrs,
        )
        
        logger.info("Starting model training...")
        trainer.train()
        logger.info("Training complete! Model has been saved.")


def main():
    parser = argparse.ArgumentParser(description="Dataset Finetuner")
    parser.add_argument("--base_model", type=str, default=".", help="Base model path")
    parser.add_argument("--output_dir", type=str, 
                        default="mul_lora_systems/loras/dataset_finetuned/mmrs_chatearth_lora", 
                        help="Output directory")
    args = parser.parse_args()
    
    config = DatasetFinetunConfig(
        base_model_path=args.base_model,
        output_dir=args.output_dir
    )
    
    finetuner = DatasetFinetuner(config)
    finetuner.train()


if __name__ == "__main__":
    main()