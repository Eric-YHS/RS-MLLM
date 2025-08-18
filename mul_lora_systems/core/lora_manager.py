import os
import json
from typing import Optional, Dict, Any
from peft import PeftModel, PeftConfig
import torch


class LoRAManager:
    def __init__(self, base_model=None):
        self.base_model = base_model
        self.current_model = None
        self.current_lora_path = None
        self.current_lora_config = None
        self.loaded_loras = {}

    def set_base_model(self, base_model):
        self.base_model = base_model
        self.current_model = None
        self.current_lora_path = None
        self.current_lora_config = None

    def load_lora(self, lora_path: str, force_reload: bool = False) -> PeftModel:
        if self.base_model is None:
            raise ValueError("Base model not set. Please call set_base_model() first.")

        if not self._validate_lora_files(lora_path):
            raise FileNotFoundError(f"LoRA files not found at: {lora_path}")

        if (not force_reload and
                self.current_lora_path == lora_path and
                self.current_model is not None):
            return self.current_model

        try:
            if lora_path in self.loaded_loras and not force_reload:
                print(f"Loading LoRA from cache: {lora_path}")
                self.current_model = self.loaded_loras[lora_path]
            else:
                print(f"Loading LoRA from disk: {lora_path}")
                self.current_lora_config = PeftConfig.from_pretrained(lora_path)
                self.current_model = PeftModel.from_pretrained(
                    self.base_model,
                    lora_path,
                    is_trainable=False
                )
                self.loaded_loras[lora_path] = self.current_model

            self.current_lora_path = lora_path
            print(f"Successfully loaded LoRA: {lora_path}")
            return self.current_model

        except Exception as e:
            print(f"Error loading LoRA from {lora_path}: {e}")
            raise

    def switch_lora(self, new_lora_path: str) -> PeftModel:
        if new_lora_path == self.current_lora_path:
            print(f"Already using LoRA: {new_lora_path}")
            return self.current_model

        print(f"Switching LoRA from {self.current_lora_path} to {new_lora_path}")
        return self.load_lora(new_lora_path)

    def get_current_lora(self) -> Optional[str]:
        return self.current_lora_path

    def get_current_model(self) -> Optional[PeftModel]:
        return self.current_model

    def get_lora_info(self) -> Optional[Dict[str, Any]]:
        if self.current_lora_path is None:
            return None

        info = {
            "lora_path": self.current_lora_path,
            "config": None,
            "adapter_config": None
        }

        if self.current_lora_config:
            info["config"] = {
                "peft_type": self.current_lora_config.peft_type,
                "task_type": self.current_lora_config.task_type,
                "r": getattr(self.current_lora_config, 'r', None),
                "lora_alpha": getattr(self.current_lora_config, 'lora_alpha', None),
                "lora_dropout": getattr(self.current_lora_config, 'lora_dropout', None),
                "target_modules": getattr(self.current_lora_config, 'target_modules', None)
            }

        adapter_config_path = os.path.join(self.current_lora_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, 'r', encoding='utf-8') as f:
                    info["adapter_config"] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read adapter config: {e}")

        return info

    def unload_lora(self):
        if self.current_model is not None:
            print(f"Unloading LoRA: {self.current_lora_path}")
            self.current_model = None
            self.current_lora_path = None
            self.current_lora_config = None
            print("LoRA unloaded. Please reload base model if needed.")

    def clear_cache(self):
        self.loaded_loras.clear()
        print("LoRA cache cleared.")

    def list_cached_loras(self) -> list:
        return list(self.loaded_loras.keys())

    def _validate_lora_files(self, lora_path: str) -> bool:
        if not os.path.exists(lora_path):
            return False

        required_files = ["adapter_config.json", "adapter_model.bin"]
        for file_name in required_files:
            file_path = os.path.join(lora_path, file_name)
            if not os.path.exists(file_path):
                print(f"Missing required file: {file_path}")
                return False
        return True

    def get_model_memory_usage(self) -> Dict[str, Any]:
        if self.current_model is None:
            return {"error": "No model loaded"}

        try:
            total_params = sum(p.numel() for p in self.current_model.parameters())
            trainable_params = sum(p.numel() for p in self.current_model.parameters() if p.requires_grad)
            
            memory_info = {}
            if torch.cuda.is_available():
                memory_info = {
                    "gpu_memory_allocated": torch.cuda.memory_allocated(),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(),
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024
                }
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": total_params - trainable_params,
                "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
                "memory_info": memory_info,
                "current_lora": self.current_lora_path
            }
            
        except Exception as e:
            return {"error": f"Could not get memory usage: {e}"}


def test_lora_manager():
    print("LoRA Manager Test (Simulated):")
    print("-" * 50)
    
    manager = LoRAManager()
    
    print(f"Current LoRA: {manager.get_current_lora()}")
    print(f"Cached LoRAs: {manager.list_cached_loras()}")
    
    test_paths = [
        "mul_lora_systems/loras/text_to_lora/target_detection_lora",
        "mul_lora_systems/loras/text_to_lora/image_caption_lora",
        "mul_lora_systems/loras/dataset_finetuned/mmrs_chatearth_lora",
        "non_existent_path"
    ]
    
    print("\nFile Validation Test:")
    for path in test_paths:
        is_valid = manager._validate_lora_files(path)
        print(f"{path}: {'Valid' if is_valid else 'Invalid'}")
    
    print("\nNote: Full load testing requires an actual base model instance.")


if __name__ == "__main__":
    test_lora_manager()
