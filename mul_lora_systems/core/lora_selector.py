import json
import os
from typing import Dict, Optional, List


class LoRASelector:
    def __init__(self, config_path: str = "mul_lora_systems/configs/lora_config.json"):
        self.config_path = config_path
        self.lora_configs = self._load_config()
        self.task_to_lora_mapping = self._build_task_mapping()

    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("lora_configs", {})
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found, using default mappings")
            return self._get_default_lora_configs()

    def _get_default_lora_configs(self) -> Dict:
        return {
            "target_detection_lora": {
                "path": "mul_lora_systems/loras/text_to_lora/target_detection_lora",
                "description": "Dedicated LoRA for target recognition and counting"
            },
            "image_caption_lora": {
                "path": "mul_lora_systems/loras/text_to_lora/image_caption_lora",
                "description": "Dedicated LoRA for image description and semantic generation"
            },
            "spatial_grounding_lora": {
                "path": "mul_lora_systems/loras/text_to_lora/spatial_grounding_lora",
                "description": "Dedicated LoRA for spatial awareness and localization"
            },
            "land_classification_lora": {
                "path": "mul_lora_systems/loras/text_to_lora/land_classification_lora",
                "description": "Dedicated LoRA for land cover classification and status assessment"
            },
            "scene_reasoning_lora": {
                "path": "mul_lora_systems/loras/text_to_lora/scene_reasoning_lora",
                "description": "Dedicated LoRA for scene reasoning and decision making"
            },
            "multimodal_fusion_lora": {
                "path": "mul_lora_systems/loras/text_to_lora/multimodal_fusion_lora",
                "description": "Dedicated LoRA for cross-modal fusion and collaborative reasoning"
            },
            "mmrs_chatearth_lora": {
                "path": "mul_lora_systems/loras/dataset_finetuned/mmrs_chatearth_lora",
                "description": "General-purpose LoRA fine-tuned on MMRS-1M and ChatEarthNet datasets"
            }
        }

    def _build_task_mapping(self) -> Dict[str, str]:
        return {
            "target_detection": "target_detection_lora",
            "image_caption": "image_caption_lora",
            "spatial_grounding": "spatial_grounding_lora",
            "land_classification": "land_classification_lora",
            "scene_reasoning": "scene_reasoning_lora",
            "multimodal_fusion": "multimodal_fusion_lora",
            "default": "mmrs_chatearth_lora"
        }

    def select_lora(self, task_type: str) -> str:
        lora_name = self.task_to_lora_mapping.get(task_type)
        if not lora_name:
            lora_name = self.task_to_lora_mapping.get("default", "mmrs_chatearth_lora")
        
        lora_config = self.lora_configs.get(lora_name)
        if not lora_config:
            default_lora_name = self.task_to_lora_mapping.get("default", "mmrs_chatearth_lora")
            lora_config = self.lora_configs.get(default_lora_name)
            if not lora_config:
                raise ValueError(f"No LoRA configuration found for task type: {task_type}")
        
        return lora_config["path"]

    def get_lora_info(self, task_type: str) -> Dict:
        lora_name = self.task_to_lora_mapping.get(task_type)
        if not lora_name:
            lora_name = self.task_to_lora_mapping.get("default", "mmrs_chatearth_lora")
        
        lora_config = self.lora_configs.get(lora_name, {})
        
        return {
            "lora_name": lora_name,
            "path": lora_config.get("path", ""),
            "description": lora_config.get("description", ""),
            "training_method": lora_config.get("training_method", "unknown"),
            "capabilities": lora_config.get("capabilities", []),
            "task_type": task_type
        }

    def list_available_loras(self) -> List[Dict]:
        lora_list = []
        for lora_name, lora_config in self.lora_configs.items():
            task_type = None
            for task, lora in self.task_to_lora_mapping.items():
                if lora == lora_name and task != "default":
                    task_type = task
                    break
            
            if not task_type and lora_name == self.task_to_lora_mapping.get("default"):
                task_type = "default"
            
            lora_info = {
                "lora_name": lora_name,
                "task_type": task_type,
                "path": lora_config.get("path", ""),
                "description": lora_config.get("description", ""),
                "training_method": lora_config.get("training_method", "unknown"),
                "capabilities": lora_config.get("capabilities", [])
            }
            lora_list.append(lora_info)
        
        return lora_list

    def validate_lora_path(self, lora_path: str) -> bool:
        if not lora_path:
            return False
        
        adapter_config_path = os.path.join(lora_path, "adapter_config.json")
        adapter_model_path = os.path.join(lora_path, "adapter_model.bin")
        
        return os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path)

    def select_lora_with_validation(self, task_type: str) -> str:
        lora_path = self.select_lora(task_type)
        
        if not self.validate_lora_path(lora_path):
            default_path = self.select_lora("default")
            if self.validate_lora_path(default_path):
                print(f"Warning: LoRA for task '{task_type}' not found, using default LoRA")
                return default_path
            else:
                raise FileNotFoundError(f"LoRA files not found for task type: {task_type}")
        
        return lora_path

    def get_task_mapping(self) -> Dict[str, str]:
        return self.task_to_lora_mapping.copy()


def test_lora_selector():
    selector = LoRASelector()
    
    test_tasks = [
        "target_detection",
        "image_caption",
        "spatial_grounding",
        "land_classification",
        "scene_reasoning",
        "multimodal_fusion",
        "unknown_task"
    ]
    
    print("LoRA Selector Test Results:")
    print("-" * 60)
    
    for task in test_tasks:
        try:
            lora_path = selector.select_lora(task)
            lora_info = selector.get_lora_info(task)
            is_valid = selector.validate_lora_path(lora_path)
            
            print(f"Task Type: {task}")
            print(f"LoRA Name: {lora_info['lora_name']}")
            print(f"LoRA Path: {lora_path}")
            print(f"Path Valid: {is_valid}")
            print(f"Description: {lora_info['description']}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Task Type: {task}")
            print(f"Error: {e}")
            print("-" * 40)
    
    print("\nAll Available LoRA Models:")
    print("-" * 60)
    
    available_loras = selector.list_available_loras()
    for lora in available_loras:
        print(f"LoRA Name: {lora['lora_name']}")
        print(f"Task Type: {lora['task_type']}")
        print(f"Training Method: {lora['training_method']}")
        print(f"Capabilities: {', '.join(lora['capabilities'])}")
        print("-" * 30)


if __name__ == "__main__":
    test_lora_selector()
