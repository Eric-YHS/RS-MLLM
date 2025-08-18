import os
import sys
from typing import Optional, Dict, Any, Tuple, Union
from PIL import Image
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.task_classifier import TaskClassifier
from core.lora_selector import LoRASelector
from core.lora_manager import LoRAManager


class MultiLoRAInference:
    def __init__(self,
                 base_model=None,
                 tokenizer=None,
                 task_config_path: str = "mul_lora_systems/configs/task_mapping.json",
                 lora_config_path: str = "mul_lora_systems/configs/lora_config.json",
                 enable_caching: bool = True,
                 verbose: bool = True):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.enable_caching = enable_caching
        self.verbose = verbose
        
        self.task_classifier = TaskClassifier(task_config_path)
        self.lora_selector = LoRASelector(lora_config_path)
        self.lora_manager = LoRAManager(base_model)
        
        self.inference_stats = {
            "total_inferences": 0,
            "task_type_counts": {},
            "lora_usage_counts": {},
            "average_confidence": 0.0
        }
        
        if self.verbose:
            print("MultiLoRAInference initialized successfully")
            print(f"Available LoRAs: {len(self.lora_selector.list_available_loras())}")

    def set_models(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.lora_manager.set_base_model(base_model)
        
        if self.verbose:
            print("Base model and tokenizer updated")

    def inference(self,
                  image: Union[str, Image.Image],
                  question: str,
                  max_length: int = 2048,
                  temperature: float = 0.7,
                  do_sample: bool = True,
                  **kwargs) -> str:
        if self.base_model is None or self.tokenizer is None:
            raise ValueError("Base model and tokenizer must be set before inference")
        
        try:
            task_type, confidence = self.task_classifier.classify_with_confidence(question)
            
            if self.verbose:
                print(f"Task classified as: {task_type} (confidence: {confidence:.3f})")
            
            lora_path = self.lora_selector.select_lora_with_validation(task_type)
            lora_info = self.lora_selector.get_lora_info(task_type)
            
            if self.verbose:
                print(f"Selected LoRA: {lora_info['lora_name']}")
                print(f"LoRA path: {lora_path}")
            
            model_with_lora = self.lora_manager.switch_lora(lora_path)
            
            response = self._execute_inference(
                model_with_lora,
                image,
                question,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
            
            self._update_stats(task_type, lora_info['lora_name'], confidence)
            
            if self.verbose:
                print(f"Inference completed successfully")
            
            return response
            
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

    def _execute_inference(self,
                           model,
                           image: Union[str, Image.Image],
                           question: str,
                           **generation_kwargs) -> str:
        try:
            if isinstance(image, str):
                if os.path.exists(image):
                    image = Image.open(image).convert('RGB')
                else:
                    raise FileNotFoundError(f"Image file not found: {image}")
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a file path or PIL Image object")
            
            msgs = [{'role': 'user', 'content': [image, question]}]
            
            response = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                **generation_kwargs
            )
            
            return response
            
        except Exception as e:
            print(f"Error in model inference: {e}")
            raise

    def batch_inference(self,
                        image_question_pairs: list,
                        **generation_kwargs) -> list:
        results = []
        
        for i, (image, question) in enumerate(image_question_pairs):
            try:
                if self.verbose:
                    print(f"Processing batch item {i+1}/{len(image_question_pairs)}")
                
                response = self.inference(image, question, **generation_kwargs)
                results.append({
                    "index": i,
                    "image": image,
                    "question": question,
                    "response": response,
                    "status": "success"
                })
                
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                results.append({
                    "index": i,
                    "image": image,
                    "question": question,
                    "response": None,
                    "status": "error",
                    "error": str(e)
                })
        
        return results

    def get_task_prediction(self, question: str) -> Dict[str, Any]:
        task_type, confidence = self.task_classifier.classify_with_confidence(question)
        all_confidences = self.task_classifier.get_task_confidence(question)
        lora_info = self.lora_selector.get_lora_info(task_type)
        
        return {
            "question": question,
            "predicted_task": task_type,
            "confidence": confidence,
            "all_task_confidences": all_confidences,
            "selected_lora": lora_info,
            "current_lora": self.lora_manager.get_current_lora()
        }

    def _update_stats(self, task_type: str, lora_name: str, confidence: float):
        self.inference_stats["total_inferences"] += 1
        
        if task_type not in self.inference_stats["task_type_counts"]:
            self.inference_stats["task_type_counts"][task_type] = 0
        self.inference_stats["task_type_counts"][task_type] += 1
        
        if lora_name not in self.inference_stats["lora_usage_counts"]:
            self.inference_stats["lora_usage_counts"][lora_name] = 0
        self.inference_stats["lora_usage_counts"][lora_name] += 1
        
        total = self.inference_stats["total_inferences"]
        current_avg = self.inference_stats["average_confidence"]
        self.inference_stats["average_confidence"] = (current_avg * (total - 1) + confidence) / total

    def get_stats(self) -> Dict[str, Any]:
        return self.inference_stats.copy()

    def reset_stats(self):
        self.inference_stats = {
            "total_inferences": 0,
            "task_type_counts": {},
            "lora_usage_counts": {},
            "average_confidence": 0.0
        }

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "available_loras": self.lora_selector.list_available_loras(),
            "task_mappings": self.lora_selector.get_task_mapping(),
            "current_lora": self.lora_manager.get_current_lora(),
            "cached_loras": self.lora_manager.list_cached_loras(),
            "model_memory_usage": self.lora_manager.get_model_memory_usage(),
            "inference_stats": self.get_stats()
        }

    def clear_cache(self):
        if self.enable_caching:
            self.lora_manager.clear_cache()
            if self.verbose:
                print("LoRA cache cleared")


def test_multi_lora_inference():
    print("Multi-LoRA Inference Engine Test (Simulated):")
    print("-" * 60)
    
    inference_engine = MultiLoRAInference(verbose=True)
    
    test_questions = [
        "How many buildings are in the image?",
        "Please describe the content of this remote sensing image.",
        "Where is the airplane in the image?",
        "What type of land cover is this area?"
    ]
    
    print("\nTask Prediction Test:")
    print("-" * 40)
    
    for question in test_questions:
        prediction = inference_engine.get_task_prediction(question)
        print(f"Question: {question}")
        print(f"Predicted Task: {prediction['predicted_task']}")
        print(f"Confidence: {prediction['confidence']:.3f}")
        print(f"Selected LoRA: {prediction['selected_lora']['lora_name']}")
        print("-" * 30)
    
    print("\nSystem Information:")
    print("-" * 40)
    system_info = inference_engine.get_system_info()
    print(f"Number of available LoRAs: {len(system_info['available_loras'])}")
    print(f"Task Mappings: {list(system_info['task_mappings'].keys())}")
    print(f"Current LoRA: {system_info['current_lora']}")
    
    print("\nNote: Full inference testing requires an actual base model and tokenizer.")


if __name__ == "__main__":
    test_multi_lora_inference()
