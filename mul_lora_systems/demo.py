#!/usr/bin/env python3
import os
import sys
import argparse
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message=".*Found missing adapter keys.*")
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.multi_lora_inference import MultiLoRAInference
from core.task_classifier import TaskClassifier
from core.lora_selector import LoRASelector
from core.lora_manager import LoRAManager


class MultiLoRADemo:
    def __init__(self, model_path: str, verbose: bool = True):
        self.model_path = model_path
        self.verbose = verbose
        self.inference_engine = None
        self.model_loaded = False
        
    def load_model(self) -> bool:
        print("🚀 Initializing Multi-LoRA Smart Selection System")
        print("=" * 60)
        
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model path does not exist: {self.model_path}")
                return False
            
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            print(f"📁 Model Path: {self.model_path}")
            
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🖥️  Device: {device} ({gpu_name}, {gpu_memory:.1f}GB)")
            else:
                device = "cpu"
                print(f"🖥️  Device: {device} (Warning: Performance will be slower)")
            
            print("\n1️⃣  Loading base model...")
            base_model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            if device == "cuda":
                base_model = base_model.cuda()
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            print("✅ Base model loaded successfully")
            
            print("\n2️⃣  Initializing Multi-LoRA inference system...")
            self.inference_engine = MultiLoRAInference(
                base_model=base_model,
                tokenizer=tokenizer,
                verbose=self.verbose
            )
            
            system_info = self.inference_engine.get_system_info()
            available_loras = system_info['available_loras']
            
            print(f"✅ Multi-LoRA system initialized successfully")
            print(f"\n📊 System Configuration:")
            print(f"   - Available LoRA adapters: {len(available_loras)}")
            print(f"   - Task Mappings: {len(system_info['task_mappings'])} task types")
            
            print(f"\n🎯 Available LoRA Adapters:")
            for i, lora in enumerate(available_loras, 1):
                print(f"   {i}. {lora['lora_name']}")
                print(f"       Task: {lora.get('task_type', 'unknown')}")
                print(f"       Training Method: {lora.get('training_method', 'unknown')}")
                print(f"       Description: {lora['description'][:80]}...")
                print()
            
            self.model_loaded = True
            print("🎉 System loaded, ready for inference!")
            
            return True
            
        except Exception as e:
            print(f"❌ System loading failed: {e}")
            return False
    
    def run_single_inference(self, image_path: str, question: str) -> Dict[str, Any]:
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        print("\n" + "=" * 60)
        print("🔍 Starting Inference Analysis")
        print("=" * 60)
        
        try:
            print(f"📷 Loading image: {image_path}")
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            image = Image.open(image_path).convert('RGB')
            print(f"   Image size: {image.size}")
            
            print(f"\n❓ Question: {question}")
            
            print(f"\n🧠 Task Analysis:")
            task_info = self.inference_engine.get_task_prediction(question)
            
            print(f"   Identified Task: {task_info['predicted_task']}")
            print(f"   Confidence: {task_info['confidence']:.3f}")
            print(f"   Selected LoRA: {task_info['selected_lora']['lora_name']}")
            print(f"   LoRA Description: {task_info['selected_lora']['description'][:60]}...")
            
            if self.verbose:
                print(f"\n📈 All Task Confidences:")
                all_confidences = task_info['all_task_confidences']
                for task, conf in sorted(all_confidences.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {task}: {conf:.3f}")
            
            print(f"\n⚡ Executing inference...")
            start_time = time.time()
            
            response = self.inference_engine.inference(
                image=image,
                question=question,
                max_length=128,
                temperature=0.7,
                do_sample=True
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            print(f"\n💬 Model Response:")
            print(f"   {response}")
            print(f"\n⏱️  Inference Time: {inference_time:.2f} seconds")
            
            result = {
                "question": question,
                "response": response,
                "task_info": task_info,
                "inference_time": inference_time,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error during inference: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def run_batch_demo(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if not self.model_loaded:
            print("❌ Model not loaded")
            return []
        
        print(f"\n🚀 Starting batch inference demo ({len(test_cases)} test cases)")
        print("=" * 80)
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test Case {i}/{len(test_cases)}")
            print("-" * 40)
            
            result = self.run_single_inference(
                test_case['image_path'],
                test_case['question']
            )
            
            results.append(result)
            
            time.sleep(0.5)
        
        return results
    
    def show_system_statistics(self):
        if not self.model_loaded:
            print("❌ Model not loaded")
            return
        
        print("\n" + "=" * 60)
        print("📊 System Statistics")
        print("=" * 60)
        
        stats = self.inference_engine.get_stats()
        
        print(f"Total Inferences: {stats['total_inferences']}")
        print(f"Average Confidence: {stats['average_confidence']:.3f}")
        
        if stats['task_type_counts']:
            print(f"\nTask Type Distribution:")
            for task, count in stats['task_type_counts'].items():
                percentage = count / stats['total_inferences'] * 100 if stats['total_inferences'] > 0 else 0
                print(f"   {task}: {count} times ({percentage:.1f}%)")
        
        if stats['lora_usage_counts']:
            print(f"\nLoRA Usage Distribution:")
            for lora, count in stats['lora_usage_counts'].items():
                percentage = count / stats['total_inferences'] * 100 if stats['total_inferences'] > 0 else 0
                print(f"   {lora}: {count} times ({percentage:.1f}%)")
        
        system_info = self.inference_engine.get_system_info()
        memory_info = system_info.get('model_memory_usage', {})
        
        if memory_info and 'total_parameters' in memory_info:
            print(f"\nModel Parameter Information:")
            print(f"   Total Parameters: {memory_info['total_parameters']:,}")
            print(f"   Trainable Parameters: {memory_info['trainable_parameters']:,}")
            print(f"   Trainable Percentage: {memory_info['trainable_percentage']:.2f}%")
        
        if 'memory_info' in memory_info and memory_info['memory_info']:
            gpu_info = memory_info['memory_info']
            print(f"\nGPU Memory Usage:")
            print(f"   Allocated: {gpu_info['gpu_memory_allocated_mb']:.1f}MB")
            print(f"   Reserved: {gpu_info['gpu_memory_reserved_mb']:.1f}MB")
    
    def run_interactive_demo(self):
        if not self.model_loaded:
            print("❌ Model not loaded")
            return
        
        print("\n" + "=" * 60)
        print("🎮 Interactive Demo Mode")
        print("=" * 60)
        print("Enter 'quit' or 'exit' to quit")
        print("Enter 'stats' to view statistics")
        print("Enter 'help' to see help")
        
        while True:
            try:
                print("\n" + "-" * 40)
                
                image_path = input("📷 Please enter the image path: ").strip()
                
                if image_path.lower() in ['quit', 'exit']:
                    break
                elif image_path.lower() == 'stats':
                    self.show_system_statistics()
                    continue
                elif image_path.lower() == 'help':
                    print("Help Information:")
                    print("   - Enter the path to an image file")
                    print("   - Enter a question text")
                    print("   - The system will automatically select the best LoRA and generate a response")
                    continue
                
                if not image_path:
                    continue
                
                question = input("❓ Please enter your question: ").strip()
                
                if not question:
                    continue
                
                result = self.run_single_inference(image_path, question)
                
                if 'error' in result:
                    print(f"❌ {result['error']}")
            
            except KeyboardInterrupt:
                print("\n\n👋 User interrupted, exiting demo")
                break
            except Exception as e:
                print(f"❌ Error during interactive session: {e}")
        
        print("\n📊 Final Statistics:")
        self.show_system_statistics()

def create_sample_test_cases() -> List[Dict[str, str]]:
    sample_image_path = "sample_image.jpg"
    if not os.path.exists(sample_image_path):
        sample_image = Image.new('RGB', (512, 512), color='lightblue')
        sample_image.save(sample_image_path)
        print(f"✅ Created sample image: {sample_image_path}")
    
    test_cases = [
        {"image_path": sample_image_path, "question": "How many buildings are in the image?"},
        {"image_path": sample_image_path, "question": "Please describe the content of this remote sensing image."},
        {"image_path": sample_image_path, "question": "Where is the airplane in the image?"},
        {"image_path": sample_image_path, "question": "What type of land cover is this area?"},
        {"image_path": sample_image_path, "question": "Analyze the future development trend of this area."},
        {"image_path": sample_image_path, "question": "Analyze this image by combining multimodal information."}
    ]
    
    return test_cases

def main():
    parser = argparse.ArgumentParser(description="Multi-LoRA System Demo Script")
    parser.add_argument("--model_path", type=str, default="FM9G4B-V", 
                        help="Path to the 9G-4B model")
    parser.add_argument("--image_path", type=str, 
                        help="Image path (for single inference mode)")
    parser.add_argument("--question", type=str, 
                        help="Question text (for single inference mode)")
    parser.add_argument("--mode", type=str, default="auto", 
                        choices=["single", "batch", "interactive", "auto"],
                        help="Run mode")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Show detailed information")
    
    args = parser.parse_args()
    
    print("🎯 Multi-LoRA Smart Selection System Demo")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    demo = MultiLoRADemo(
        model_path=args.model_path,
        verbose=args.verbose
    )
    
    if not demo.load_model():
        print("❌ Model loading failed, exiting demo")
        sys.exit(1)
    
    if args.mode == "single" and args.image_path and args.question:
        result = demo.run_single_inference(args.image_path, args.question)
        if 'error' not in result:
            print("\n✅ Inference complete!")
            
    elif args.mode == "batch":
        test_cases = create_sample_test_cases()
        results = demo.run_batch_demo(test_cases)
        print(f"\n✅ Batch demo complete! Processed {len(results)} test cases")
        
    elif args.mode == "interactive":
        demo.run_interactive_demo()
        
    else:
        if args.image_path and args.question:
            result = demo.run_single_inference(args.image_path, args.question)
            if 'error' not in result:
                print("\n✅ Inference complete!")
        else:
            print("\n🎮 No specific parameters provided, running batch demo...")
            test_cases = create_sample_test_cases()
            results = demo.run_batch_demo(test_cases)
            print(f"\n✅ Batch demo complete! Processed {len(results)} test cases")
    
    demo.show_system_statistics()
    
    print(f"\n🎉 Demo finished! Thank you for using the Multi-LoRA Smart Selection System")

if __name__ == "__main__":
    main()