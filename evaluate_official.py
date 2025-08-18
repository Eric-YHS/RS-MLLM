#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

import torch
from PIL import Image
from tqdm import tqdm

try:
    from mul_lora_systems.integration.baseline_integration import MultiLoRAModelManager
    print("✅ Successfully imported the Multi-LoRA Smart Selection System Manager.")
except ImportError:
    print("❌ Error: Multi-LoRA system not found. Please ensure this script is run from the 'neww/' root directory.")
    sys.exit(1)

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def load_task_data(task_file_path: str) -> Optional[List[Dict]]:
    if not os.path.exists(task_file_path):
        logger.warning(f"Task file not found, skipping: {task_file_path}")
        return None
    try:
        with open(task_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i, sample in enumerate(data):
            question_id = sample.get("Question id", f"sample_{i}")
            clean_id = question_id.replace("/", "_")
            sample['unique_id'] = f"{os.path.splitext(os.path.basename(task_file_path))[0]}_{clean_id}"
        return data
    except Exception as e:
        logger.error(f"Failed to load or process task file {task_file_path}: {e}")
        return None

def _extract_choice(response: str) -> str:
    response = response.strip().upper()
    explicit_match = re.search(r'(?:ANSWER IS|ANSWER:|OPTION)\s*([A-Z])', response)
    if explicit_match: return explicit_match.group(1)
    start_match = re.match(r'^([A-Z])[\.\s\)]', response)
    if start_match: return start_match.group(1)
    word_match = re.search(r'\b([A-Z])\b', response)
    if word_match: return word_match.group(1)
    return ""


def run_evaluation(model_manager: MultiLoRAModelManager, test_set_path: str):
    logger.info("="*80)
    logger.info("🚀 Starting the official evaluation process...")
    logger.info(f"Test set path: {test_set_path}")
    
    task_files = ["en_caption.json", "zh_caption.json", "en_mcq.json", "zh_mcq.json"]
    all_predictions = {"caption_predictions": [], "vqa_predictions": []}

    for task_filename in task_files:
        task_path = os.path.join(test_set_path, task_filename)
        task_data = load_task_data(task_path)
        
        if not task_data: continue

        task_type = "caption" if "caption" in task_filename else "vqa"
        logger.info(f"\n--- Processing task: {task_filename} ({len(task_data)} samples) ---")
        
        progress_bar = tqdm(total=len(task_data), desc=f"Evaluating {task_filename}", unit="sample")
        
        for sample in task_data:
            unique_id = sample.get("unique_id")
            
            image_relative_path = sample.get("Image") or sample.get("Image1") or sample.get("image") or sample.get("img_path")
            question = sample.get("Text") or sample.get("question") or sample.get("prompt")
            
            if not image_relative_path or not question:
                logger.warning(f"Skipping sample {unique_id}, image or question field not found. Sample content: {str(sample)[:150]}...")
                progress_bar.update(1)
                continue

            image_full_path = os.path.join(test_set_path, "images", image_relative_path)
            
            try:
                image = Image.open(image_full_path).convert("RGB")
            except FileNotFoundError:
                progress_bar.update(1)
                continue
            except Exception as e:
                logger.warning(f"Skipping sample {unique_id}, failed to load image: {e}")
                progress_bar.update(1)
                continue

            if task_type == "vqa":
                choices = sample.get("Answer choices") or sample.get("choices") or sample.get("options")
                if choices and isinstance(choices, list):
                    choices_text = "\n".join(choices)
                    prompt = f"{question}\n\nOptions:\n{choices_text}\n\nPlease choose the correct option:"
                else:
                    prompt = question
            else:
                prompt = question
                
            model_response = model_manager.generate_response(image, prompt)
            
            if task_type == "vqa":
                final_answer = _extract_choice(model_response)
                all_predictions["vqa_predictions"].append({"id": unique_id, "answer": final_answer})
            else:
                all_predictions["caption_predictions"].append({"id": unique_id, "caption": model_response})

            progress_bar.update(1)
        
        progress_bar.close()

    return all_predictions


def save_predictions(predictions: Dict, output_file: str):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        logger.info("="*80)
        logger.info(f"✅ Evaluation complete! Predictions have been saved to: {output_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save predictions: {e}")


def main():
    parser = argparse.ArgumentParser(description="Official Competition One-Click Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model (e.g., '.')")
    parser.add_argument("--test_set_path", type=str, required=True, help="Path to the official test set folder (e.g., './valid')")
    parser.add_argument("--output_file", type=str, default="predictions.json", help="Output prediction file name")
    
    args = parser.parse_args()

    global logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    model_manager = MultiLoRAModelManager(model_path=args.model_path, lora_path=None)
    
    if not model_manager.load_model():
        logger.error("❌ Model loading failed, terminating evaluation.")
        return

    predictions = run_evaluation(model_manager, args.test_set_path)

    if predictions:
        save_predictions(predictions, args.output_file)
    else:
        logger.error("❌ No predictions were generated during the evaluation process.")

if __name__ == "__main__":
    main()