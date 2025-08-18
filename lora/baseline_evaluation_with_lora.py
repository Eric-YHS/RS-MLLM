#!/usr/bin/env python3
"""
九格4B多模态模型Baseline评测系统 - 支持LoRA适配器版本
用于在VRSBench和MME RealWorld RS数据集上评测模型性能

使用方法:
python baseline_evaluation_with_lora.py --model_path /path/to/FM9G4B-V --lora_path ./fm9g_simple_lora --data_root . --output_file evaluation_results.txt

新增参数:
--lora_path: LoRA适配器路径（可选）
--no_lora: 禁用LoRA，使用原始模型

依赖库:
- torch
- transformers
- peft
- PIL (Pillow)
- numpy
"""
import warnings
# 屏蔽PEFT库关于缺失键的警告
warnings.filterwarnings('ignore', category=UserWarning, message=".*Found missing adapter keys.*")

# 屏蔽transformers库关于生成参数的警告
warnings.filterwarnings('ignore', category=UserWarning, message=".*`do_sample` is set to `False`.*")

# 屏蔽transformers库关于image_processor_class的弃用警告
warnings.filterwarnings('ignore', category=FutureWarning, message=".*image_processor_class is deprecated.*")

# 屏蔽transformers库关于seen_tokens的弃用警告
# (这个警告可能来自不同模块，我们用更通用的方式)
warnings.filterwarnings('ignore', message=".*The `seen_tokens` attribute is deprecated.*")
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import re

# 修复相对导入问题
def fix_model_imports(model_path):
    """修复模型的相对导入问题"""
    abs_model_path = os.path.abspath(model_path)
    if abs_model_path not in sys.path:
        sys.path.insert(0, abs_model_path)
        print(f"已添加模型路径到Python路径: {abs_model_path}")

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import numpy as np

# 导入PEFT库用于LoRA
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
    print("✓ PEFT库可用，支持LoRA适配器")
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠ 未安装PEFT库，无法使用LoRA适配器")
    print("安装命令: pip install peft")

# ==========================================================
# === 多LoRA智能选择系统 集成补丁 ===
# ==========================================================
import warnings
# 屏蔽PEFT库在加载LoRA时关于"missing adapter keys"的特定UserWarning
warnings.filterwarnings('ignore', category=UserWarning, message=".*Found missing adapter keys.*")

# 动态添加 mul_lora_systems 目录到 Python 路径
mul_lora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mul_lora_systems'))
if mul_lora_path not in sys.path:
    sys.path.insert(0, mul_lora_path)
    print(f"✅ 已注入多LoRA系统路径: {mul_lora_path}")

try:
    # 导入我们新的、更智能的模型管理器
    from integration.baseline_integration import MultiLoRAModelManager
    
    # 魔法发生的地方：用新的管理器替换掉原来的那个！
    ModelManagerWithLoRA = MultiLoRAModelManager
    print("🚀 成功集成多LoRA智能选择系统！评测将使用智能切换模式。")
    
except ImportError as e:
    print(f"⚠️ 未能集成多LoRA系统，将回退到原始LoRA模式: {e}")
# ==========================================================


# 尝试导入tqdm，如果没有安装则提供简单替代
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("警告: 未安装tqdm库，使用简单进度显示")
    print("建议安装: pip install tqdm")
    TQDM_AVAILABLE = False
    
    # 简单的tqdm替代实现
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit='it', **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc or ""
            self.current = 0
            self.start_time = time.time()
            
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.update()
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            self.close()
            
        def update(self, n=1):
            self.current += n
            if self.current % 10 == 0 or self.current >= self.total:
                elapsed = time.time() - self.start_time
                rate = self.current / elapsed if elapsed > 0 else 0
                print(f"\r{self.desc}: {self.current}/{self.total} ({self.current/self.total*100:.1f}%) - {rate:.1f}it/s", end='', flush=True)
                
        def close(self):
            print()  # 换行
            
        def set_postfix(self, **kwargs):
            pass  # 简单实现，忽略postfix

# 使用自定义BLEU实现，避免NLTK依赖问题
print("使用自定义BLEU计算实现（避免NLTK依赖问题）")

class SmoothingFunction:
    @staticmethod
    def method1():
        return None

def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None):
    if not hypothesis:
        return 0.0
    
    if isinstance(hypothesis, str):
        hypothesis = hypothesis.split()
    
    ref_tokens_list = []
    for ref in references:
        if isinstance(ref, str):
            ref_tokens_list.append(ref.split())
        else:
            ref_tokens_list.append(ref)
    
    precisions = []
    for n in range(1, len(weights) + 1):
        if len(hypothesis) < n:
            precisions.append(0.0)
            continue
            
        hyp_ngrams = {}
        for i in range(len(hypothesis) - n + 1):
            ngram = tuple(hypothesis[i:i+n])
            hyp_ngrams[ngram] = hyp_ngrams.get(ngram, 0) + 1
        
        ref_ngrams = {}
        for ref_tokens in ref_tokens_list:
            if len(ref_tokens) >= n:
                for i in range(len(ref_tokens) - n + 1):
                    ngram = tuple(ref_tokens[i:i+n])
                    ref_ngrams[ngram] = max(ref_ngrams.get(ngram, 0), 
                                          ref_tokens.count(ngram) if n == 1 else 1)
        
        matches = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
            total += count
        
        if total == 0:
            precisions.append(1e-12)  # Add smoothing for 0
        else:
            precisions.append(matches / total)
    
    if any(p == 0 for p in precisions):
        return 0.0
    
    import math
    log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
    
    # Brevity Penalty
    hyp_len = len(hypothesis)
    ref_lens = [len(ref) for ref in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
    
    if hyp_len > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / hyp_len) if hyp_len > 0 else 0
        
    return bp * math.exp(log_sum)


@dataclass
class EvaluationResult:
    """评测结果数据结构"""
    task_name: str
    dataset_name: str
    total_samples: int
    successful_samples: int
    metrics: Dict[str, float]
    execution_time: float


class OriginalModelManagerWithLoRA:
    """九格4B模型管理器 - 支持LoRA适配器 (这是原始版本，如果补丁失败会使用这个)"""
    
    def __init__(self, model_path: str, lora_path: Optional[str] = None):
        self.model_path = model_path
        self.lora_path = lora_path
        self.model = None
        self.tokenizer = None
        self.is_lora_loaded = False
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            print("警告: PyTorch未检测到CUDA支持")
        
    def load_model(self):
        print(f"正在加载九格4B模型从: {self.model_path}")
        if self.lora_path:
            print(f"LoRA适配器路径: {self.lora_path}")
        print(f"使用设备: {self.device}")
        
        fix_model_imports(self.model_path)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB")
        
        try:
            print("1. 加载基础模型...")
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_cache=True,
            )
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                use_fast=True,
                padding_side='left'
            )
            
            print("✓ 基础模型加载成功")
            
            if self.lora_path and PEFT_AVAILABLE:
                print("2. 加载LoRA适配器...")
                
                if not os.path.exists(self.lora_path):
                    print(f"❌ LoRA路径不存在: {self.lora_path}")
                    return False
                
                try:
                    self.model = PeftModel.from_pretrained(
                        self.model, 
                        self.lora_path,
                        torch_dtype=torch.bfloat16
                    )
                    self.is_lora_loaded = True
                    print("✓ LoRA适配器加载成功")
                except Exception as lora_error:
                    print(f"❌ LoRA适配器加载失败: {lora_error}")
                    self.is_lora_loaded = False
            
            self.model = self.model.eval()
            
            print("🎉 模型加载完成!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def generate_response(self, image: Image.Image, prompt: str) -> str:
        try:
            msgs = [{'role': 'user', 'content': [image, prompt]}]
            
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                max_new_tokens=32,
                sampling=False
            )
            
            return response if response else ""
            
        except Exception as e:
            print(f"推理错误: {e}")
            return ""

# 如果补丁成功，ModelManagerWithLoRA 已经被替换为 MultiLoRAModelManager
# 如果补丁失败，这里会定义原始的类，然后代码会使用它
if "MultiLoRAModelManager" not in globals():
    ModelManagerWithLoRA = OriginalModelManagerWithLoRA

class DataLoaderWithLoRA:
    """数据加载器"""
    
    def __init__(self, data_root: str):
        self.data_root = os.path.abspath(data_root)
        self.vrsbench_path = os.path.join(self.data_root, "open_datasets", "datasets--xiang709--VRSBench")
        self.mme_path = os.path.join(self.data_root, "open_datasets", "datasets--yifanzhang114--MME-RealWorld")
        
        print(f"数据集根目录: {self.data_root}")
        print(f"VRSBench期望路径: {self.vrsbench_path}")
        print(f"MME期望路径: {self.mme_path}")
        
    def load_vrsbench_caption(self) -> List[Dict]:
        caption_file = os.path.join(self.vrsbench_path, "VRSBench_EVAL_Cap.json")
        if not os.path.exists(caption_file):
            print(f"错误: 无法找到VRSBench_EVAL_Cap.json文件在 {self.vrsbench_path}")
            return []
        with open(caption_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载VRSBench图像描述数据: {len(data)}个样本")
        return data
    
    def load_vrsbench_referring(self) -> List[Dict]:
        referring_file = os.path.join(self.vrsbench_path, "VRSBench_EVAL_referring.json")
        if not os.path.exists(referring_file):
            print(f"错误: 无法找到VRSBench_EVAL_referring.json文件在 {self.vrsbench_path}")
            return []
        with open(referring_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载VRSBench视觉定位数据: {len(data)}个样本")
        return data
    
    def load_vrsbench_vqa(self) -> List[Dict]:
        vqa_file = os.path.join(self.vrsbench_path, "VRSBench_EVAL_vqa.json")
        if not os.path.exists(vqa_file):
            print(f"错误: 无法找到VRSBench_EVAL_vqa.json文件在 {self.vrsbench_path}")
            return []
        with open(vqa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载VRSBench视觉问答数据: {len(data)}个样本")
        return data

    def load_mme_realworld(self) -> List[Dict]:
        mme_file = os.path.join(self.mme_path, "MME_RealWorld.json")
        if not os.path.exists(mme_file):
            print(f"错误: 无法找到MME_RealWorld.json文件在 {self.mme_path}")
            return []
        with open(mme_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载MME RealWorld数据: {len(data)}个样本")
        return data

    def get_image_path(self, dataset: str, image_id: str) -> str:
        if dataset == "vrsbench":
            return os.path.join(self.vrsbench_path, "Images_val", image_id)
        elif dataset == "mme":
            return os.path.join(self.mme_path, image_id)
        return ""

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            if os.path.exists(image_path):
                return Image.open(image_path).convert('RGB')
            else:
                # print(f"图像文件不存在: {image_path}") # Suppress for cleaner logs
                return None
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None
            
    def preprocess_data(self, data, dataset_type):
        processed = []
        # (Preprocessing logic as in original script)
        for item in data:
            if dataset_type == "vrsbench_caption":
                processed.append({ 'image_path': self.get_image_path("vrsbench", item['image_id']), **item })
            elif dataset_type == "vrsbench_referring":
                processed.append({ 'image_path': self.get_image_path("vrsbench", item['image_id']), **item })
            elif dataset_type == "vrsbench_vqa":
                processed.append({ 'image_path': self.get_image_path("vrsbench", item['image_id']), **item })
            elif dataset_type == "mme_realworld":
                processed.append({ 'image_path': self.get_image_path("mme", item['Image']), **item })
        return processed

class MetricComputerWithLoRA:
    @staticmethod
    def compute_bleu(predictions, references):
        bleu1_scores, bleu2_scores, bleu4_scores = [], [], []
        for pred, ref_list in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens_list = [ref.lower().split() for ref in ref_list]
            bleu1_scores.append(sentence_bleu(ref_tokens_list, pred_tokens, weights=(1, 0, 0, 0)))
            bleu2_scores.append(sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.5, 0.5, 0, 0)))
            bleu4_scores.append(sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25)))
        
        avg_bleu1 = np.mean(bleu1_scores)
        avg_bleu2 = np.mean(bleu2_scores)
        avg_bleu4 = np.mean(bleu4_scores)
        avg_bleu = (avg_bleu1 + avg_bleu2 + avg_bleu4) / 3
        
        return {'bleu1': avg_bleu1, 'bleu2': avg_bleu2, 'bleu4': avg_bleu4, 'bleu_avg': avg_bleu}

    @staticmethod
    def parse_coordinates(coord_str):
        match = re.search(r'\{<(\d+)><(\d+)><(\d+)><(\d+)>\}', coord_str)
        return tuple(map(int, match.groups())) if match else (0, 0, 0, 0)

    @staticmethod
    def compute_iou(box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def compute_accuracy(predictions, ground_truths):
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p.strip().lower() == g.strip().lower())
        return correct / len(predictions) if predictions else 0.0

    @staticmethod
    def compute_vrsbench_score(caption_score, referring_score, vqa_score):
        return caption_score + referring_score + vqa_score

    @staticmethod
    def compute_mme_score(accuracy):
        return accuracy * 100

    @staticmethod
    def compute_final_score(vrsbench_score, mme_score):
        return (vrsbench_score + mme_score) / 2

class CaptionEvaluatorWithLoRA:
    def __init__(self, model_manager, data_loader):
        self.model_manager = model_manager
        self.data_loader = data_loader
        self.metric_computer = MetricComputerWithLoRA()
    
    def evaluate(self, data):
        start_time = time.time()
        predictions, references, successful_samples = [], [], 0
        
        for item in tqdm(data, desc="图像描述"):
            image = self.data_loader.load_image(item['image_path'])
            if image:
                prediction = self.model_manager.generate_response(image, item['question'])
                if prediction.strip():
                    successful_samples += 1
                predictions.append(prediction)
                references.append(item['ground_truth'])
        
        bleu_metrics = self.metric_computer.compute_bleu(predictions, references)
        task_score = bleu_metrics['bleu_avg'] * 25
        
        return EvaluationResult("Image Caption", "VRSBench", len(data), successful_samples, {'bleu_avg': bleu_metrics['bleu_avg'], 'task_score': task_score}, time.time() - start_time)

class ReferringEvaluatorWithLoRA:
    def __init__(self, model_manager, data_loader):
        self.model_manager = model_manager
        self.data_loader = data_loader
        self.metric_computer = MetricComputerWithLoRA()

    def evaluate(self, data):
        start_time = time.time()
        successful_localizations, successful_samples = 0, 0

        for item in tqdm(data, desc="视觉定位"):
            image = self.data_loader.load_image(item['image_path'])
            if image:
                prompt = f"Please locate the following object: {item['question']}"
                prediction = self.model_manager.generate_response(image, prompt)
                if prediction.strip():
                    successful_samples += 1
                    pred_coords = self.metric_computer.parse_coordinates(prediction)
                    gt_coords = item['obj_corner']
                    gt_coords_rect = (gt_coords[0], gt_coords[1], gt_coords[2], gt_coords[3])
                    if self.metric_computer.compute_iou(pred_coords, gt_coords_rect) >= 0.5:
                        successful_localizations += 1
        
        accuracy = successful_localizations / len(data) if data else 0.0
        task_score = accuracy * 25

        return EvaluationResult("Visual Grounding", "VRSBench", len(data), successful_samples, {'localization_accuracy': accuracy, 'task_score': task_score}, time.time() - start_time)


class VQAEvaluatorWithLoRA:
    def __init__(self, model_manager, data_loader):
        self.model_manager = model_manager
        self.data_loader = data_loader
        self.metric_computer = MetricComputerWithLoRA()

    def _extract_choice(self, response):
        match = re.search(r'\b([A-E])\b', response.strip().upper())
        return match.group(1) if match else ""

    def evaluate_vrsbench_vqa(self, data):
        start_time = time.time()
        predictions, ground_truths, successful_samples = [], [], 0
        for item in tqdm(data, desc="VRSBench问答"):
            image = self.data_loader.load_image(item['image_path'])
            if image:
                prediction = self.model_manager.generate_response(image, item['question'])
                if prediction.strip():
                    successful_samples += 1
                predictions.append(prediction)
                ground_truths.append(item['ground_truth'])
        
        accuracy = self.metric_computer.compute_accuracy(predictions, ground_truths)
        task_score = accuracy * 50
        return EvaluationResult("VQA", "VRSBench", len(data), successful_samples, {'accuracy': accuracy, 'task_score': task_score}, time.time() - start_time)

    def evaluate_mme_vqa(self, data):
        start_time = time.time()
        predictions, ground_truths, successful_samples = [], [], 0
        for item in tqdm(data, desc="MME问答"):
            image = self.data_loader.load_image(item['image_path'])
            if image:
                choices_text = "\n".join(item['Answer choices'])
                prompt = f"{item['Text']}\nChoices:\n{choices_text}\nPlease select the correct answer (A, B, C, D, or E):"
                prediction = self.model_manager.generate_response(image, prompt)
                if prediction.strip():
                    successful_samples += 1
                predictions.append(self._extract_choice(prediction))
                ground_truths.append(item['Ground truth'])
                
        accuracy = self.metric_computer.compute_accuracy(predictions, ground_truths)
        task_score = accuracy * 100
        return EvaluationResult("VQA", "MME RealWorld RS", len(data), successful_samples, {'accuracy': accuracy, 'task_score': task_score}, time.time() - start_time)

class ResultRecorderWithLoRA:
    def __init__(self, output_file):
        self.output_file = output_file
        self.start_time = datetime.now()

    def generate_report(self, vrsbench_results, mme_result, final_scores, model_info):
        report = []
        report.append("="*80)
        report.append("九格4B多模态模型评测报告")
        report.append(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80 + "\n")
        
        report.append("--- 模型配置 ---")
        for key, value in model_info.items():
            report.append(f"{key}: {value}")
        report.append("\n--- VRSBench 评测结果 ---")
        report.append(f"图像描述 (X1*25): {vrsbench_results['caption'].metrics['task_score']:.2f}")
        report.append(f"视觉定位 (X2*25): {vrsbench_results['referring'].metrics['task_score']:.2f}")
        report.append(f"视觉问答 (X3*50): {vrsbench_results['vqa'].metrics['task_score']:.2f}")
        report.append(f"VRSBench总分 (S1): {final_scores['vrsbench_score']:.2f}\n")
        
        report.append("--- MME RealWorld RS 评测结果 ---")
        report.append(f"视觉问答准确率 (X1): {mme_result.metrics['accuracy']:.4f}")
        report.append(f"MME总分 (S2): {final_scores['mme_score']:.2f}\n")
        
        report.append("--- 综合得分 ---")
        report.append(f"开源评测集综合得分 S = (S1 + S2) / 2: {final_scores['final_score']:.2f}")
        
        return "\n".join(report)

    def save_report(self, report_content):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n评测报告已保存到: {self.output_file}")

    def print_summary(self, final_scores):
        #print("\n" + "="*60 + "\n评测总结\n" + "="*60)
        #print(f"VRSBench总分: {final_scores['vrsbench_score']:.2f}/100")
        #print(f"MME RealWorld RS总分: {final_scores['mme_score']:.2f}/100")
        #print(f"综合得分: {final_scores['final_score']:.2f}/100")
        print()

def main():
    parser = argparse.ArgumentParser(description="九格4B模型Baseline评测系统 - 支持LoRA")
    parser.add_argument("--model_path", type=str, default=".", help="九格4B模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA适配器路径（可选）")
    parser.add_argument("--no_lora", action="store_true", help="禁用LoRA")
    parser.add_argument("--data_root", type=str, default=".", help="数据集根目录")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt", help="输出文件")

    args = parser.parse_args()
    
    # 当使用多LoRA系统时，--lora_path被内部管理，但--no_lora可以用来强制使用基础模型
    # MultiLoRAModelManager 会忽略 lora_path 参数
    if "MultiLoRAModelManager" in globals() and not args.no_lora:
        print("多LoRA系统已集成，将忽略 --lora_path 参数，并自动选择LoRA。")
        args.lora_path = "multi_lora_system" # 标记为多LoRA系统
    elif args.no_lora:
        args.lora_path = None
        #print("🚫 已禁用LoRA，将仅使用基础模型进行评测。")

    model_manager = ModelManagerWithLoRA(args.model_path, args.lora_path)
    if not model_manager.load_model():
        return

    data_loader = DataLoaderWithLoRA(args.data_root)
    metric_computer = MetricComputerWithLoRA()
    result_recorder = ResultRecorderWithLoRA(args.output_file)
    
    vrsbench_results = {}
    
    # VRSBench
    vrs_caption_data = data_loader.preprocess_data(data_loader.load_vrsbench_caption(), "vrsbench_caption")
    vrsbench_results["caption"] = CaptionEvaluatorWithLoRA(model_manager, data_loader).evaluate(vrs_caption_data)

    vrs_referring_data = data_loader.preprocess_data(data_loader.load_vrsbench_referring(), "vrsbench_referring")
    vrsbench_results["referring"] = ReferringEvaluatorWithLoRA(model_manager, data_loader).evaluate(vrs_referring_data)
    
    vrs_vqa_data = data_loader.preprocess_data(data_loader.load_vrsbench_vqa(), "vrsbench_vqa")
    vrsbench_results["vqa"] = VQAEvaluatorWithLoRA(model_manager, data_loader).evaluate_vrsbench_vqa(vrs_vqa_data)

    # MME
    mme_data = data_loader.preprocess_data(data_loader.load_mme_realworld(), "mme_realworld")
    mme_result = VQAEvaluatorWithLoRA(model_manager, data_loader).evaluate_mme_vqa(mme_data)
    
    # Final scores
    vrsbench_score = metric_computer.compute_vrsbench_score(
        vrsbench_results["caption"].metrics['task_score'],
        vrsbench_results["referring"].metrics['task_score'],
        vrsbench_results["vqa"].metrics['task_score']
    )
    mme_score = mme_result.metrics['task_score']
    final_score = metric_computer.compute_final_score(vrsbench_score, mme_score)
    
    final_scores = {'vrsbench_score': vrsbench_score, 'mme_score': mme_score, 'final_score': final_score}

    report = result_recorder.generate_report(vrsbench_results, mme_result, final_scores, model_manager.get_model_info())
    result_recorder.save_report(report)
    #result_recorder.print_summary(final_scores)

if __name__ == "__main__":
    main()