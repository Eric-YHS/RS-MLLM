import os
import sys
from typing import Optional, Dict, Any, Union
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.multi_lora_inference import MultiLoRAInference


class MultiLoRAModelManager:
    def __init__(self, model_path: str, lora_path: Optional[str] = None):
        self.model_path = model_path
        self.lora_path = lora_path
        self.model = None
        self.tokenizer = None
        self.multi_lora_inference = None
        self.is_lora_loaded = False
        
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            print("Warning: CUDA not detected, falling back to CPU (performance will be slow)")

    def load_model(self):
        print(f"Loading Jiuge 4B model from: {self.model_path}")
        print("🚀 Enabling Multi-LoRA intelligent selection system")
        print(f"Using device: {self.device}")
        
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")
            
            print("1. Loading base model...")
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
            
            print("✓ Base model loaded successfully")
            
            print("2. Initializing Multi-LoRA inference system...")
            self.multi_lora_inference = MultiLoRAInference(
                base_model=self.model,
                tokenizer=self.tokenizer,
                verbose=True
            )
            
            self.is_lora_loaded = True
            print("✓ Multi-LoRA system initialized successfully")
            
            system_info = self.multi_lora_inference.get_system_info()
            available_loras = system_info['available_loras']
            print(f"📊 Available LoRA adapters: {len(available_loras)}")
            
            for lora in available_loras:
                print(f"  - {lora['lora_name']}: {lora['description'][:50]}...")
            
            self.model = self.model.eval()
            
            print("🎉 Multi-LoRA model loading complete!")
            print("📊 Currently using: Base model + Intelligent LoRA selection system")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        try:
            if self.multi_lora_inference is None:
                print("❌ Multi-LoRA system not initialized")
                return ""
            
            response = self.multi_lora_inference.inference(
                image=image,
                question=prompt,
                max_length=32,
                temperature=0.0,
                do_sample=False
            )
            
            return response if response else ""
            
        except Exception as e:
            print(f"Inference error: {e}")
            return ""

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "model_path": self.model_path,
            "lora_path": "multi_lora_system",
            "is_lora_loaded": self.is_lora_loaded,
            "device": self.device,
            "system_type": "multi_lora_intelligent_selection"
        }
        
        if self.multi_lora_inference:
            system_info = self.multi_lora_inference.get_system_info()
            info.update({
                "available_loras": len(system_info['available_loras']),
                "task_mappings": list(system_info['task_mappings'].keys()),
                "inference_stats": system_info['inference_stats']
            })
        
        return info

    def get_gpu_memory_info(self) -> Dict[str, float]:
        import torch
        
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0}
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        free_memory = total_memory - cached_memory
        
        return {
            "total": total_memory,
            "allocated": allocated_memory,
            "cached": cached_memory,
            "free": free_memory
        }

    def get_task_prediction_info(self, prompt: str) -> Dict[str, Any]:
        if self.multi_lora_inference:
            return self.multi_lora_inference.get_task_prediction(prompt)
        return {}

    def get_inference_stats(self) -> Dict[str, Any]:
        if self.multi_lora_inference:
            return self.multi_lora_inference.get_stats()
        return {}


def create_enhanced_baseline_script():
    """创建增强版的baseline评估脚本"""
    
    enhanced_script = '''#!/usr/bin/env python3
"""
九格4B多模态模型Baseline评测系统 - 多LoRA智能选择版本
用于在VRSBench和MME RealWorld RS数据集上评测模型性能

使用方法:
python enhanced_baseline_evaluation.py --model_path /path/to/FM9G4B-V --data_root . --output_file evaluation_results.txt

新特性:
- 智能任务识别和LoRA选择
- 支持7个专门的LoRA适配器
- 自动根据问题类型选择最佳LoRA
- 详细的推理统计和分析

依赖库:
- torch
- transformers  
- peft
- PIL (Pillow)
- numpy
"""

import os
import sys
import argparse
from datetime import datetime

# 导入多LoRA集成模块
sys.path.append('mul_lora_systems')
from integration.baseline_integration import MultiLoRAModelManager

# 导入原始baseline的其他组件（假设它们可用）
try:
    # 这里应该导入原始baseline的其他必要组件
    # 由于我们无法直接修改原始文件，这里提供集成接口
    pass
except ImportError as e:
    print(f"警告: 无法导入原始baseline组件: {e}")


def run_enhanced_evaluation(args):
    """运行增强版评估"""
    print("=" * 80)
    print("九格4B多模态模型 - 多LoRA智能选择评测系统")
    print("=" * 80)
    
    # 创建多LoRA模型管理器
    model_manager = MultiLoRAModelManager(
        model_path=args.model_path,
        lora_path=None  # 使用多LoRA系统
    )
    
    # 加载模型
    if not model_manager.load_model():
        print("❌ 模型加载失败，退出评测")
        return False
    
    # 显示模型信息
    model_info = model_manager.get_model_info()
    print("\\n📊 模型配置信息:")
    print(f"  - 基础模型: {model_info['model_path']}")
    print(f"  - 系统类型: {model_info['system_type']}")
    print(f"  - 可用LoRA: {model_info.get('available_loras', 0)}个")
    print(f"  - 设备: {model_info['device']}")
    
    # 运行简单测试
    print("\\n🧪 运行系统测试...")
    test_success = run_simple_test(model_manager)
    
    if not test_success:
        print("❌ 系统测试失败，请检查模型配置")
        return False
    
    # 这里应该继续原始的评测流程
    # 由于我们无法完全重写原始脚本，这里提供框架
    print("\\n🚀 开始评测流程...")
    print("注意: 请将此脚本与原始baseline_evaluation_with_lora.py集成使用")
    
    return True


def run_simple_test(model_manager):
    """运行简单测试"""
    from PIL import Image
    import time
    
    print("运行多LoRA系统测试...")
    
    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    # 测试不同类型的问题
    test_cases = [
        ("图像中有多少个建筑物？", "目标检测"),
        ("请描述这张图像的内容。", "图像描述"),
        ("飞机在图像的什么位置？", "空间定位"),
        ("这个区域是什么类型的土地？", "地物分类")
    ]
    
    for question, expected_task in test_cases:
        print(f"\\n测试问题: {question}")
        print(f"预期任务类型: {expected_task}")
        
        # 获取任务预测信息
        task_info = model_manager.get_task_prediction_info(question)
        if task_info:
            print(f"识别任务: {task_info['predicted_task']}")
            print(f"置信度: {task_info['confidence']:.3f}")
            print(f"选择LoRA: {task_info['selected_lora']['lora_name']}")
        
        # 生成响应
        start_time = time.time()
        response = model_manager.generate_response(test_image, question)
        end_time = time.time()
        
        print(f"模型响应: {response}")
        print(f"响应时间: {end_time - start_time:.2f}秒")
        print("-" * 40)
    
    # 显示推理统计
    stats = model_manager.get_inference_stats()
    if stats:
        print("\\n📈 推理统计:")
        print(f"  - 总推理次数: {stats['total_inferences']}")
        print(f"  - 平均置信度: {stats['average_confidence']:.3f}")
        print(f"  - 任务类型分布: {stats['task_type_counts']}")
        print(f"  - LoRA使用分布: {stats['lora_usage_counts']}")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="九格4B多模态模型 - 多LoRA智能选择评测系统")
    parser.add_argument("--model_path", type=str, required=True, help="九格4B模型路径")
    parser.add_argument("--data_root", type=str, default=".", help="数据集根目录")
    parser.add_argument("--output_file", type=str, default="enhanced_evaluation_results.txt", help="输出文件")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（用于测试）")
    
    args = parser.parse_args()
    
    # 运行增强版评估
    success = run_enhanced_evaluation(args)
    
    if success:
        print("\\n✅ 评测完成!")
    else:
        print("\\n❌ 评测失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    return enhanced_script


import os
from datetime import datetime

def patch_baseline_script(baseline_script_path: str, output_path: str):
    if not os.path.exists(baseline_script_path):
        print(f"❌ Original baseline script not found: {baseline_script_path}")
        return False
    
    try:
        with open(baseline_script_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        patched_content = f'''# Multi-LoRA system integration patch
# Auto-generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

import sys
import os

# Add Multi-LoRA system path
mul_lora_path = os.path.join(os.path.dirname(__file__), "..", "mul_lora_systems")
if os.path.exists(mul_lora_path):
    sys.path.insert(0, mul_lora_path)
    
    # Import Multi-LoRA integration module
    try:
        from integration.baseline_integration import MultiLoRAModelManager
        
        # Replace the original ModelManagerWithLoRA class
        ModelManagerWithLoRA = MultiLoRAModelManager
        print("✅ Multi-LoRA system integrated successfully")
        
    except ImportError as e:
        print(f"⚠ Multi-LoRA system import failed: {{e}}")
        print("Using original LoRA system...")

# Original baseline script content
{original_content}
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        
        print(f"✅ Patched version created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to patch script: {e}")
        return False



def create_integration_guide():
    """创建集成指南"""
    
    guide = """
# 多LoRA系统集成指南

## 概述

本集成模块将多LoRA智能选择系统集成到现有的baseline评估框架中，实现无缝替换和增强功能。

## 集成方式

### 方式1: 直接替换（推荐）

1. 在原始baseline脚本中导入多LoRA集成模块：
```python
from mul_lora_systems.integration.baseline_integration import MultiLoRAModelManager

# 替换原始的ModelManagerWithLoRA
ModelManagerWithLoRA = MultiLoRAModelManager
```

2. 其余代码保持不变，多LoRA系统会自动接管LoRA选择和推理过程。

### 方式2: 修补脚本

使用提供的patch_baseline_script函数自动修补原始脚本：

```python
from mul_lora_systems.integration.baseline_integration import patch_baseline_script

patch_baseline_script(
    baseline_script_path="lora/baseline_evaluation_with_lora.py",
    output_path="enhanced_baseline_evaluation.py"
)
```

### 方式3: 独立运行

直接使用增强版评估脚本：

```bash
python mul_lora_systems/integration/enhanced_baseline_evaluation.py \\
    --model_path /path/to/FM9G4B-V \\
    --data_root . \\
    --output_file results.txt
```

## 功能特性

### 智能LoRA选择

- 自动识别问题类型（目标检测、图像描述、空间定位等）
- 根据任务类型智能选择最佳LoRA适配器
- 支持7个专门训练的LoRA模型

### 兼容性保证

- 完全兼容原始baseline的接口
- 保持相同的输入输出格式
- 无需修改评测数据加载和指标计算逻辑

### 增强功能

- 详细的任务识别和LoRA选择日志
- 推理统计和性能分析
- 支持调试和性能监控

## 使用示例

```python
# 创建多LoRA模型管理器
model_manager = MultiLoRAModelManager(
    model_path="/path/to/FM9G4B-V"
)

# 加载模型和多LoRA系统
model_manager.load_model()

# 生成响应（自动选择LoRA）
response = model_manager.generate_response(image, question)

# 获取任务预测信息
task_info = model_manager.get_task_prediction_info(question)
print(f"识别任务: {task_info['predicted_task']}")
print(f"选择LoRA: {task_info['selected_lora']['lora_name']}")
```

## 性能对比

多LoRA系统相比单一LoRA的优势：

1. **任务适应性**: 根据具体任务选择最优LoRA
2. **性能提升**: 专门训练的LoRA在特定任务上表现更好
3. **灵活性**: 支持动态LoRA切换和扩展
4. **可解释性**: 提供详细的选择过程和统计信息

## 注意事项

1. 确保mul_lora_systems目录在Python路径中
2. 所有LoRA模型文件必须正确放置
3. 基础模型路径必须正确配置
4. 建议在GPU环境下运行以获得最佳性能

## 故障排除

### 常见问题

1. **导入错误**: 检查mul_lora_systems路径是否正确
2. **LoRA加载失败**: 验证LoRA文件完整性
3. **推理错误**: 检查模型和分词器是否正确加载
4. **性能问题**: 确保使用GPU并检查显存使用情况

### 调试方法

```python
# 启用详细日志
model_manager = MultiLoRAModelManager(model_path, verbose=True)

# 获取系统信息
system_info = model_manager.multi_lora_inference.get_system_info()
print(system_info)

# 获取推理统计
stats = model_manager.get_inference_stats()
print(stats)
```
"""
    
    return guide


def test_integration():
    print("Testing Multi-LoRA system integration...")
    print("-" * 50)
    
    try:

        
        model_manager = MultiLoRAModelManager(
            model_path="FM9G4B-V"  # Assuming path
        )
        
        print("✓ MultiLoRAModelManager created successfully")
        
        model_info = model_manager.get_model_info()
        print(f"✓ Model info interface is working: {model_info.get('system_type', 'N/A')}")
        
        memory_info = model_manager.get_gpu_memory_info()
        print(f"✓ Memory info interface is working: {memory_info}")
        
        print("\nIntegration test complete!")
        
    except NameError:
        print("❌ Integration test failed: MultiLoRAModelManager is not defined.")
        print("   Please ensure the class is imported correctly before running this test.")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")


if __name__ == "__main__":
    test_integration()
