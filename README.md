# **"NineGrids" Multimodal Large Model Remote Sensing Analysis System - Operation Guide**

![CI](https://img.shields.io/github/actions/workflow/status/Eric-YHS/RS-MLLM/ci.yml?branch=master&logo=githubactions&logoColor=white&label=CI)
![License](https://img.shields.io/badge/code-MIT-orange)

## **System Demo**

<div align="center">
  <img src="gif/1.gif" alt="System Demo 1" width="400"/>
  <img src="gif/2.gif" alt="System Demo 2" width="400"/>
</div>

### **1. System Overview**
This system is a remote sensing image analysis solution based on Qiyuan's "NineGrids" 4B multimodal large model. Its core feature is the integration of a **Multi-LoRA Intelligent Selection System**, which can automatically select and load the optimal LoRA (Low-Rank Adaptation) adapter for inference based on the type of user query, thereby achieving the best performance on different remote sensing tasks.

This document will provide detailed instructions for evaluators on how to configure the environment, prepare data, and run the system's core functionalities, including function demonstrations, open-source dataset evaluation, and training procedures.

We offer a complete, powerful, and easy-to-use remote sensing analysis solution. We kindly ask evaluators to first experience the system's intelligent LoRA selection feature through demo.py.

```plain
python mul_lora_systems/demo.py --model_path . --image_path 1.jpg --question "Please describe what is in this picture?"
```

Then run

```plain
python evaluate_official.py \ --model_path . \ --test_set_path ./valid(test) \ --output_file predictions.json
```

Test the performance on a closed-source dataset.

---

### **2. Environment Configuration**
#### **2.1 Hardware Requirements**
+ **Inference/Evaluation**: NVIDIA A100/A800/H100/H800 series GPUs are recommended.
+ **Training**: The Text-to-LoRA training script is built to run on an **NVIDIA A100 80GB**.

#### **2.2 Software Requirements**
+ Operating System: Linux (CentOS 7.x or Ubuntu 20.04+ is recommended)
+ Python: 3.10.x
+ CUDA: 11.8+
+ Dependencies: See requirements.txt in the project's root directory.

#### **2.3 Installation Steps**
+ **Create and Activate Conda Environment**

```plain
# Create a conda environment with python 3.10
conda create -n fm9g4bv python=3.10

# Activate the environment
conda activate fm9g4bv
```

+ **Install Dependencies**

```plain
# Change to the project's root directory
cd /path/to/your/project/

# Install dependencies from requirements.txt using pip
pip install -r requirements.txt
```

---

### **3. Model and Data Preparation**
+ **Base Model**: Please place the official "NineGrids" 4B model files in the project's root directory. Ensure the following file and directory structure is correct:

```plain
./
├── modeling_fm9g.py
├── modeling_fm9gv.py
├── config.json
├── pytorch_model.bin
├── ... (other model files)
└── mul_lora_systems/
└── lora/
```

+ **Open-Source Datasets**: Please download the VRSBench and MME RealWorld RS datasets from the link provided by the organizers and decompress them into the open_datasets directory, resulting in the following structure:

```plain
./
├── open_datasets/
│   ├── datasets--xiang709--VRSBench/
│   │   ├── Images_val/
│   │   ├── VRSBench_EVAL_Cap.json
│   │   └── ...
│   └── datasets--yifanzhang114--MME-RealWorld/
│       ├── MME_RealWorld.json
│       └── ...
```

---

### **4. Core Functionality Guide**
#### **4.1 Multi-LoRA System Functionality Demo**
This command is used to demonstrate the system's core capability: intelligently selecting and performing inference with a LoRA based on the question.

+ **Command Format**:

```plain
python mul_lora_systems/demo.py --model_path . --image_path <image_path> --question "<your_question>"
```

+ **Example**:

```plain
python mul_lora_systems/demo.py --model_path . --image_path 1.jpg --question "Please describe what is in this picture?"
```

+ **Expected Output**: 
The system will output a detailed inference analysis process, including the task identification result, the selected LoRA, the confidence score, and the final model response.

```plain
🎯 Multi-LoRA Smart Selection System Demo
================================================================================
...
🚀 Initializing Multi-LoRA Smart Selection System
...
🧠 Task Analysis:
   Identified Task: image_caption
   Confidence: 1.855
   Selected LoRA: image_caption_lora
   LoRA Description: Image Description and Semantic Generation Capability...
...
⚡ Executing inference...
...
💬 Model Response:
   This image shows a modern residential building...
...
```

#### **4.2 Open-Source Dataset Evaluation**
This command uses an evaluation script integrated with the Multi-LoRA intelligent selection system to perform comprehensive evaluation on VRSBench and MME RealWorld RS datasets.

+ **Command Format**:

```plain
python -m lora.baseline_evaluation_with_lora --model_path . --data_root . --output_file <output_report_path>
```

+ **Example**:

```plain
python -m lora.baseline_evaluation_with_lora --model_path . --data_root . --output_file multi_lora_evaluation_results.txt
```

    - **Note**: The --lora_path parameter is ignored in this mode as the system automatically selects LoRA. To evaluate base model performance without any LoRA, add the --no_lora parameter.
+ **Expected Behavior**:  
The script will sequentially evaluate the three subtasks of VRSBench and the MME dataset. During the evaluation of each task, the system will dynamically load and switch LoRA based on different questions. After evaluation, a detailed performance report will be generated in the specified output file (multi_lora_evaluation_results.txt).

#### **4.3 Official Validation/Test Set Evaluation**
This command is used to run the model on the official validation set or final closed-source test set and generate prediction result files.

+ **Command Format**:

```plain
python evaluate_official.py --model_path . --test_set_path <validation/test_set_path> --output_file <prediction_output_path>
```

+ **Example (assuming validation set is in ./valid directory)**:

```plain
python evaluate_official.py --model_path . --test_set_path ./valid --output_file predictions.json
```

+ **Expected Behavior**:  
The script will read test data from the specified path, use our Multi-LoRA system for inference, and save all prediction results in the format required by officials to the predictions.json file for evaluation submission.

---

### **5. Model Training Guide**
#### **5.1 Text-to-LoRA Training**
This script demonstrates the process of generating 6 specialized LoRAs using Text-to-LoRA technology.

+ **Command Format**:

```plain
python mul_lora_systems/training/text_to_lora_trainer.py --base_model . --output_dir mul_lora_systems/loras/text_to_lora
```

#### **5.2 Dataset Fine-tuning**
This script fine-tunes a general-purpose LoRA based on MMRS-1M and ChatEarthNet datasets.

+ **Command Format**:

```plain
python mul_lora_systems/training/dataset_finetuner.py --base_model . --output_dir mul_lora_systems/loras/dataset_finetuned/mmrs_chatearth_lora
```

---

### **6. Summary**
We provide a complete, powerful, and easy-to-use remote sensing analysis solution. We kindly ask evaluators to first experience the system's intelligent LoRA selection feature through demo.py

```plain
python mul_lora_systems/demo.py --model_path . --image_path 1.jpg --question "Please describe what is in this picture?"
```

Then run

```plain
python evaluate_official.py \ --model_path . \ --test_set_path ./valid(test) \ --output_file predictions.json
```

Test the performance on a closed-source dataset.


---

### **7. Repository Layout**

```plain
.
├── mul_lora_systems/          # Multi-LoRA system (this project's main contribution)
│   ├── configs/               # lora_config.json / task_mapping.json / training_config.json
│   ├── core/                  # lora_manager, lora_selector, task_classifier, multi_lora_inference
│   ├── integration/           # baseline integration
│   ├── training/              # text_to_lora_trainer.py, dataset_finetuner.py, data_formats.py
│   └── demo.py                # end-to-end demo entry point
├── evaluate_official.py       # benchmark evaluation entry point (VRSBench / MME-RealWorld RS)
├── modeling_fm9g*.py, tokenization_*.py, resampler.py   # NineGrids remote code (upstream, see LICENSE)
├── YAOGAN_CHAT-main/          # demo web system: React front-end + FastAPI/Celery back-end
├── Dockerfile                 # inference environment image (weights are NOT baked in)
└── requirements.txt           # torch 2.3.0 + transformers 4.44.2 + peft + vllm
```

The web system is started from its own compose file:

```bash
cd YAOGAN_CHAT-main
docker compose up --build     # backend :8000, frontend :3000, redis, plus a Celery worker
```

The root `Dockerfile` only builds the Python/GPU environment; model weights are not stored in
Git and are not copied into the image, so the weight directory has to be mounted at run time
(see the comments inside the Dockerfile).

### **8. Licence**

The Multi-LoRA system and evaluation scripts (`mul_lora_systems/`, `evaluate_official.py`,
`lora/`) are released under the [MIT Licence](LICENSE). The `modeling_*` / `tokenization_*` /
`resampler.py` files are upstream model code and keep their original copyright headers and
licences, `YAOGAN_CHAT-main/` ships without its own licence file, and the base model weights
(`pytorch_model.bin`, not stored in this repository) are distributed by the NineGrids team under
their own terms - see [LICENSE](LICENSE) for the full breakdown.
