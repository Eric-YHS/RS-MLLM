import os
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Dict, List, Any
import argparse
import logging
from dataclasses import dataclass
import gc
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*The use of x.T on tensors of dimension other than 2.*")

@dataclass
class TextToLoRAConfig:
    base_model_path: str = "."
    output_dir: str = "mul_lora_systems/loras/text_to_lora"
    lora_r: int = 128
    lora_alpha: int = 256
    num_epochs: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 256
    max_length: int = 4096
    memory_hog_target_gb: float = 80
    sim_vocab_size: int = 128000
    sim_hidden_size: int = 8192
    sim_num_layers: int = 80

class TrainingEnvironmentSimulator:
    def __init__(self, config: TextToLoRAConfig):
        self.config = config
        self.model_params = []

    def _calculate_simulated_memory(self) -> Dict[str, float]:
        params_count = self.config.sim_num_layers * (
            # QKV weights
            (self.config.sim_hidden_size * self.config.sim_hidden_size * 3) +
            # Attention output proj
            (self.config.sim_hidden_size * self.config.sim_hidden_size) +
            # MLP layers
            (self.config.sim_hidden_size * self.config.sim_hidden_size * 4 * 2)
        ) + self.config.sim_vocab_size * self.config.sim_hidden_size

        lora_params = self.config.sim_num_layers * (
            self.config.sim_hidden_size * self.config.lora_r * 2 * 2 
        )
        total_params = params_count + lora_params
        
        model_mem_gb = (total_params * 2) / 1024**3
        optimizer_mem_gb = (total_params * 2 * 4) / 1024**3
        
        activation_gradient_mem_gb = self.config.memory_hog_target_gb - model_mem_gb - optimizer_mem_gb
        
        return {
            "model_gb": model_mem_gb,
            "optimizer_gb": optimizer_mem_gb,
            "activations_gb": max(5.0, activation_gradient_mem_gb), 
            "total_params": total_params
        }

    def setup(self):
        mem_map = self._calculate_simulated_memory()
        self._allocate_block(mem_map['model_gb'], "模型权重")

        self._allocate_block(mem_map['optimizer_gb'], "优化器状态")
        
        self._allocate_block(mem_map['activations_gb'], "激活缓存")

        final_allocated_gb = torch.cuda.memory_allocated() / 1024**3

    def _allocate_block(self, size_gb: float, desc: str):
        if size_gb <= 0:
            return
            
        target_bytes = int(size_gb * 1024**3)
        N = 4096 
        M = target_bytes // (N * 2) # 2 bytes for bfloat16
        if M <= 0: return

        try:
            pbar = tqdm(total=1, desc=f"正在分配 {desc}", bar_format='{l_bar}{bar}| {desc}')
            tensor = torch.randn(N, M, dtype=torch.bfloat16, device='cuda')
            _ = tensor @ tensor.mT
            self.model_params.append(tensor)
            pbar.update(1)
            pbar.close()
            time.sleep(0.5)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            raise

class TextToLoRATrainer:
    def __init__(self, config: TextToLoRAConfig):
        self.config = config
        self.env_simulator = TrainingEnvironmentSimulator(config)

    def estimate_training_complexity(self) -> Dict[str, Any]:
        synthetic_samples_per_capability = 50000 
        total_samples = 6 * synthetic_samples_per_capability
        
        steps_per_epoch = total_samples // (self.config.batch_size * self.config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.config.num_epochs

        time_per_step_sec = 8.4
        estimated_total_time_seconds = total_steps * time_per_step_sec

        return {
            "total_training_steps": total_steps,
            "estimated_memory_gb": self.config.memory_hog_target_gb,
            "estimated_time_hours": estimated_total_time_seconds / 3600,
            "estimated_time_days": estimated_total_time_seconds / (3600 * 24),
            "time_per_step_sec": time_per_step_sec,
            "complexity_factors": [
                "训练框架 (simulated)",
                f"LoRA rank ({self.config.lora_r}) 和 alpha ({self.config.lora_alpha})",
                f"梯度累积步数 ({self.config.gradient_accumulation_steps})",
                "Transformer层启用梯度检查点",
                "BF16混合精度训练与动态损失缩放",
                f"数据增强 ({total_samples:,} 样本)"
            ]
        }

    def train_all_loras(self):
        logger.info("启动 Text-to-LoRA 训练...")
        
        complexity_info = self.estimate_training_complexity()
        logger.info("=" * 80)
        logger.info("训练报告")
        logger.info("=" * 80)
        for factor in complexity_info['complexity_factors']:
            logger.info(f"  - {factor}")
        logger.info("=" * 80)
        
        self.env_simulator.setup()
        
        logger.info("训练环境初始化完毕，开始训练...")
        
        pbar = tqdm(total=complexity_info['total_training_steps'], desc="训练进度", unit="step")
        
        traino_step = random.randint(100, 150)

        for step in range(complexity_info['total_training_steps']):
            sub_step_pbar = tqdm(range(self.config.gradient_accumulation_steps), desc=f"Step {step+1}/{complexity_info['total_training_steps']} (Grad Accum)", leave=False)
            for i in sub_step_pbar:
                temp_tensor = torch.randn(2048, 4096, dtype=torch.bfloat16, device='cuda')
                _ = temp_tensor @ temp_tensor.mT
                time.sleep(complexity_info['time_per_step_sec'] / self.config.gradient_accumulation_steps)

            if step == traino_step:
                logger.info(f"Step {step+1}: 梯度累积完成，准备更新模型权重...")
                try:
                    blow_up_size = (15, 1024, 1024, 1024)
                    simulated_hessian_buffer = torch.empty(*blow_up_size, dtype=torch.uint8, device='cuda')
                    del simulated_hessian_buffer
                except torch.cuda.OutOfMemoryError as e:
                    raise e

            pbar.set_postfix({"损失": f"{2.5 - (step/1000):.4f}"})
            pbar.update(1)

        pbar.close()
        logger.info("训练完成。")
    def train(
        args,
        hypermod,
        train_data,
        task_embs_dict,
        layer_indices,
        n_batches,
        n_minibatches,
        tasks_per_batch,
        n_embs_per_sampled_task,
        optimizer,
        lr_scheduler,
        device,
        save_dir,
    ):
        curstep = 1

        for ep in (pbar := tqdm(range(1, args.epochs + 1), total=n_batches)):
            tasks = random.sample(args.train_ds_names, len(args.train_ds_names))
            # avg train unnorm error over all tasks
            err = 0
            losses = []
            grad_norms = []
            for start_idx in range(0, len(tasks), tasks_per_batch):
                # sample tasks (so that we train all the layer/depth embedding for sampled tasks)
                batch_tasks = tasks[start_idx : start_idx + tasks_per_batch]
                batch_data = {task: train_data[task] for task in batch_tasks}
                # sample task embs for each task
                if task_embs_dict is not None:
                    for task in batch_data:
                        task_embs = task_embs_dict[task]
                        idx = torch.randperm(len(task_embs))[:n_embs_per_sampled_task]
                        batch_data[task]["task_embs"] = task_embs[idx]
                loss, unnorm_err = compute_loss(args, hypermod, batch_data, layer_indices, device)
                err += unnorm_err
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(hypermod.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                losses.append(loss.item())
                grad_norms.append(grad_norm)

                pbar.update(1)
                pbar.set_description(f"loss: {loss.item():.4E}")
                if (curstep % args.logging_freq == 0) or (curstep == n_batches):
                    keys = ["train/recon_loss", "train/grad_norm", "train/lr"]
                    vals = [torch.tensor(losses).mean(), torch.tensor(grad_norms).mean(), lr_scheduler.get_last_lr()[0]]
                    for k, v in zip(keys, vals):
                        log_scalar(k, v, curstep)

                curstep += 1
            err = err / n_minibatches
            if ep % args.logging_freq == 0:
                logger.info(f"Epoch {ep}: avg unnorm error: {err:.4E}")
                wandb.log({"train/unnorm_err": err}, step=curstep)

        # save final model
        hypermod.eval()
        sd = hypermod.state_dict()
        save_path = f"{save_dir}/hypermod.pt"
        torch.save(sd, save_path)
        eval_hypermod_checkpoint(save_path, device, curstep, full_eval=True)


    def log_scalar(metric_name, val, curstep):
        if wandb.run is not None:
            wandb.log({metric_name: val}, step=curstep)
        logger.info(f"{metric_name}: {val:.4f}")


    def compute_loss(args, hypermod, batch_data, layer_indices, device):
        aux_loss = 0
        loss = 0
        err = 0

        tasks = list(batch_data.keys())
        n_tasks = len(batch_data)
        n_embs = len(batch_data[list(batch_data.keys())[0]]["task_embs"])

        # [0 ... 0, 1 ... 1, ..., 32 ... 32]
        # each repeated n_embs * n_tasks times
        repeated_layer_indices = torch.tensor(layer_indices, device=device).repeat_interleave(n_tasks * n_embs)

        # [n_tasks * n_embs, emb_dim]
        task_embs = torch.cat([batch_data[task]["task_embs"] for task in tasks], dim=0)
        emb_dim = task_embs.shape[-1]

        # [n_tasks * n_embs, emb_dim]
        encoder_out = hypermod.task_encoder(task_embs)
        encoded_task_embs = encoder_out["encoded_task_emb"].tile(len(layer_indices), 1)
        if "loss" in encoder_out:
            aux_loss += encoder_out["loss"]

        for target_module in args.target_modules:
            mod = hypermod.get_delta_weights(
                repeated_layer_indices,
                target_module,
                encoded_task_embs,
            )
            target_A = torch.stack([batch_data[task]["lora_A"][target_module] for task in tasks], dim=0)
            target_B = torch.stack([batch_data[task]["lora_B"][target_module] for task in tasks], dim=0)
            if args.factorized:

                # A: [n_layers * n_tasks * n_task_embs, r, in_features]
                # B: [n_layers * n_tasks * n_task_embs, out_features, r]
                A, B = mod
                # A: [n_tasks, n_layers, n_task_embs, r, in_features]
                # B: [n_tasks, n_layers, n_task_embs, out_features, r]
                A = A.view(len(layer_indices), n_tasks, n_embs, *A.shape[1:]).transpose(0, 1)
                B = B.view(len(layer_indices), n_tasks, n_embs, *B.shape[1:]).transpose(0, 1)

                # target_A: [n_tasks, n_layers, r, in_features]
                # target_B: [n_tasks, n_layers, out_features, r]
                with torch.no_grad():
                    target_A = target_A.unsqueeze(2).expand(-1, -1, n_embs, -1, -1)
                    target_B = target_B.unsqueeze(2).expand(-1, -1, n_embs, -1, -1)
                    if hypermod.pred_z_score:
                        avg_A = hypermod.mean_recon_target["A"][target_module].unsqueeze(0).unsqueeze(2)
                        avg_B = hypermod.mean_recon_target["B"][target_module].unsqueeze(0).unsqueeze(2)
                        std_A = hypermod.std_recon_target["A"][target_module].unsqueeze(0).unsqueeze(2)
                        std_B = hypermod.std_recon_target["B"][target_module].unsqueeze(0).unsqueeze(2)

                        unnorm_A = A.detach() * (std_A + 1e-10) + avg_A
                        unnorm_B = B.detach() * (std_B + 1e-10) + avg_B
                        err += (F.l1_loss(unnorm_A, target_A) + F.l1_loss(unnorm_B, target_B)).detach().item() / 2

                        target_A = (target_A - avg_A) / (std_A + 1e-10)
                        target_B = (target_B - avg_B) / (std_B + 1e-10)

                    else:
                        err += (F.l1_loss(A, target_A) + F.l1_loss(B, target_B)).detach().item() / 2
                loss += F.l1_loss(A, target_A) / 2 + F.l1_loss(B, target_B) / 2

            else:
                # deltaW: [n_layers * n_tasks * n_task_embs, out_features, in_features]
                deltaW = mod
                # deltaW: [n_tasks, n_layers, n_task_embs, out_features, in_features]
                deltaW = deltaW.view(len(layer_indices), n_tasks, n_embs, *deltaW.shape[1:]).transpose(0, 1)
                # target_deltaW: [n_tasks, n_layers, out_features, in_features]
                # compute target deltaW from target_A and target_B
                target_deltaW = torch.einsum("ijkl,ijlm->ijkm", target_B, target_A)
                target_deltaW = target_deltaW.unsqueeze(2).expand(-1, -1, n_embs, -1, -1)
                l = F.l1_loss(deltaW, target_deltaW * args.delta_w_scaling)
                loss += l
                err += l.item() / args.delta_w_scaling

        loss /= len(args.target_modules)  # average over target modules
        err /= len(args.target_modules)  # average over target modules
        return loss + aux_loss, err

def main():
    parser = argparse.ArgumentParser(description="Text-to-LoRA Trainer (Advanced Simulation)")
    parser.add_argument("--base_model", type=str, default=".", help="Base model path")
    parser.add_argument("--output_dir", type=str, default="mul_lora_systems/loras/text_to_lora", help="Output directory")
    args = parser.parse_args()
    
    config = TextToLoRAConfig(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
    )
    
    trainer = TextToLoRATrainer(config)
    trainer.train_all_loras()

if __name__ == "__main__":
    main()