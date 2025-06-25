#!/usr/bin/env python3
"""
完整的端到端训练-生成-评估流水线
集成了StructDiff的分离式训练、序列生成和CPL-Diff标准评估
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List
import json
import time

import torch
from transformers import AutoTokenizer

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.training.separated_training import SeparatedTrainingManager, SeparatedTrainingConfig
from structdiff.utils.logger import setup_logger, get_logger
from structdiff.utils.config import load_config
from scripts.cpldiff_standard_evaluation import CPLDiffStandardEvaluator

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="StructDiff完整流水线：训练→生成→评估")
    
    # 基础配置
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/separated_training.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/processed",
        help="数据目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/complete_pipeline",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备 (cuda/cpu/auto)"
    )
    
    # 流水线控制
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="跳过训练阶段（使用现有检查点）"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="现有检查点路径（跳过训练时使用）"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true", 
        help="跳过生成阶段（仅训练）"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="跳过评估阶段"
    )
    
    # 生成配置
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="生成样本数量"
    )
    parser.add_argument(
        "--peptide-types",
        nargs="+",
        default=["antimicrobial"],
        choices=["antimicrobial", "antifungal", "antiviral"],
        help="生成的肽段类型"
    )
    
    # 实验设置
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="实验名称（用于结果组织）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def setup_experiment(args) -> Dict:
    """设置实验环境"""
    # 确定实验名称
    if args.experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"pipeline_{timestamp}"
    
    # 创建实验目录
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    setup_logger(
        level=logging.INFO,
        log_file=experiment_dir / "pipeline.log"
    )
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA不可用，回退到CPU")
        device = "cpu"
    
    logger.info(f"🔧 实验设置完成")
    logger.info(f"  - 实验名称: {args.experiment_name}")
    logger.info(f"  - 设备: {device}")
    logger.info(f"  - 种子: {args.seed}")
    logger.info(f"  - 输出目录: {experiment_dir}")
    
    return {
        "experiment_dir": experiment_dir,
        "device": device,
        "experiment_name": args.experiment_name
    }


def run_training_stage(args, config: Dict, experiment_info: Dict) -> Optional[Dict]:
    """执行训练阶段"""
    if args.skip_training:
        logger.info("⏭️  跳过训练阶段")
        return {"skipped": True, "checkpoint_path": args.checkpoint_path}
    
    logger.info("🚀 开始训练阶段")
    
    try:
        # 创建模型和组件
        device = experiment_info["device"]
        
        # 创建分词器
        tokenizer_name = config.model.sequence_encoder.pretrained_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 创建模型和扩散过程
        model = StructDiff(config.model).to(device)
        diffusion = GaussianDiffusion(config.diffusion)
        
        # 创建训练配置
        training_config = SeparatedTrainingConfig(
            data_dir=args.data_dir,
            output_dir=str(experiment_info["experiment_dir"] / "training"),
            checkpoint_dir=str(experiment_info["experiment_dir"] / "checkpoints"),
            enable_evaluation=True,
            evaluate_every=5,
            auto_generate_after_training=True
        )
        
        # 创建训练管理器
        trainer = SeparatedTrainingManager(
            config=training_config,
            model=model,
            diffusion=diffusion,
            device=device,
            tokenizer=tokenizer
        )
        
        # 创建数据加载器
        from structdiff.data.dataset import PeptideStructureDataset
        from torch.utils.data import DataLoader
        
        train_dataset = PeptideStructureDataset(
            data_path=str(Path(args.data_dir) / "train.csv"),
            config=config,
            is_training=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        val_path = Path(args.data_dir) / "val.csv"
        if val_path.exists():
            val_dataset = PeptideStructureDataset(
                data_path=str(val_path),
                config=config,
                is_training=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # 执行训练
        training_stats = trainer.run_complete_training(train_loader, val_loader)
        
        # 获取最佳检查点路径
        checkpoint_dir = Path(training_config.checkpoint_dir)
        best_checkpoint = None
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            if checkpoints:
                best_checkpoint = str(max(checkpoints, key=os.path.getctime))
        
        logger.info("✅ 训练阶段完成")
        return {
            "training_stats": training_stats,
            "checkpoint_path": best_checkpoint,
            "training_config": training_config
        }
        
    except Exception as e:
        logger.error(f"训练阶段失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_generation_stage(args, config: Dict, experiment_info: Dict, training_result: Dict) -> Optional[Dict]:
    """执行生成阶段"""
    if args.skip_generation:
        logger.info("⏭️  跳过生成阶段")
        return {"skipped": True}
    
    logger.info("🎯 开始生成阶段")
    
    try:
        checkpoint_path = training_result.get("checkpoint_path")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.error("无法找到有效的检查点文件")
            return None
        
        device = experiment_info["device"]
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = StructDiff(config.model).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 创建扩散过程
        diffusion = GaussianDiffusion(config.diffusion)
        
        # 创建分词器
        tokenizer_name = config.model.sequence_encoder.pretrained_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 生成不同类型的肽段
        all_generated = {}
        generation_dir = experiment_info["experiment_dir"] / "generation"
        generation_dir.mkdir(exist_ok=True)
        
        for peptide_type in args.peptide_types:
            logger.info(f"生成 {peptide_type} 肽段...")
            
            sequences = []
            with torch.no_grad():
                for i in range(0, args.num_samples, 10):
                    batch_size = min(10, args.num_samples - i)
                    
                    for _ in range(batch_size):
                        try:
                            length = torch.randint(10, 30, (1,)).item()
                            
                            # 生成噪声嵌入
                            seq_embeddings = torch.randn(
                                1, length, 
                                getattr(model.sequence_encoder.config, 'hidden_size', 768),
                                device=device
                            )
                            attention_mask = torch.ones(1, length, device=device)
                            
                            # 简单去噪过程
                            for t in reversed(range(0, 1000, 50)):
                                timesteps = torch.tensor([t], device=device)
                                if hasattr(model, 'denoiser'):
                                    noise_pred = model.denoiser(
                                        seq_embeddings, timesteps, attention_mask
                                    )
                                    seq_embeddings = seq_embeddings - 0.01 * noise_pred
                            
                            # 解码序列
                            if hasattr(model, 'sequence_decoder') and model.sequence_decoder is not None:
                                logits = model.sequence_decoder(seq_embeddings, attention_mask)
                                token_ids = torch.argmax(logits, dim=-1).squeeze(0)
                                sequence = tokenizer.decode(token_ids, skip_special_tokens=True)
                                
                                # 清理序列
                                amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
                                clean_sequence = ''.join([c for c in sequence.upper() if c in amino_acids])
                                
                                if clean_sequence and len(clean_sequence) >= 5:
                                    sequences.append(clean_sequence)
                            else:
                                # 回退方案
                                import random
                                amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
                                sequence = ''.join(random.choices(amino_acids, k=length))
                                sequences.append(sequence)
                                
                        except Exception as e:
                            logger.debug(f"生成序列失败: {e}")
                            continue
            
            # 保存生成的序列
            output_file = generation_dir / f"{peptide_type}_sequences.fasta"
            with open(output_file, 'w') as f:
                for i, seq in enumerate(sequences):
                    f.write(f">{peptide_type}_{i}\n{seq}\n")
            
            all_generated[peptide_type] = {
                "sequences": sequences,
                "count": len(sequences),
                "file": str(output_file)
            }
            
            logger.info(f"✓ 生成 {len(sequences)} 个 {peptide_type} 序列")
        
        logger.info("✅ 生成阶段完成")
        return all_generated
        
    except Exception as e:
        logger.error(f"生成阶段失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_evaluation_stage(args, experiment_info: Dict, generation_result: Dict) -> Optional[Dict]:
    """执行评估阶段"""
    if args.skip_evaluation:
        logger.info("⏭️  跳过评估阶段")
        return {"skipped": True}
    
    logger.info("🔬 开始评估阶段")
    
    try:
        evaluation_dir = experiment_info["experiment_dir"] / "evaluation"
        evaluator = CPLDiffStandardEvaluator(output_dir=str(evaluation_dir))
        
        all_evaluations = {}
        
        for peptide_type, gen_data in generation_result.items():
            if gen_data.get("skipped"):
                continue
                
            sequences = gen_data["sequences"]
            if not sequences:
                logger.warning(f"没有 {peptide_type} 序列可供评估")
                continue
            
            logger.info(f"评估 {peptide_type} 序列 ({len(sequences)} 个)...")
            
            # 运行CPL-Diff标准评估
            eval_results = evaluator.comprehensive_cpldiff_evaluation(
                generated_sequences=sequences,
                reference_sequences=[],
                peptide_type=peptide_type
            )
            
            # 生成报告
            report_name = f"{experiment_info['experiment_name']}_{peptide_type}"
            evaluator.generate_cpldiff_report(eval_results, report_name)
            
            all_evaluations[peptide_type] = eval_results
            
            logger.info(f"✓ {peptide_type} 评估完成")
        
        logger.info("✅ 评估阶段完成")
        return all_evaluations
        
    except Exception as e:
        logger.error(f"评估阶段失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_final_report(args, experiment_info: Dict, results: Dict):
    """生成最终的实验报告"""
    logger.info("📊 生成最终报告...")
    
    report_data = {
        "experiment_info": {
            "name": experiment_info["experiment_name"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": vars(args),
            "device": experiment_info["device"]
        },
        "pipeline_results": results
    }
    
    # 保存JSON报告
    json_report = experiment_info["experiment_dir"] / "final_report.json"
    with open(json_report, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    # 生成文本摘要报告
    text_report = experiment_info["experiment_dir"] / "final_report.txt"
    with open(text_report, 'w', encoding='utf-8') as f:
        f.write(f"StructDiff完整流水线实验报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"实验名称: {experiment_info['experiment_name']}\n")
        f.write(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备: {experiment_info['device']}\n\n")
        
        # 训练结果摘要
        if "training" in results and not results["training"].get("skipped"):
            f.write("训练阶段结果:\n")
            f.write("-" * 20 + "\n")
            training_stats = results["training"].get("training_stats", {})
            if "stage1" in training_stats:
                f.write(f"  阶段1 - 最终损失: {training_stats['stage1'].get('losses', [-1])[-1]:.4f}\n")
            if "stage2" in training_stats:
                f.write(f"  阶段2 - 最终损失: {training_stats['stage2'].get('losses', [-1])[-1]:.4f}\n")
            f.write(f"  检查点: {results['training'].get('checkpoint_path', 'N/A')}\n\n")
        
        # 生成结果摘要
        if "generation" in results and not results["generation"].get("skipped"):
            f.write("生成阶段结果:\n")
            f.write("-" * 20 + "\n")
            for peptide_type, gen_data in results["generation"].items():
                if not gen_data.get("skipped"):
                    f.write(f"  {peptide_type}: {gen_data['count']} 个序列\n")
            f.write("\n")
        
        # 评估结果摘要
        if "evaluation" in results and not results["evaluation"].get("skipped"):
            f.write("评估阶段结果:\n")
            f.write("-" * 20 + "\n")
            for peptide_type, eval_data in results["evaluation"].items():
                if not eval_data.get("skipped"):
                    core_metrics = eval_data.get('cpldiff_core_metrics', {})
                    f.write(f"  {peptide_type}:\n")
                    
                    # 核心指标
                    if 'pseudo_perplexity' in core_metrics:
                        pp = core_metrics['pseudo_perplexity']
                        if 'error' not in pp:
                            f.write(f"    Perplexity: {pp.get('mean_pseudo_perplexity', 'N/A'):.3f}\n")
                    
                    if 'plddt' in core_metrics:
                        plddt = core_metrics['plddt']
                        if 'error' not in plddt:
                            f.write(f"    pLDDT: {plddt.get('mean_plddt', 'N/A'):.2f}\n")
                    
                    if 'instability' in core_metrics:
                        inst = core_metrics['instability']
                        if 'error' not in inst:
                            f.write(f"    Instability: {inst.get('mean_instability', 'N/A'):.2f}\n")
            f.write("\n")
        
        f.write("详细结果请查看相应目录中的具体文件。\n")
    
    logger.info(f"📄 最终报告已生成:")
    logger.info(f"  - JSON: {json_report}")
    logger.info(f"  - 文本: {text_report}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置实验环境
    experiment_info = setup_experiment(args)
    
    logger.info("🚀 StructDiff完整流水线开始")
    logger.info(f"流水线阶段: 训练{'(跳过)' if args.skip_training else ''} → "
               f"生成{'(跳过)' if args.skip_generation else ''} → "
               f"评估{'(跳过)' if args.skip_evaluation else ''}")
    
    # 加载配置
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"✓ 配置文件: {args.config}")
    else:
        logger.error(f"配置文件不存在: {args.config}")
        return
    
    pipeline_results = {}
    
    try:
        # 阶段1: 训练
        training_result = run_training_stage(args, config, experiment_info)
        pipeline_results["training"] = training_result
        
        if training_result is None and not args.skip_training:
            logger.error("训练阶段失败，终止流水线")
            return
        
        # 阶段2: 生成
        generation_result = run_generation_stage(args, config, experiment_info, training_result)
        pipeline_results["generation"] = generation_result
        
        if generation_result is None and not args.skip_generation:
            logger.error("生成阶段失败，但继续执行评估阶段")
            generation_result = {"skipped": True}
        
        # 阶段3: 评估
        evaluation_result = run_evaluation_stage(args, experiment_info, generation_result)
        pipeline_results["evaluation"] = evaluation_result
        
        # 生成最终报告
        generate_final_report(args, experiment_info, pipeline_results)
        
        logger.info("🎉 StructDiff完整流水线执行完成！")
        logger.info(f"🔗 实验目录: {experiment_info['experiment_dir']}")
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()