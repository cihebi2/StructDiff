#!/usr/bin/env python3
"""
分阶段训练控制器 - 根据开发规划自动管理训练流程
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class StageController:
    """训练阶段控制器"""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.stage_history = []
        self.current_stage = None
        
        # 阶段配置
        self.stages = {
            "stage1_validation": {
                "name": "基础验证与稳定性测试",
                "duration_days": 7,
                "config_overrides": {
                    "data": {
                        "batch_size": 4,
                        "max_length": 30,
                        "train_subset_size": 1000,
                        "val_subset_size": 200
                    },
                    "training": {
                        "num_epochs": 10,
                        "gradient_accumulation_steps": 2
                    },
                    "experiment": {
                        "name": "peptide_stage1_validation"
                    }
                },
                "success_criteria": {
                    "min_epochs_completed": 5,
                    "max_train_loss": 2.0,
                    "min_generated_sequences": 100
                }
            },
            
            "stage2_optimization": {
                "name": "中等规模训练与优化",
                "duration_days": 14,
                "config_overrides": {
                    "data": {
                        "batch_size": 8,
                        "max_length": 40,
                        "train_subset_size": 10000,
                        "val_subset_size": 2000
                    },
                    "training": {
                        "num_epochs": 30,
                        "gradient_accumulation_steps": 4
                    },
                    "experiment": {
                        "name": "peptide_stage2_optimization"
                    }
                },
                "success_criteria": {
                    "min_epochs_completed": 20,
                    "max_train_loss": 1.5,
                    "min_val_accuracy": 0.6
                }
            },
            
            "stage3_scaling": {
                "name": "大规模训练与性能提升",
                "duration_days": 21,
                "config_overrides": {
                    "data": {
                        "batch_size": 16,
                        "max_length": 50,
                        "train_subset_size": None,  # 使用全部数据
                        "val_subset_size": None
                    },
                    "training": {
                        "num_epochs": 50,
                        "gradient_accumulation_steps": 8
                    },
                    "experiment": {
                        "name": "peptide_stage3_scaling"
                    }
                },
                "success_criteria": {
                    "min_epochs_completed": 40,
                    "max_train_loss": 1.0,
                    "min_val_accuracy": 0.75
                }
            }
        }
    
    def create_stage_config(self, stage_name: str) -> str:
        """为指定阶段创建配置文件"""
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        # 加载基础配置
        with open(self.base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # 应用阶段特定的覆盖
        stage_config = base_config.copy()
        overrides = self.stages[stage_name]["config_overrides"]
        
        for section, settings in overrides.items():
            if section in stage_config:
                stage_config[section].update(settings)
            else:
                stage_config[section] = settings
        
        # 保存阶段配置
        stage_config_path = f"configs/stage_{stage_name}.yaml"
        with open(stage_config_path, 'w') as f:
            yaml.dump(stage_config, f, default_flow_style=False, indent=2)
        
        return stage_config_path
    
    def start_stage(self, stage_name: str) -> Dict:
        """开始指定训练阶段"""
        print(f"🚀 开始阶段: {self.stages[stage_name]['name']}")
        
        # 创建阶段配置
        config_path = self.create_stage_config(stage_name)
        
        # 记录阶段开始
        stage_info = {
            "stage_name": stage_name,
            "start_time": datetime.now().isoformat(),
            "config_path": config_path,
            "status": "running"
        }
        
        self.current_stage = stage_info
        self.stage_history.append(stage_info)
        
        # 保存进度
        self.save_progress()
        
        print(f"📋 阶段配置已创建: {config_path}")
        print(f"⏱️ 预计持续时间: {self.stages[stage_name]['duration_days']} 天")
        
        return stage_info
    
    def complete_stage(self, stage_name: str, metrics: Dict) -> bool:
        """完成阶段并验证成功标准"""
        if not self.current_stage or self.current_stage["stage_name"] != stage_name:
            print(f"❌ 当前没有运行阶段 {stage_name}")
            return False
        
        success_criteria = self.stages[stage_name]["success_criteria"]
        success = True
        
        print(f"🏁 完成阶段: {self.stages[stage_name]['name']}")
        print("📊 验证成功标准:")
        
        for criterion, threshold in success_criteria.items():
            actual_value = metrics.get(criterion, 0)
            
            if criterion.startswith("min_"):
                passed = actual_value >= threshold
                comparison = ">="
            elif criterion.startswith("max_"):
                passed = actual_value <= threshold
                comparison = "<="
            else:
                passed = True
                comparison = "="
            
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}: {actual_value} {comparison} {threshold}")
            
            if not passed:
                success = False
        
        # 更新阶段状态
        self.current_stage["end_time"] = datetime.now().isoformat()
        self.current_stage["status"] = "completed" if success else "failed"
        self.current_stage["metrics"] = metrics
        self.current_stage["success"] = success
        
        self.save_progress()
        
        if success:
            print(f"🎉 阶段 {stage_name} 成功完成!")
            self.recommend_next_stage()
        else:
            print(f"⚠️ 阶段 {stage_name} 未达到成功标准，建议重新训练或调整参数")
        
        return success
    
    def recommend_next_stage(self):
        """推荐下一个训练阶段"""
        completed_stages = [s["stage_name"] for s in self.stage_history if s.get("success", False)]
        
        stage_order = ["stage1_validation", "stage2_optimization", "stage3_scaling"]
        
        for stage in stage_order:
            if stage not in completed_stages:
                print(f"💡 推荐下一阶段: {self.stages[stage]['name']}")
                print(f"   运行命令: python3 scripts/stage_controller.py --start {stage}")
                break
        else:
            print("🏆 所有基础阶段已完成! 可以开始高级优化阶段")
    
    def save_progress(self):
        """保存训练进度"""
        progress_file = "training_progress.json"
        progress_data = {
            "current_stage": self.current_stage,
            "stage_history": self.stage_history,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_progress(self):
        """加载训练进度"""
        progress_file = "training_progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                self.current_stage = progress_data.get("current_stage")
                self.stage_history = progress_data.get("stage_history", [])
    
    def show_status(self):
        """显示当前状态"""
        print("📊 训练进度状态")
        print("=" * 50)
        
        if self.current_stage:
            print(f"🔄 当前阶段: {self.current_stage['stage_name']}")
            print(f"   开始时间: {self.current_stage['start_time']}")
            print(f"   状态: {self.current_stage['status']}")
        else:
            print("⏸️ 当前没有运行的阶段")
        
        print(f"\n📈 历史阶段: {len(self.stage_history)} 个")
        for stage in self.stage_history:
            status_icon = "✅" if stage.get("success") else "❌" if stage.get("success") is False else "🔄"
            print(f"  {status_icon} {stage['stage_name']} - {stage['status']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="训练阶段控制器")
    parser.add_argument("--start", type=str, help="开始指定阶段")
    parser.add_argument("--complete", type=str, help="完成指定阶段")
    parser.add_argument("--status", action="store_true", help="显示状态")
    parser.add_argument("--config", type=str, default="configs/peptide_esmfold_config.yaml", help="基础配置文件")
    
    args = parser.parse_args()
    
    controller = StageController(args.config)
    controller.load_progress()
    
    if args.start:
        controller.start_stage(args.start)
    elif args.complete:
        # 这里需要从实际训练结果获取metrics
        metrics = {
            "min_epochs_completed": 10,  # 示例数据
            "max_train_loss": 1.8,
            "min_generated_sequences": 150
        }
        controller.complete_stage(args.complete, metrics)
    elif args.status:
        controller.show_status()
    else:
        print("请指定操作: --start, --complete, 或 --status")


if __name__ == "__main__":
    main()