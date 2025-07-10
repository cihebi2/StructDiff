#!/usr/bin/env python3
"""
验证分离式训练环境配置
检查所有必要组件是否正常工作
"""

import os
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf

# 设置项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_environment():
    """检查环境配置"""
    print("🔍 环境检查...")
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # PyTorch版本和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA不可用")
        return False
    
    return True

def check_data_files():
    """检查数据文件"""
    print("\n📁 数据文件检查...")
    
    data_dir = Path("./data/processed")
    files_to_check = ["train.csv", "val.csv"]
    
    all_exist = True
    for file_name in files_to_check:
        file_path = data_dir / file_name
        if file_path.exists():
            line_count = sum(1 for _ in open(file_path))
            print(f"✅ {file_path}: {line_count} 行")
        else:
            print(f"❌ {file_path}: 文件不存在")
            all_exist = False
    
    return all_exist

def check_structure_cache():
    """检查结构缓存"""
    print("\n🗂️  结构缓存检查...")
    
    cache_dir = Path("./cache")
    subdirs = ["train", "val"]
    
    cache_info = {}
    for subdir in subdirs:
        subdir_path = cache_dir / subdir
        if subdir_path.exists():
            pkl_files = list(subdir_path.glob("*.pkl"))
            cache_info[subdir] = len(pkl_files)
            print(f"✅ {subdir_path}: {len(pkl_files)} 个缓存文件")
        else:
            cache_info[subdir] = 0
            print(f"⚠️  {subdir_path}: 目录不存在")
    
    return cache_info

def check_config_file():
    """检查配置文件"""
    print("\n⚙️  配置文件检查...")
    
    config_path = Path("configs/separated_training_production.yaml")
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    try:
        config = OmegaConf.load(config_path)
        print(f"✅ 配置文件加载成功: {config_path}")
        
        # 检查关键配置
        key_sections = ["model", "data", "separated_training", "resources"]
        for section in key_sections:
            if section in config:
                print(f"  ✅ {section} 配置存在")
            else:
                print(f"  ❌ {section} 配置缺失")
        
        # 检查结构特征设置
        use_structures = config.data.get('use_predicted_structures', False)
        print(f"  结构特征: {'启用' if use_structures else '禁用'}")
        
        return config
    
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def check_model_imports():
    """检查模型相关导入"""
    print("\n📦 模型导入检查...")
    
    try:
        from structdiff.models.structdiff import StructDiff
        print("✅ StructDiff模型导入成功")
    except Exception as e:
        print(f"❌ StructDiff模型导入失败: {e}")
        return False
    
    try:
        from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
        print("✅ GaussianDiffusion导入成功")
    except Exception as e:
        print(f"❌ GaussianDiffusion导入失败: {e}")
        return False
    
    try:
        from structdiff.training.separated_training import SeparatedTrainingManager
        print("✅ SeparatedTrainingManager导入成功")
    except Exception as e:
        print(f"❌ SeparatedTrainingManager导入失败: {e}")
        return False
    
    try:
        from structdiff.data.dataset import PeptideStructureDataset
        print("✅ PeptideStructureDataset导入成功")
    except Exception as e:
        print(f"❌ PeptideStructureDataset导入失败: {e}")
        return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n🧪 模型创建测试...")
    
    try:
        # 创建简化配置
        config = OmegaConf.create({
            "model": {
                "type": "StructDiff",
                "sequence_encoder": {
                    "pretrained_model": "facebook/esm2_t6_8M_UR50D",
                    "freeze_encoder": False
                },
                "structure_encoder": {
                    "type": "multi_scale",
                    "hidden_dim": 256,
                    "use_esmfold": False  # 测试时禁用
                },
                "denoiser": {
                    "hidden_dim": 320,
                    "num_layers": 4,
                    "num_heads": 4,
                    "dropout": 0.1,
                    "use_cross_attention": False
                },
                "sequence_decoder": {
                    "hidden_dim": 320,
                    "num_layers": 2,
                    "vocab_size": 33
                }
            },
            "diffusion": {
                "num_timesteps": 100,
                "noise_schedule": "sqrt",
                "beta_start": 0.0001,
                "beta_end": 0.02
            }
        })
        
        from structdiff.models.structdiff import StructDiff
        from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
        
        # 创建模型
        model = StructDiff(config.model)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型创建成功，参数数量: {param_count:,}")
        
        # 创建扩散过程
        diffusion = GaussianDiffusion(
            num_timesteps=config.diffusion.num_timesteps,
            noise_schedule=config.diffusion.noise_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )
        print("✅ 扩散过程创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_allocation():
    """测试GPU分配"""
    print("\n🖥️  GPU分配测试...")
    
    # 检查目标GPU可用性
    target_gpus = [2, 3, 4, 5]
    available_gpus = []
    
    for gpu_id in target_gpus:
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            try:
                # 尝试在GPU上创建张量
                test_tensor = torch.randn(10, 10).to(device)
                available_gpus.append(gpu_id)
                print(f"✅ GPU {gpu_id} 可用")
            except Exception as e:
                print(f"❌ GPU {gpu_id} 不可用: {e}")
        else:
            print(f"❌ GPU {gpu_id} 不存在")
    
    print(f"可用目标GPU: {available_gpus}")
    return len(available_gpus) >= 2  # 至少需要2个GPU

def main():
    """主验证函数"""
    print("🚀 StructDiff分离式训练环境验证")
    print("=" * 50)
    
    all_checks = []
    
    # 环境检查
    all_checks.append(check_environment())
    
    # 数据文件检查
    all_checks.append(check_data_files())
    
    # 结构缓存检查
    cache_info = check_structure_cache()
    all_checks.append(sum(cache_info.values()) > 0)
    
    # 配置文件检查
    config = check_config_file()
    all_checks.append(config is not None)
    
    # 模型导入检查
    all_checks.append(check_model_imports())
    
    # 模型创建测试
    all_checks.append(test_model_creation())
    
    # GPU分配测试
    all_checks.append(test_gpu_allocation())
    
    # 总结
    print("\n" + "=" * 50)
    print("🎯 验证总结:")
    
    passed_checks = sum(all_checks)
    total_checks = len(all_checks)
    
    if passed_checks == total_checks:
        print("🎉 所有检查通过！分离式训练环境配置正确")
        print("\n🚀 可以开始训练:")
        print("  ./start_separated_training.sh both")
        print("  或分阶段训练:")
        print("  ./start_separated_training.sh 1  # 只训练阶段1")
        print("  ./start_separated_training.sh 2  # 只训练阶段2")
        return True
    else:
        print(f"❌ {total_checks - passed_checks}/{total_checks} 个检查失败")
        print("请修复上述问题后重新验证")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 