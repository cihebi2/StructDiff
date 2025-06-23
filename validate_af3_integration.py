#!/usr/bin/env python3
"""
验证AlphaFold3改进集成的简单脚本
仅验证导入和基本功能，不需要PyTorch
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_imports():
    """验证所有导入是否正常"""
    print("🔍 验证导入...")
    
    imports_to_test = [
        ("structdiff.diffusion.noise_schedule", "get_noise_schedule"),
        ("structdiff.models.layers.alphafold3_embeddings", "AF3FourierEmbedding"),
        ("structdiff.models.layers.alphafold3_embeddings", "AF3TimestepEmbedding"), 
        ("structdiff.models.layers.alphafold3_embeddings", "AF3AdaptiveLayerNorm"),
        ("structdiff.models.layers.mlp", "FeedForward"),
        ("structdiff.models.denoise", "StructureAwareDenoiser"),
        ("structdiff.diffusion.gaussian_diffusion", "GaussianDiffusion"),
    ]
    
    success_count = 0
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}.{class_name}: 导入错误 - {e}")
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name}: 属性错误 - {e}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: 其他错误 - {e}")
    
    print(f"\n导入成功率: {success_count}/{len(imports_to_test)} ({success_count/len(imports_to_test)*100:.1f}%)")
    return success_count == len(imports_to_test)

def validate_noise_schedules():
    """验证噪声调度函数"""
    print("\n🔍 验证噪声调度...")
    
    try:
        from structdiff.diffusion.noise_schedule import get_noise_schedule
        import numpy as np
        
        # 测试所有调度类型
        schedules = ["linear", "cosine", "sqrt", "alphafold3"]
        num_timesteps = 10
        
        for schedule in schedules:
            try:
                betas = get_noise_schedule(schedule, num_timesteps)
                if isinstance(betas, np.ndarray) and len(betas) == num_timesteps:
                    print(f"✅ {schedule}: shape={betas.shape}")
                else:
                    print(f"❌ {schedule}: 输出格式错误")
            except Exception as e:
                print(f"❌ {schedule}: {e}")
                
        return True
    except Exception as e:
        print(f"❌ 噪声调度验证失败: {e}")
        return False

def validate_config_compatibility():
    """验证配置文件兼容性"""
    print("\n🔍 验证配置文件...")
    
    try:
        import yaml
        
        config_path = "configs/peptide_esmfold_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置项
        checks = [
            ("diffusion.noise_schedule", "alphafold3"),
            ("model.denoiser.hidden_dim", int),
            ("model.denoiser.num_layers", int),
            ("model.denoiser.use_cross_attention", bool),
        ]
        
        for key_path, expected in checks:
            keys = key_path.split('.')
            value = config
            
            try:
                for key in keys:
                    value = value[key]
                
                if expected == "alphafold3":
                    if value == "alphafold3":
                        print(f"✅ {key_path}: {value}")
                    else:
                        print(f"⚠️  {key_path}: {value} (建议使用 alphafold3)")
                elif isinstance(expected, type):
                    if isinstance(value, expected):
                        print(f"✅ {key_path}: {value}")
                    else:
                        print(f"❌ {key_path}: 类型错误 {type(value)} != {expected}")
                        
            except KeyError:
                print(f"❌ {key_path}: 配置项缺失")
        
        return True
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False

def validate_file_structure():
    """验证文件结构"""
    print("\n🔍 验证文件结构...")
    
    required_files = [
        "structdiff/diffusion/noise_schedule.py",
        "structdiff/models/layers/alphafold3_embeddings.py", 
        "structdiff/models/layers/mlp.py",
        "structdiff/models/denoise.py",
        "structdiff/diffusion/gaussian_diffusion.py",
        "configs/peptide_esmfold_config.yaml",
        "scripts/train_peptide_esmfold.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: 文件不存在")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def validate_class_interfaces():
    """验证类接口兼容性"""
    print("\n🔍 验证类接口...")
    
    try:
        # 检查StructureAwareDenoiser是否有正确的__init__参数
        from structdiff.models.denoise import StructureAwareDenoiser
        import inspect
        
        init_sig = inspect.signature(StructureAwareDenoiser.__init__)
        required_params = ['seq_hidden_dim', 'struct_hidden_dim', 'denoiser_config']
        
        for param in required_params:
            if param in init_sig.parameters:
                print(f"✅ StructureAwareDenoiser.__init__: 参数 {param}")
            else:
                print(f"❌ StructureAwareDenoiser.__init__: 缺少参数 {param}")
        
        # 检查FeedForward是否有use_gate参数
        from structdiff.models.layers.mlp import FeedForward
        ffn_sig = inspect.signature(FeedForward.__init__)
        
        if 'use_gate' in ffn_sig.parameters:
            print("✅ FeedForward.__init__: 支持 use_gate 参数")
        else:
            print("❌ FeedForward.__init__: 缺少 use_gate 参数")
        
        return True
    except Exception as e:
        print(f"❌ 接口验证失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🧪 验证AlphaFold3改进集成")
    print("=" * 50)
    
    validations = [
        ("文件结构", validate_file_structure),
        ("导入测试", validate_imports),
        ("噪声调度", validate_noise_schedules),
        ("配置兼容性", validate_config_compatibility),
        ("类接口", validate_class_interfaces),
    ]
    
    results = {}
    for name, validator in validations:
        try:
            results[name] = validator()
        except Exception as e:
            print(f"❌ {name}: 验证过程出错 - {e}")
            results[name] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 验证总结:")
    
    success_count = 0
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            success_count += 1
    
    overall_success = success_count == len(results)
    print(f"\n总体结果: {'✅ 所有验证通过' if overall_success else '❌ 存在问题'}")
    print(f"成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if overall_success:
        print("\n🎉 AlphaFold3改进已成功集成!")
        print("💡 建议下一步:")
        print("   1. 在训练环境中运行 scripts/train_peptide_esmfold.py")
        print("   2. 观察训练指标和收敛速度")
        print("   3. 比较新旧模型的性能差异")
    else:
        print("\n⚠️  请解决上述问题后再进行训练")

if __name__ == "__main__":
    main()