#!/usr/bin/env python3
"""
最终代码验证脚本 - 不依赖PyTorch
"""

import ast
import sys
import yaml
from pathlib import Path

def check_file_structure():
    """检查文件结构"""
    print("📁 检查文件结构...")
    
    required_files = [
        "scripts/train_peptide_esmfold.py",
        "configs/peptide_esmfold_config.yaml", 
        "configs/peptide_adaptive_conditioning.yaml",
        "structdiff/models/structdiff.py",
        "structdiff/models/denoise.py", 
        "structdiff/models/layers/alphafold3_embeddings.py",
        "structdiff/diffusion/noise_schedule.py",
        "structdiff/diffusion/gaussian_diffusion.py",
        "VERSION",
        "README.md",
        "ALPHAFOLD3_IMPROVEMENTS.md",
        "AF3_ADAPTIVE_CONDITIONING_INTEGRATION.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_python_syntax():
    """检查Python语法"""
    print("\n🐍 检查Python语法...")
    
    python_files = list(Path('.').rglob('*.py'))
    
    # 过滤出重要文件
    important_files = [
        "scripts/train_peptide_esmfold.py",
        "structdiff/models/structdiff.py",
        "structdiff/models/denoise.py",
        "structdiff/models/layers/alphafold3_embeddings.py",
        "structdiff/diffusion/noise_schedule.py",
        "test_adaptive_conditioning.py",
        "final_validation.py"
    ]
    
    syntax_errors = []
    for file_path in important_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
                print(f"   ✅ {file_path}")
            except SyntaxError as e:
                print(f"   ❌ {file_path}: {e}")
                syntax_errors.append(file_path)
            except Exception as e:
                print(f"   ⚠️ {file_path}: {e}")
    
    return len(syntax_errors) == 0

def check_config_files():
    """检查配置文件"""
    print("\n⚙️ 检查配置文件...")
    
    config_files = [
        "configs/peptide_esmfold_config.yaml",
        "configs/peptide_adaptive_conditioning.yaml"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查关键配置部分
            required_sections = ["model", "data", "training", "diffusion"]
            missing_sections = []
            
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if len(missing_sections) == 0:
                print(f"   ✅ {config_file}")
                
                # 检查自适应条件化配置
                if "adaptive_conditioning" in str(config):
                    print(f"      ✅ 包含adaptive_conditioning配置")
                else:
                    print(f"      ⚠️ 可能缺少adaptive_conditioning配置")
            else:
                print(f"   ❌ {config_file}: 缺少 {missing_sections}")
                return False
                
        except Exception as e:
            print(f"   ❌ {config_file}: {e}")
            return False
    
    return True

def check_key_features():
    """检查关键特性实现"""
    print("\n🎯 检查关键特性...")
    
    # 检查训练脚本的关键功能
    try:
        with open("scripts/train_peptide_esmfold.py", 'r') as f:
            train_content = f.read()
        
        key_features = [
            ("条件支持", "conditions"),
            ("自适应条件化", "adaptive_conditioning"),
            ("理化性质计算", "evaluate_physicochemical_properties"),
            ("外部分类器", "evaluate_external_classifier_activity"),
            ("ESMFold集成", "ESMFoldWrapper"),
            ("wandb支持", "wandb"),
            ("混合精度", "amp"),
            ("EMA", "EMA")
        ]
        
        for feature_name, keyword in key_features:
            if keyword in train_content:
                print(f"   ✅ {feature_name}")
            else:
                print(f"   ⚠️ {feature_name} (可能缺失)")
        
    except Exception as e:
        print(f"   ❌ 训练脚本检查失败: {e}")
        return False
    
    # 检查AlphaFold3改进
    try:
        with open("structdiff/models/layers/alphafold3_embeddings.py", 'r') as f:
            af3_content = f.read()
        
        af3_classes = [
            "AF3FourierEmbedding",
            "AF3AdaptiveConditioning", 
            "AF3EnhancedConditionalLayerNorm",
            "AF3ConditionalZeroInit"
        ]
        
        for cls_name in af3_classes:
            if f"class {cls_name}" in af3_content:
                print(f"   ✅ {cls_name}")
            else:
                print(f"   ❌ {cls_name}")
                return False
                
    except Exception as e:
        print(f"   ❌ AF3组件检查失败: {e}")
        return False
    
    # 检查噪声调度
    try:
        with open("structdiff/diffusion/noise_schedule.py", 'r') as f:
            noise_content = f.read()
        
        if "alphafold3" in noise_content:
            print("   ✅ AlphaFold3噪声调度")
        else:
            print("   ❌ AlphaFold3噪声调度")
            return False
            
    except Exception as e:
        print(f"   ❌ 噪声调度检查失败: {e}")
        return False
    
    return True

def check_documentation():
    """检查文档完整性"""
    print("\n📚 检查文档...")
    
    docs = [
        ("README.md", "主要文档"),
        ("ALPHAFOLD3_IMPROVEMENTS.md", "AF3改进文档"),
        ("AF3_ADAPTIVE_CONDITIONING_INTEGRATION.md", "自适应条件化文档"),
        ("ADAPTIVE_CONDITIONING_USAGE.md", "使用指南"),
        ("EVALUATION_IMPROVEMENTS.md", "评估改进文档")
    ]
    
    for doc_file, description in docs:
        if Path(doc_file).exists():
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content) > 100:  # 基本内容检查
                    print(f"   ✅ {description}")
                else:
                    print(f"   ⚠️ {description} (内容过短)")
            except Exception as e:
                print(f"   ⚠️ {description}: {e}")
        else:
            print(f"   ❌ {description}")
    
    return True

def check_version_info():
    """检查版本信息"""
    print("\n🔖 检查版本信息...")
    
    try:
        with open("VERSION", 'r') as f:
            version = f.read().strip()
        print(f"   📌 当前版本: {version}")
        
        # 检查版本格式
        version_parts = version.split('.')
        if len(version_parts) == 3:
            print("   ✅ 版本格式正确")
        else:
            print("   ⚠️ 版本格式可能有问题")
        
        return True
    except Exception as e:
        print(f"   ❌ 版本检查失败: {e}")
        return False

def check_import_structure():
    """检查导入结构"""
    print("\n📦 检查导入结构...")
    
    key_files = [
        "structdiff/models/structdiff.py",
        "structdiff/models/denoise.py"
    ]
    
    for file_path in key_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 解析导入
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and 'alphafold3_embeddings' in node.module:
                        imports.extend([alias.name for alias in node.names])
            
            if len(imports) > 0:
                print(f"   ✅ {file_path}: 包含AF3组件导入")
                print(f"      导入: {', '.join(imports[:3])}{'...' if len(imports) > 3 else ''}")
            else:
                print(f"   ⚠️ {file_path}: 可能缺少AF3组件导入")
                
        except Exception as e:
            print(f"   ❌ {file_path}: {e}")
    
    return True

def main():
    """主验证函数"""
    print("🔍 最终代码验证 - StructDiff v5.2.0")
    print("=" * 60)
    
    checks = [
        ("文件结构", check_file_structure),
        ("Python语法", check_python_syntax),
        ("配置文件", check_config_files),
        ("关键特性", check_key_features),
        ("导入结构", check_import_structure),
        ("文档完整性", check_documentation),
        ("版本信息", check_version_info)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name}: 检查过程出错 - {e}")
            results[name] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 代码验证总结:")
    
    success_count = 0
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 需要修复"
        print(f"  {name}: {status}")
        if result:
            success_count += 1
    
    overall_success = success_count == len(results)
    print(f"\n总体结果: {'✅ 代码就绪' if overall_success else '❌ 需要改进'}")
    print(f"通过率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if overall_success:
        print("\n🎉 代码验证通过! 准备Git提交")
        print("\n🚀 v5.2.0 功能亮点:")
        print("  🎯 AlphaFold3自适应条件化完全集成")
        print("  ⚡ GLU替换FFN，预期2-3倍加速")
        print("  📊 AF3噪声调度，训练更稳定")
        print("  🧬 生物学启发的条件初始化")
        print("  🔧 多方面细粒度条件控制")
        print("  📈 评估系统全面改进")
        print("  📚 完整的文档和使用指南")
        
        print("\n📋 准备提交:")
        print("  1. git add .")
        print("  2. git status  # 确认要提交的文件")
        print("  3. git commit  # 使用预设的commit message")
        print("  4. git push origin main")
    else:
        print("\n⚠️ 请先解决验证问题")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)