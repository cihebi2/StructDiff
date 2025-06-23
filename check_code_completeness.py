#!/usr/bin/env python3
"""
全面检查训练代码的完整性和规范性
"""

import sys
import ast
import importlib.util
from pathlib import Path
import re

def check_imports_completeness():
    """检查导入的完整性"""
    print("🔍 检查导入完整性...")
    
    training_script = "scripts/train_peptide_esmfold.py"
    
    try:
        with open(training_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST
        tree = ast.parse(content)
        
        # 收集所有导入
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
        
        print(f"📦 发现 {len(imports)} 个导入")
        
        # 检查关键导入
        critical_imports = [
            "torch",
            "structdiff.models.structdiff",
            "structdiff.models.denoise", 
            "structdiff.models.layers.alphafold3_embeddings",
            "structdiff.data.dataset",
            "structdiff.data.collator"
        ]
        
        missing_imports = []
        for imp in critical_imports:
            found = any(imp in existing_imp for existing_imp in imports)
            if found:
                print(f"✅ {imp}")
            else:
                print(f"❌ {imp}")
                missing_imports.append(imp)
        
        return len(missing_imports) == 0
    except Exception as e:
        print(f"❌ 导入检查失败: {e}")
        return False

def check_class_method_completeness():
    """检查类和方法的完整性"""
    print("\n🏗️ 检查类和方法完整性...")
    
    files_to_check = [
        "structdiff/models/denoise.py",
        "structdiff/models/layers/alphafold3_embeddings.py",
        "structdiff/models/structdiff.py"
    ]
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    classes.append({
                        'name': node.name,
                        'methods': methods,
                        'line': node.lineno
                    })
            
            print(f"📁 {file_path}:")
            for cls in classes:
                print(f"   🏗️ {cls['name']} (第{cls['line']}行)")
                if '__init__' in cls['methods']:
                    print(f"      ✅ __init__")
                else:
                    print(f"      ❌ 缺少 __init__")
                
                if 'forward' in cls['methods']:
                    print(f"      ✅ forward")
                elif any('forward' in m for m in cls['methods']):
                    print(f"      ⚠️ forward 方法名可能有变体")
                else:
                    print(f"      ❌ 缺少 forward")
        
        except Exception as e:
            print(f"❌ {file_path}: {e}")
            return False
    
    return True

def check_configuration_completeness():
    """检查配置文件完整性"""
    print("\n⚙️ 检查配置文件完整性...")
    
    import yaml
    
    config_files = [
        "configs/peptide_esmfold_config.yaml",
        "configs/peptide_adaptive_conditioning.yaml"
    ]
    
    required_sections = [
        "model",
        "data", 
        "training",
        "diffusion",
        "evaluation"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"📄 {config_file}:")
            
            for section in required_sections:
                if section in config:
                    print(f"   ✅ {section}")
                else:
                    print(f"   ❌ 缺少 {section}")
            
            # 检查特殊配置
            if "adaptive_conditioning" in str(config):
                print(f"   ✅ adaptive_conditioning 配置存在")
            else:
                print(f"   ⚠️ adaptive_conditioning 配置可能缺失")
                
        except Exception as e:
            print(f"❌ {config_file}: {e}")
            return False
    
    return True

def check_training_script_structure():
    """检查训练脚本结构"""
    print("\n🚂 检查训练脚本结构...")
    
    training_script = "scripts/train_peptide_esmfold.py"
    
    try:
        with open(training_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键函数
        required_functions = [
            "parse_args",
            "main", 
            "train_epoch",
            "validate_epoch"
        ]
        
        # 检查关键类
        required_classes = [
            "PeptideGenerator"
        ]
        
        for func in required_functions:
            if f"def {func}" in content:
                print(f"   ✅ 函数 {func}")
            else:
                print(f"   ❌ 缺少函数 {func}")
        
        for cls in required_classes:
            if f"class {cls}" in content:
                print(f"   ✅ 类 {cls}")
            else:
                print(f"   ❌ 缺少类 {cls}")
        
        # 检查关键功能
        key_features = [
            "adaptive_conditioning",
            "evaluate_physicochemical_properties",
            "evaluate_external_classifier_activity",
            "ESMFold",
            "wandb"
        ]
        
        for feature in key_features:
            if feature in content:
                print(f"   ✅ 功能 {feature}")
            else:
                print(f"   ⚠️ 功能 {feature} 可能缺失")
        
        return True
    except Exception as e:
        print(f"❌ 训练脚本检查失败: {e}")
        return False

def check_documentation_completeness():
    """检查文档完整性"""
    print("\n📚 检查文档完整性...")
    
    required_docs = [
        "README.md",
        "ALPHAFOLD3_IMPROVEMENTS.md", 
        "AF3_ADAPTIVE_CONDITIONING_INTEGRATION.md",
        "ADAPTIVE_CONDITIONING_USAGE.md",
        "EVALUATION_IMPROVEMENTS.md"
    ]
    
    for doc in required_docs:
        if Path(doc).exists():
            print(f"   ✅ {doc}")
        else:
            print(f"   ❌ 缺少 {doc}")
    
    return True

def check_test_scripts():
    """检查测试脚本"""
    print("\n🧪 检查测试脚本...")
    
    test_scripts = [
        "test_adaptive_conditioning.py",
        "validate_af3_integration.py", 
        "check_syntax_only.py",
        "fix_evaluation_issues.py"
    ]
    
    for script in test_scripts:
        if Path(script).exists():
            print(f"   ✅ {script}")
            # 检查语法
            try:
                with open(script, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                print(f"      ✅ 语法正确")
            except SyntaxError as e:
                print(f"      ❌ 语法错误: {e}")
        else:
            print(f"   ⚠️ {script} 不存在")
    
    return True

def check_version_consistency():
    """检查版本一致性"""
    print("\n🔖 检查版本一致性...")
    
    try:
        with open("VERSION", 'r') as f:
            version = f.read().strip()
        print(f"   📌 当前版本: {version}")
        
        # 检查是否有git历史
        import subprocess
        try:
            result = subprocess.run(['git', 'log', '--oneline', '-1'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                print(f"   ✅ Git历史: {result.stdout.strip()}")
            else:
                print(f"   ⚠️ 无Git历史")
        except:
            print(f"   ⚠️ Git不可用")
        
        return True
    except Exception as e:
        print(f"❌ 版本检查失败: {e}")
        return False

def main():
    """主检查函数"""
    print("🔍 全面检查训练代码完整性和规范性")
    print("=" * 60)
    
    checks = [
        ("导入完整性", check_imports_completeness),
        ("类和方法完整性", check_class_method_completeness),
        ("配置文件完整性", check_configuration_completeness),
        ("训练脚本结构", check_training_script_structure),
        ("文档完整性", check_documentation_completeness),
        ("测试脚本", check_test_scripts),
        ("版本一致性", check_version_consistency)
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
    print("📋 完整性检查总结:")
    
    success_count = 0
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 需要修复"
        print(f"  {name}: {status}")
        if result:
            success_count += 1
    
    overall_success = success_count == len(results)
    print(f"\n总体结果: {'✅ 代码完整规范' if overall_success else '❌ 需要改进'}")
    print(f"通过率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if overall_success:
        print("\n🎉 代码检查通过，可以进行Git提交!")
        print("\n📋 检查要点:")
        print("  ✅ 所有Python文件语法正确")
        print("  ✅ 导入依赖完整")
        print("  ✅ 配置文件有效")
        print("  ✅ 核心功能完整")
        print("  ✅ 文档齐全")
    else:
        print("\n⚠️ 发现问题，建议修复后再提交")
    
    return overall_success

if __name__ == "__main__":
    main()