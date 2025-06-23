#!/usr/bin/env python3
"""
安装增强评估套件的依赖包
"""

import subprocess
import sys
import os

def install_package(package_name, pip_name=None):
    """安装Python包"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        __import__(package_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"📦 正在安装 {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {pip_name} 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {pip_name} 安装失败: {e}")
            return False

def main():
    """主安装流程"""
    print("🔧 开始安装增强评估套件依赖...")
    print("=" * 50)
    
    # 核心依赖
    core_packages = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("Bio", "biopython"), 
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("modlamp", "modlamp")
    ]
    
    failed_packages = []
    
    for package, pip_name in core_packages:
        if not install_package(package, pip_name):
            failed_packages.append(pip_name)
    
    print("\n" + "=" * 50)
    print("📋 安装总结:")
    
    if failed_packages:
        print(f"❌ 以下包安装失败: {', '.join(failed_packages)}")
        print("\n💡 解决方案:")
        
        if "modlamp" in failed_packages:
            print("   modlamp安装问题:")
            print("   1. 可能需要先安装: pip install numpy scipy")
            print("   2. 或使用conda: conda install -c bioconda modlamp")
            print("   3. 如果仍失败，评估套件会跳过不稳定性指数计算")
        
        if "torch" in failed_packages:
            print("   PyTorch安装问题:")
            print("   1. 访问 https://pytorch.org 获取适合您系统的安装命令")
            print("   2. 选择合适的CUDA版本（如果有GPU）")
        
        if "biopython" in failed_packages:
            print("   BioPython安装问题:")
            print("   1. 尝试: pip install biopython --no-cache-dir")
            print("   2. 或使用conda: conda install -c bioconda biopython")
        
        print(f"\n⚠️ 注意: 即使某些包安装失败，评估套件仍可运行，但会跳过相关功能")
    else:
        print("✅ 所有依赖包安装成功!")
    
    print("\n🎯 下一步:")
    print("   运行测试: python scripts/enhanced_evaluation_suite.py")

if __name__ == "__main__":
    main()