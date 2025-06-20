# install_esmfold_deps.py
"""
安装 ESMFold 相关依赖的脚本
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """运行命令并处理错误"""
    print(f"正在{description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✓ {description}成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败:")
        print(f"  错误: {e}")
        print(f"  输出: {e.stdout}")
        print(f"  错误输出: {e.stderr}")
        return False

def install_accelerate():
    """安装 accelerate 库"""
    commands = [
        "pip install accelerate>=0.26.0",
        "pip install accelerate>=0.26.0 --upgrade"
    ]
    
    for cmd in commands:
        if run_command(cmd, "安装 accelerate"):
            return True
    
    print("⚠️ 尝试使用conda安装...")
    return run_command("conda install -c conda-forge accelerate", "conda安装 accelerate")

def install_transformers():
    """安装最新版本的 transformers"""
    commands = [
        "pip install transformers>=4.30.0",
        "pip install transformers --upgrade"
    ]
    
    for cmd in commands:
        if run_command(cmd, "安装 transformers"):
            return True
    return False

def install_fair_esm():
    """安装 fair-esm 库"""
    commands = [
        "pip install fair-esm",
        "pip install git+https://github.com/facebookresearch/esm.git"
    ]
    
    for cmd in commands:
        if run_command(cmd, "安装 fair-esm"):
            return True
    return False

def check_installation():
    """检查安装是否成功"""
    print("\n检查安装状态...")
    
    packages = [
        ("accelerate", "import accelerate; print(f'accelerate {accelerate.__version__}')"),
        ("transformers", "import transformers; print(f'transformers {transformers.__version__}')"),
        ("fair-esm", "import esm; print('fair-esm installed')"),
        ("torch", "import torch; print(f'torch {torch.__version__}')"),
    ]
    
    success_count = 0
    for package, test_cmd in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_cmd],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ {package}: {result.stdout.strip()}")
            success_count += 1
        except subprocess.CalledProcessError:
            print(f"❌ {package}: 未正确安装")
    
    return success_count == len(packages)

def main():
    """主安装函数"""
    print("=== ESMFold 依赖安装脚本 ===\n")
    
    print("Python 版本:", sys.version)
    print("Python 路径:", sys.executable)
    print()
    
    # 安装各个包
    success = True
    
    # 1. 安装 accelerate (最重要)
    print("1. 安装 accelerate...")
    if not install_accelerate():
        print("❌ accelerate 安装失败，这可能导致 ESMFold 无法加载")
        success = False
    
    # 2. 安装 transformers
    print("\n2. 安装 transformers...")
    if not install_transformers():
        print("❌ transformers 安装失败")
        success = False
    
    # 3. 安装 fair-esm
    print("\n3. 安装 fair-esm...")
    if not install_fair_esm():
        print("❌ fair-esm 安装失败")
        success = False
    
    # 4. 检查安装
    print("\n4. 验证安装...")
    if check_installation():
        print("\n🎉 所有依赖安装成功！")
    else:
        print("\n⚠️ 部分依赖安装失败，请检查错误信息")
        success = False
    
    # 5. 安装后建议
    print("\n安装完成后的建议:")
    print("1. 重启Python解释器或Jupyter内核")
    print("2. 运行测试脚本: python load_esmfold.py")
    print("3. 如果仍有问题，尝试创建新的conda环境")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 