#!/usr/bin/env python3
"""
修复评估模块的关键问题
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def fix_modlamp_dependency():
    """修复modlamp依赖问题"""
    print("🔧 修复modlamp依赖问题...")
    
    training_script = "scripts/train_peptide_esmfold.py"
    
    # 读取文件
    with open(training_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复方案1: 添加简单的理化性质计算替代
    fallback_code = '''
def compute_simple_physicochemical_properties(sequences):
    """
    简单的理化性质计算（不依赖modlamp）
    """
    from collections import Counter
    
    # 氨基酸属性表
    aa_properties = {
        # 电荷 (pH=7.4)
        'charge': {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1},
        # 疏水性 (Eisenberg scale)
        'hydrophobicity': {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        },
        # 等电点贡献
        'isoelectric': {
            'D': -1, 'E': -1, 'R': 1, 'K': 1, 'H': 0.5,
            'C': 0.3, 'Y': 0.3  # 简化
        }
    }
    
    # 芳香性氨基酸
    aromatic_aa = set('FWY')
    
    properties = {
        'charge': {'mean_charge': 0.0, 'std_charge': 0.0},
        'isoelectric_point': {'mean_isoelectric_point': 0.0, 'std_isoelectric_point': 0.0}, 
        'hydrophobicity': {'mean_hydrophobicity': 0.0, 'std_hydrophobicity': 0.0},
        'aromaticity': {'mean_aromaticity': 0.0, 'std_aromaticity': 0.0}
    }
    
    if not sequences:
        return properties
    
    # 计算各项属性
    charges = []
    hydrophobicities = []
    isoelectric_points = []
    aromaticities = []
    
    for seq in sequences:
        # 净电荷
        charge = sum(aa_properties['charge'].get(aa, 0) for aa in seq)
        charges.append(charge)
        
        # 平均疏水性
        hydro = [aa_properties['hydrophobicity'].get(aa, 0) for aa in seq]
        avg_hydro = sum(hydro) / len(hydro) if hydro else 0
        hydrophobicities.append(avg_hydro)
        
        # 简化等电点估算
        basic_count = sum(1 for aa in seq if aa in 'RKH')
        acidic_count = sum(1 for aa in seq if aa in 'DE')
        if basic_count > acidic_count:
            iep = 8.5 + basic_count * 0.5  # 碱性
        elif acidic_count > basic_count:
            iep = 6.0 - acidic_count * 0.3  # 酸性  
        else:
            iep = 7.0  # 中性
        isoelectric_points.append(max(3.0, min(11.0, iep)))  # 限制在合理范围
        
        # 芳香性
        aromatic_ratio = sum(1 for aa in seq if aa in aromatic_aa) / len(seq)
        aromaticities.append(aromatic_ratio)
    
    # 计算统计值
    properties['charge']['mean_charge'] = sum(charges) / len(charges)
    properties['charge']['std_charge'] = (
        sum((c - properties['charge']['mean_charge'])**2 for c in charges) / len(charges)
    )**0.5 if len(charges) > 1 else 0.0
    
    properties['hydrophobicity']['mean_hydrophobicity'] = sum(hydrophobicities) / len(hydrophobicities)
    properties['hydrophobicity']['std_hydrophobicity'] = (
        sum((h - properties['hydrophobicity']['mean_hydrophobicity'])**2 for h in hydrophobicities) / len(hydrophobicities)
    )**0.5 if len(hydrophobicities) > 1 else 0.0
    
    properties['isoelectric_point']['mean_isoelectric_point'] = sum(isoelectric_points) / len(isoelectric_points)
    properties['isoelectric_point']['std_isoelectric_point'] = (
        sum((i - properties['isoelectric_point']['mean_isoelectric_point'])**2 for i in isoelectric_points) / len(isoelectric_points)
    )**0.5 if len(isoelectric_points) > 1 else 0.0
    
    properties['aromaticity']['mean_aromaticity'] = sum(aromaticities) / len(aromaticities)
    properties['aromaticity']['std_aromaticity'] = (
        sum((a - properties['aromaticity']['mean_aromaticity'])**2 for a in aromaticities) / len(aromaticities)
    )**0.5 if len(aromaticities) > 1 else 0.0
    
    return properties
'''
    
    # 查找插入位置 (在类定义之前)
    class_def_pos = content.find("class PeptideGenerator:")
    if class_def_pos == -1:
        print("❌ 找不到PeptideGenerator类定义")
        return False
    
    # 插入新函数
    new_content = content[:class_def_pos] + fallback_code + "\n\n" + content[class_def_pos:]
    
    # 修改evaluate_physicochemical_properties方法
    old_method = """        if not MODLAMP_AVAILABLE:
            logger.warning("⚠️ modlamp未安装，跳过理化性质计算")
            return {
                'mean_charge': 0.0, 'mean_isoelectric_point': 0.0,
                'mean_hydrophobicity': 0.0, 'mean_aromaticity': 0.0
            }"""
    
    new_method = """        if not MODLAMP_AVAILABLE:
            logger.warning("⚠️ modlamp未安装，使用简化的理化性质计算")
            return compute_simple_physicochemical_properties(sequences)"""
    
    new_content = new_content.replace(old_method, new_method)
    
    # 写回文件
    with open(training_script, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ modlamp依赖问题已修复")
    return True

def fix_memory_management():
    """改进内存管理，避免ESMFold OOM"""
    print("🔧 改进内存管理...")
    
    config_file = "configs/peptide_esmfold_config.yaml"
    
    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加内存优化配置
    memory_config = """
# 内存优化配置 (v5.2.0新增)
memory_optimization:
  # 评估阶段禁用ESMFold避免OOM
  disable_esmfold_in_eval: true
  # 生成时的批次大小
  generation_batch_size: 4
  # 清理频率
  cleanup_frequency: 50
  # 使用梯度检查点
  gradient_checkpointing: true
"""
    
    # 在文件末尾添加
    new_content = content.rstrip() + memory_config
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 内存管理配置已优化")
    return True

def create_evaluation_fix_script():
    """创建评估修复的快速脚本"""
    fix_script_content = '''#!/usr/bin/env python3
"""
快速修复评估问题的脚本
"""

import torch
import gc
import os

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print(f"🧹 GPU内存已清理，当前使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

def set_environment_for_eval():
    """设置环境变量优化评估"""
    # PyTorch内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 禁用一些调试功能
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    print("✅ 环境变量已优化")

if __name__ == "__main__":
    print("🔧 运行评估修复...")
    set_environment_for_eval()
    clear_gpu_memory()
    print("✅ 评估环境已优化，可以重新运行训练脚本")
'''
    
    with open("fix_eval_environment.py", 'w') as f:
        f.write(fix_script_content)
    
    print("✅ 评估修复脚本已创建: fix_eval_environment.py")
    return True

def fix_instability_index_display():
    """修复不稳定性指数显示问题"""
    print("🔧 修复不稳定性指数显示...")
    
    training_script = "scripts/train_peptide_esmfold.py"
    
    # 读取文件
    with open(training_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复表格显示中的不稳定性指数
    old_line = "            instability = results.get('instability_index', {}).get('mean_instability', 0.0)"
    new_line = """            instability = results.get('instability_index', {}).get('mean_instability', 0.0)
            if instability == 0.0:  # 备用键名
                instability = results.get('instability_index', {}).get('mean', 0.0)
                if instability == 0.0:  # 再次备用
                    instability_data = results.get('instability_index', {})
                    if isinstance(instability_data, dict) and 'mean_instability_index' in instability_data:
                        instability = instability_data['mean_instability_index']"""
    
    new_content = content.replace(old_line, new_line)
    
    # 写回文件
    with open(training_script, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 不稳定性指数显示已修复")
    return True

def create_improved_classifier():
    """创建改进的外部分类器"""
    print("🔧 创建改进的外部分类器...")
    
    # 检查外部分类器文件
    classifier_file = "structdiff/utils/external_classifiers.py"
    
    if not os.path.exists(classifier_file):
        print("❌ 外部分类器文件不存在")
        return False
    
    # 读取现有文件
    with open(classifier_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加改进的分类逻辑
    improved_logic = '''
    def _improved_antimicrobial_rules(self, seq):
        """改进的抗菌肽识别规则"""
        # 基本特征
        length = len(seq)
        positive_aa = sum(1 for aa in seq if aa in 'RKH')
        hydrophobic_aa = sum(1 for aa in seq if aa in 'AILMFVWY')
        
        # 规则1: 长度在8-50之间
        if not (8 <= length <= 50):
            return 0.1
        
        # 规则2: 正电荷氨基酸比例
        positive_ratio = positive_aa / length
        if positive_ratio < 0.15:  # 至少15%正电荷
            return 0.2
        
        # 规则3: 疏水性氨基酸比例
        hydrophobic_ratio = hydrophobic_aa / length
        if not (0.3 <= hydrophobic_ratio <= 0.7):  # 疏水性在30-70%
            return 0.3
        
        # 规则4: 两亲性 (简化检测)
        if positive_ratio > 0.25 and hydrophobic_ratio > 0.4:
            return 0.8  # 高置信度
        elif positive_ratio > 0.2 and hydrophobic_ratio > 0.35:
            return 0.6  # 中等置信度
        else:
            return 0.4  # 低置信度
'''
    
    # 查找替换位置
    if "_simple_antimicrobial_rules" in content:
        # 替换现有方法
        import re
        pattern = r'def _simple_antimicrobial_rules\(self, seq\):.*?return [0-9.]+.*?(?=\n    def|\n\nclass|\nclass|$)'
        new_content = re.sub(pattern, improved_logic.strip(), content, flags=re.DOTALL)
        
        with open(classifier_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ 外部分类器已改进")
        return True
    else:
        print("⚠️ 未找到分类器方法，跳过改进")
        return False

def main():
    """主修复函数"""
    print("🛠️ 开始修复评估问题...\n")
    
    fixes = [
        ("理化性质计算", fix_modlamp_dependency),
        ("内存管理", fix_memory_management), 
        ("不稳定性指数显示", fix_instability_index_display),
        ("环境修复脚本", create_evaluation_fix_script),
        ("外部分类器", create_improved_classifier),
    ]
    
    success_count = 0
    for name, fix_func in fixes:
        try:
            print(f"🔧 正在修复: {name}")
            if fix_func():
                print(f"✅ {name} 修复成功")
                success_count += 1
            else:
                print(f"⚠️ {name} 修复部分成功")
        except Exception as e:
            print(f"❌ {name} 修复失败: {e}")
    
    print(f"\n📊 修复总结: {success_count}/{len(fixes)} 个问题已修复")
    
    if success_count >= 3:
        print("\n🎉 主要问题已修复！建议操作:")
        print("1. 运行: python3 fix_eval_environment.py")
        print("2. 重新训练: python3 scripts/train_peptide_esmfold.py")
        print("3. 观察理化性质是否有实际数值")
    else:
        print("\n⚠️ 还有问题需要手动解决，请检查错误信息")

if __name__ == "__main__":
    main()