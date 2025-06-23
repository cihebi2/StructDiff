#!/usr/bin/env python3
"""
CPL-Diff标准评估演示脚本
展示如何使用标准评估套件评估生成的肽序列
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.cpldiff_standard_evaluation import CPLDiffStandardEvaluator

def main():
    """CPL-Diff标准评估演示"""
    print("🚀 CPL-Diff标准评估演示")
    print("=" * 50)
    
    # 创建评估器
    evaluator = CPLDiffStandardEvaluator(output_dir="evaluation_results")
    
    # 示例生成序列（抗菌肽）
    generated_sequences = [
        "KRWWKWIRWKK",     # 经典抗菌肽模式
        "FRLKWFKRLLK",     # 高疏水性
        "KLRFKKLRWFK",     # 平衡电荷和疏水性
        "GILDTILKILR",     # 较长序列
        "KLAKLRWKLKL",     # 重复模式
        "KWKLFKKIEK",      # 短序列
        "GLFDVIKKV",       # 中等长度
        "RWWRRRWWRR",      # 高芳香性
        "KLKLLLLLKL",      # 重复亮氨酸
        "AIKGKFAKFK"       # 经典AMP序列
    ]
    
    # 参考序列（已知抗菌肽）
    reference_sequences = [
        "MAGAININ",        # 经典天然抗菌肽
        "KRWWKWIRWKK",     # 与生成序列重复，测试相似性
        "CECROPIN",        # 天然抗菌肽
        "DEFENSIN",        # 防御素类
        "MELITTIN",        # 蜂毒素
        "BOMBININ"         # 铃蟾肽
    ]
    
    print(f"📊 生成序列数量: {len(generated_sequences)}")
    print(f"📚 参考序列数量: {len(reference_sequences)}")
    print(f"🧬 序列示例: {generated_sequences[0]}, {generated_sequences[1]}, ...")
    print()
    
    # 运行CPL-Diff标准综合评估
    print("🔬 开始CPL-Diff标准评估...")
    results = evaluator.comprehensive_cpldiff_evaluation(
        generated_sequences=generated_sequences,
        reference_sequences=reference_sequences,
        peptide_type='antimicrobial'
    )
    
    # 生成报告
    print("\n📊 生成评估报告...")
    evaluator.generate_cpldiff_report(results, "demo_cpldiff_standard")
    
    # 显示核心结果摘要
    print("\n✨ CPL-Diff核心指标摘要:")
    print("-" * 40)
    
    core_metrics = results.get('cpldiff_core_metrics', {})
    
    # 1. Perplexity ↓
    if 'pseudo_perplexity' in core_metrics:
        pp = core_metrics['pseudo_perplexity']
        if 'error' not in pp:
            print(f"1. Perplexity ↓: {pp.get('mean_pseudo_perplexity', 'N/A'):.3f}±{pp.get('std_pseudo_perplexity', 0):.3f}")
        else:
            print(f"1. Perplexity ↓: {pp.get('error', 'N/A')}")
    
    # 2. pLDDT ↑
    if 'plddt' in core_metrics:
        plddt = core_metrics['plddt']
        if 'error' not in plddt:
            print(f"2. pLDDT ↑: {plddt.get('mean_plddt', 'N/A'):.2f}±{plddt.get('std_plddt', 0):.2f}")
        else:
            print(f"2. pLDDT ↑: {plddt.get('error', 'N/A')}")
    
    # 3. Instability ↓
    if 'instability' in core_metrics:
        inst = core_metrics['instability']
        if 'error' not in inst:
            print(f"3. Instability ↓: {inst.get('mean_instability', 'N/A'):.2f}±{inst.get('std_instability', 0):.2f}")
        else:
            print(f"3. Instability ↓: {inst.get('error', 'N/A')}")
    
    # 4. Similarity ↓
    if 'similarity' in core_metrics:
        sim = core_metrics['similarity']
        if 'error' not in sim:
            print(f"4. Similarity ↓: {sim.get('mean_similarity', 'N/A'):.2f}±{sim.get('std_similarity', 0):.2f}")
        else:
            print(f"4. Similarity ↓: {sim.get('error', 'N/A')}")
    
    # 5. Activity ↑
    if 'activity' in core_metrics:
        act = core_metrics['activity']
        if 'error' not in act:
            print(f"5. Activity ↑: {act.get('activity_ratio', 'N/A'):.3f}")
            print(f"   活性序列: {act.get('active_sequences', 0)}/{act.get('total_sequences', 0)}")
        else:
            print(f"5. Activity ↑: {act.get('error', 'N/A')}")
    
    print("\n📄 详细报告文件:")
    print("  - demo_cpldiff_standard.json")
    print("  - demo_cpldiff_standard_summary.txt")
    
    print("\n✅ CPL-Diff标准评估演示完成!")
    
    # 显示依赖状态
    if 'metadata' in results and 'available_dependencies' in results['metadata']:
        deps = results['metadata']['available_dependencies']
        print(f"\n🔧 依赖库状态:")
        available = [k for k, v in deps.items() if v]
        missing = [k for k, v in deps.items() if not v]
        if available:
            print(f"  ✅ 可用: {', '.join(available)}")
        if missing:
            print(f"  ❌ 缺失: {', '.join(missing)}")
        
        if missing:
            print("\n💡 提示: 安装缺失依赖可获得更精确的评估:")
            if 'esm2' in [k for k, v in deps.items() if not v]:
                print("  - pip install transformers torch  # ESM-2伪困惑度")
            if 'biopython' in [k for k, v in deps.items() if not v]:
                print("  - pip install biopython  # BLOSUM62相似性")
            if 'modlamp' in [k for k, v in deps.items() if not v]:
                print("  - pip install modlamp  # 不稳定性指数")

if __name__ == "__main__":
    main()