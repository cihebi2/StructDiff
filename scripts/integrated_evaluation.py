#!/usr/bin/env python3
"""
StructDiff集成评估脚本 - 整合CPL-Diff评估指标
支持现有evaluation_suite.py和新的lightweight_evaluation_suite.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查评估依赖的可用性"""
    deps = {}
    
    libraries = [
        'transformers', 'torch', 'Bio', 'scipy', 
        'matplotlib', 'seaborn', 'modlamp', 'numpy', 'pandas'
    ]
    
    for lib in libraries:
        try:
            __import__(lib)
            deps[lib] = True
        except ImportError:
            deps[lib] = False
    
    return deps

def run_enhanced_evaluation(generated_sequences: List[str], 
                          reference_sequences: Optional[List[str]] = None,
                          peptide_type: str = 'antimicrobial',
                          output_name: str = "integrated_evaluation"):
    """运行增强评估（需要完整依赖）"""
    try:
        from enhanced_evaluation_suite import EnhancedPeptideEvaluationSuite
        
        print("🎯 使用增强评估套件 (完整CPL-Diff指标)")
        evaluator = EnhancedPeptideEvaluationSuite()
        
        results = evaluator.comprehensive_evaluation(
            generated_sequences=generated_sequences,
            reference_sequences=reference_sequences,
            peptide_type=peptide_type
        )
        
        evaluator.generate_report(results, f"{output_name}_enhanced")
        return results, "enhanced"
        
    except ImportError as e:
        print(f"⚠️ 增强评估套件依赖缺失: {e}")
        return None, "enhanced_failed"

def run_lightweight_evaluation(generated_sequences: List[str], 
                             reference_sequences: Optional[List[str]] = None,
                             peptide_type: str = 'antimicrobial',
                             output_name: str = "integrated_evaluation"):
    """运行轻量级评估（纯Python实现）"""
    try:
        from lightweight_evaluation_suite import LightweightPeptideEvaluationSuite
        
        print("🔧 使用轻量级评估套件 (简化CPL-Diff指标)")
        evaluator = LightweightPeptideEvaluationSuite()
        
        results = evaluator.comprehensive_evaluation(
            generated_sequences=generated_sequences,
            reference_sequences=reference_sequences,
            peptide_type=peptide_type
        )
        
        evaluator.generate_report(results, f"{output_name}_lightweight")
        return results, "lightweight"
        
    except Exception as e:
        print(f"❌ 轻量级评估套件运行失败: {e}")
        return None, "lightweight_failed"

def run_original_evaluation(generated_sequences: List[str], 
                          reference_sequences: Optional[List[str]] = None,
                          peptide_type: str = 'antimicrobial',
                          output_name: str = "integrated_evaluation"):
    """运行原始评估套件"""
    try:
        from evaluation_suite import PeptideEvaluationSuite
        
        print("📊 使用原始评估套件 (StructDiff原有指标)")
        evaluator = PeptideEvaluationSuite()
        
        # 原始评估
        results = {}
        results['basic_quality'] = evaluator.evaluate_sequence_quality(generated_sequences)
        
        if reference_sequences:
            sequences_by_condition = {peptide_type: generated_sequences}
            results['condition_specificity'] = evaluator.evaluate_condition_specificity(sequences_by_condition)
            results['diversity'] = evaluator.evaluate_diversity(generated_sequences)
            results['novelty'] = evaluator.evaluate_novelty(generated_sequences, reference_sequences)
        
        # 保存结果
        evaluator.generate_report(results, f"{output_name}_original")
        return results, "original"
        
    except Exception as e:
        print(f"⚠️ 原始评估套件运行失败: {e}")
        return None, "original_failed"

def integrated_evaluation(generated_sequences: List[str], 
                        reference_sequences: Optional[List[str]] = None,
                        peptide_type: str = 'antimicrobial',
                        output_name: str = "integrated_evaluation",
                        prefer_enhanced: bool = True):
    """
    集成评估 - 自动选择最佳可用的评估方法
    
    Args:
        generated_sequences: 生成的多肽序列列表
        reference_sequences: 参考序列列表（可选）
        peptide_type: 多肽类型 ('antimicrobial', 'antifungal', 'antiviral')
        output_name: 输出文件名前缀
        prefer_enhanced: 是否优先使用增强评估
    
    Returns:
        评估结果字典和使用的方法
    """
    print("🚀 StructDiff集成评估启动")
    print("=" * 50)
    print(f"📊 生成序列数量: {len(generated_sequences)}")
    print(f"📚 参考序列数量: {len(reference_sequences) if reference_sequences else 0}")
    print(f"🏷️ 多肽类型: {peptide_type}")
    
    # 检查依赖
    deps = check_dependencies()
    missing_deps = [lib for lib, available in deps.items() if not available]
    
    print(f"\n🔍 依赖检查:")
    print(f"  可用库: {[lib for lib, available in deps.items() if available]}")
    if missing_deps:
        print(f"  缺失库: {missing_deps}")
    
    results_summary = {
        'metadata': {
            'peptide_type': peptide_type,
            'generated_count': len(generated_sequences),
            'reference_count': len(reference_sequences) if reference_sequences else 0,
            'available_dependencies': deps,
            'evaluation_methods_attempted': []
        },
        'evaluation_results': {}
    }
    
    success_methods = []
    
    # 1. 尝试增强评估（如果优先且依赖可用）
    if prefer_enhanced and all(deps.get(lib, False) for lib in ['transformers', 'torch', 'Bio']):
        print(f"\n🎯 尝试增强评估...")
        enhanced_results, method = run_enhanced_evaluation(
            generated_sequences, reference_sequences, peptide_type, output_name
        )
        results_summary['metadata']['evaluation_methods_attempted'].append(method)
        
        if enhanced_results:
            results_summary['evaluation_results']['enhanced'] = enhanced_results
            success_methods.append('enhanced')
            print("✅ 增强评估完成")
        else:
            print("❌ 增强评估失败")
    
    # 2. 尝试轻量级评估
    print(f"\n🔧 尝试轻量级评估...")
    lightweight_results, method = run_lightweight_evaluation(
        generated_sequences, reference_sequences, peptide_type, output_name
    )
    results_summary['metadata']['evaluation_methods_attempted'].append(method)
    
    if lightweight_results:
        results_summary['evaluation_results']['lightweight'] = lightweight_results
        success_methods.append('lightweight')
        print("✅ 轻量级评估完成")
    else:
        print("❌ 轻量级评估失败")
    
    # 3. 尝试原始评估（作为备选）
    print(f"\n📊 尝试原始评估...")
    original_results, method = run_original_evaluation(
        generated_sequences, reference_sequences, peptide_type, output_name
    )
    results_summary['metadata']['evaluation_methods_attempted'].append(method)
    
    if original_results:
        results_summary['evaluation_results']['original'] = original_results
        success_methods.append('original')
        print("✅ 原始评估完成")
    else:
        print("❌ 原始评估失败")
    
    # 生成集成报告
    if success_methods:
        print(f"\n📋 生成集成报告...")
        _generate_integrated_report(results_summary, output_name, success_methods)
        
        print(f"\n🎉 集成评估完成!")
        print(f"✅ 成功的评估方法: {success_methods}")
        print(f"📁 报告文件前缀: {output_name}")
        
        return results_summary, success_methods
    else:
        print(f"\n❌ 所有评估方法都失败了")
        return None, []

def _generate_integrated_report(results_summary: Dict, output_name: str, success_methods: List[str]):
    """生成集成报告"""
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # JSON报告
    json_path = output_dir / f"{output_name}_integrated.json"
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # 文本摘要
    summary_path = output_dir / f"{output_name}_integrated_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("StructDiff集成评估摘要\n")
        f.write("=" * 40 + "\n\n")
        
        # 元数据
        meta = results_summary['metadata']
        f.write("评估概况:\n")
        f.write(f"  肽类型: {meta.get('peptide_type', 'N/A')}\n")
        f.write(f"  生成序列数: {meta.get('generated_count', 0)}\n")
        f.write(f"  参考序列数: {meta.get('reference_count', 0)}\n")
        f.write(f"  成功评估方法: {', '.join(success_methods)}\n")
        f.write(f"  尝试的方法: {', '.join(meta.get('evaluation_methods_attempted', []))}\n\n")
        
        # 依赖状态
        deps = meta.get('available_dependencies', {})
        available_deps = [lib for lib, status in deps.items() if status]
        missing_deps = [lib for lib, status in deps.items() if not status]
        
        f.write("依赖状态:\n")
        f.write(f"  可用: {', '.join(available_deps) if available_deps else '无'}\n")
        f.write(f"  缺失: {', '.join(missing_deps) if missing_deps else '无'}\n\n")
        
        # 各方法的关键结果
        eval_results = results_summary.get('evaluation_results', {})
        
        for method in success_methods:
            if method in eval_results:
                f.write(f"{method.upper()}评估结果:\n")
                f.write("-" * 20 + "\n")
                
                method_results = eval_results[method]
                
                # 基础质量
                if 'basic_quality' in method_results:
                    bq = method_results['basic_quality']
                    f.write(f"  有效序列率: {bq.get('valid_sequences', 0)}/{bq.get('total_sequences', 0)}\n")
                    f.write(f"  平均长度: {bq.get('average_length', 0):.1f}\n")
                
                # CPL-Diff指标
                if 'information_entropy' in method_results:
                    ie = method_results['information_entropy']
                    if 'error' not in ie:
                        f.write(f"  信息熵: {ie.get('mean_entropy', 'N/A'):.3f}\n")
                
                if 'pseudo_perplexity' in method_results:
                    pp = method_results['pseudo_perplexity']
                    if 'error' not in pp:
                        f.write(f"  伪困惑度: {pp.get('mean_pseudo_perplexity', 'N/A'):.3f}\n")
                        f.write(f"  计算方法: {pp.get('method', 'N/A')}\n")
                
                # 相似性
                if 'similarity_analysis' in method_results:
                    sa = method_results['similarity_analysis']
                    if 'error' not in sa:
                        f.write(f"  新颖性比例: {sa.get('novelty_ratio', 'N/A'):.3f}\n")
                
                f.write("\n")
        
        f.write("说明:\n")
        f.write("- enhanced: 使用完整CPL-Diff指标（需要transformers, torch等）\n")
        f.write("- lightweight: 使用简化CPL-Diff指标（纯Python实现）\n")
        f.write("- original: 使用StructDiff原有指标\n")
    
    print(f"📝 集成报告已保存:")
    print(f"   - {json_path}")
    print(f"   - {summary_path}")

def main():
    """主函数 - 演示集成评估"""
    # 示例数据
    generated_sequences = [
        "KRWWKWIRWKK",
        "FRLKWFKRLLK", 
        "KLRFKKLRWFK",
        "GILDTILKILR",
        "KLAKLRWKLKL",
        "KWKLFKKIEK",
        "GLFDVIKKV",
        "RWWRRRWWRR",
        "KLKLLLLLKL",
        "AIKGKFAKFK"
    ]
    
    reference_sequences = [
        "MAGAININ1PEPTIDE",
        "KRWWKWIRWKK",
        "CECROPINPEPTIDE",
        "DEFENSINPEPTIDE",
        "MELITTINPEPTIDE"
    ]
    
    # 运行集成评估
    results, methods = integrated_evaluation(
        generated_sequences=generated_sequences,
        reference_sequences=reference_sequences,
        peptide_type='antimicrobial',
        output_name='demo_integrated',
        prefer_enhanced=True
    )
    
    if results:
        print(f"\n🎊 集成评估演示成功完成!")
        print(f"📊 使用的评估方法: {methods}")
    else:
        print(f"\n💥 集成评估演示失败")

if __name__ == "__main__":
    main()