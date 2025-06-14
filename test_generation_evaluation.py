#!/usr/bin/env python3
"""
测试生成和评估功能
"""

import os
import sys
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.train_peptide_esmfold import generate_and_validate

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_generation_and_evaluation():
    """测试生成和评估功能"""
    logger.info("🧪 开始测试生成和评估功能...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载配置
    config_path = "configs/peptide_esmfold_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    config = OmegaConf.load(config_path)
    logger.info(f"✅ 配置加载成功: {config_path}")
    
    # 修改配置以适应测试
    config.data.use_predicted_structures = False  # 简化测试
    
    try:
        # 运行生成和验证
        logger.info("🚀 开始运行生成和验证...")
        generate_and_validate(config, device)
        logger.info("🎉 生成和验证测试完成！")
        
    except Exception as e:
        logger.error(f"❌ 生成和验证测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation_and_evaluation() 