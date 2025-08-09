#!/usr/bin/env python3
"""
结构特征预计算脚本
将所有训练数据的结构特征提前计算并缓存，避免训练时的ESMFold计算瓶颈
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import hashlib
from tqdm import tqdm
import logging
from pathlib import Path
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# 添加项目路径
sys.path.append('/home/qlyu/sequence/StructDiff-7.0.0')

from structdiff.models.esmfold_wrapper import ESMFoldWrapper

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('structure_precompute.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StructureFeatureCache:
    """结构特征缓存管理器"""
    
    def __init__(self, cache_dir="./structure_cache", max_memory_gb=20):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_gb = max_memory_gb
        self.memory_cache = {}  # 内存缓存
        self.disk_cache_index = {}  # 磁盘缓存索引
        self.load_disk_index()
        
    def get_sequence_hash(self, sequence):
        """获取序列的哈希值作为缓存键"""
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def load_disk_index(self):
        """加载磁盘缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.disk_cache_index = json.load(f)
            logger.info(f"加载磁盘缓存索引: {len(self.disk_cache_index)} 个条目")
        else:
            self.disk_cache_index = {}
    
    def save_disk_index(self):
        """保存磁盘缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.disk_cache_index, f, indent=2)
    
    def get_cache_file_path(self, seq_hash):
        """获取缓存文件路径"""
        # 使用哈希的前两个字符作为子目录，避免单个目录文件过多
        subdir = seq_hash[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{seq_hash}.pkl"
    
    def has_cached_structure(self, sequence):
        """检查是否已缓存结构特征"""
        seq_hash = self.get_sequence_hash(sequence)
        
        # 检查内存缓存
        if seq_hash in self.memory_cache:
            return True
        
        # 检查磁盘缓存
        if seq_hash in self.disk_cache_index:
            cache_file = self.get_cache_file_path(seq_hash)
            return cache_file.exists()
        
        return False
    
    def get_cached_structure(self, sequence):
        """获取缓存的结构特征"""
        seq_hash = self.get_sequence_hash(sequence)
        
        # 首先检查内存缓存
        if seq_hash in self.memory_cache:
            return self.memory_cache[seq_hash]
        
        # 然后检查磁盘缓存
        if seq_hash in self.disk_cache_index:
            cache_file = self.get_cache_file_path(seq_hash)
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        structure_features = pickle.load(f)
                    
                    # 加载到内存缓存（如果内存允许）
                    self.add_to_memory_cache(seq_hash, structure_features)
                    return structure_features
                except Exception as e:
                    logger.warning(f"加载缓存文件失败 {cache_file}: {e}")
                    # 删除损坏的缓存文件
                    cache_file.unlink(missing_ok=True)
                    del self.disk_cache_index[seq_hash]
        
        return None
    
    def cache_structure(self, sequence, structure_features):
        """缓存结构特征"""
        seq_hash = self.get_sequence_hash(sequence)
        
        # 保存到磁盘
        cache_file = self.get_cache_file_path(seq_hash)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(structure_features, f)
            
            # 更新索引
            self.disk_cache_index[seq_hash] = {
                'sequence_length': len(sequence),
                'cache_file': str(cache_file.relative_to(self.cache_dir)),
                'timestamp': time.time()
            }
            
            # 加载到内存缓存
            self.add_to_memory_cache(seq_hash, structure_features)
            
            return True
        except Exception as e:
            logger.error(f"缓存结构特征失败: {e}")
            return False
    
    def add_to_memory_cache(self, seq_hash, structure_features):
        """添加到内存缓存"""
        # 简单的内存管理：如果超过限制，清空内存缓存
        current_memory_gb = self.estimate_memory_usage()
        if current_memory_gb > self.max_memory_gb:
            self.clear_memory_cache()
        
        self.memory_cache[seq_hash] = structure_features
    
    def estimate_memory_usage(self):
        """估计当前内存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0
    
    def clear_memory_cache(self):
        """清空内存缓存"""
        self.memory_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        return {
            'disk_cache_count': len(self.disk_cache_index),
            'memory_cache_count': len(self.memory_cache),
            'cache_dir_size_mb': sum(f.stat().st_size for f in self.cache_dir.rglob('*.pkl')) / 1e6
        }

def precompute_single_structure(esmfold_wrapper, sequence, seq_id, cache_manager):
    """预计算单个序列的结构特征"""
    try:
        # 检查是否已缓存
        if cache_manager.has_cached_structure(sequence):
            return seq_id, "cached", None
        
        # 预测结构
        start_time = time.time()
        structure_features = esmfold_wrapper.predict_structure(sequence)
        prediction_time = time.time() - start_time
        
        if structure_features is not None:
            # 缓存结构特征
            success = cache_manager.cache_structure(sequence, structure_features)
            if success:
                return seq_id, "computed", prediction_time
            else:
                return seq_id, "cache_failed", prediction_time
        else:
            return seq_id, "prediction_failed", prediction_time
            
    except Exception as e:
        logger.error(f"预计算序列 {seq_id} 失败: {e}")
        return seq_id, "error", str(e)

def precompute_batch_structures(esmfold_wrapper, sequences_batch, cache_manager):
    """批量预计算结构特征"""
    results = []
    
    for seq_id, sequence in sequences_batch:
        result = precompute_single_structure(esmfold_wrapper, sequence, seq_id, cache_manager)
        results.append(result)
        
        # 定期清理内存
        if len(results) % 10 == 0:
            cache_manager.clear_memory_cache()
    
    return results

def load_dataset_sequences(data_path):
    """加载数据集序列"""
    logger.info(f"加载数据集: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        sequences = []
        
        for idx, row in df.iterrows():
            if 'sequence' in row and pd.notna(row['sequence']):
                sequences.append((idx, row['sequence']))
            elif 'peptide_sequence' in row and pd.notna(row['peptide_sequence']):
                sequences.append((idx, row['peptide_sequence']))
        
        logger.info(f"加载了 {len(sequences)} 个序列")
        return sequences
        
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return []

def precompute_all_structures():
    """预计算所有结构特征"""
    logger.info("🚀 开始结构特征预计算...")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，无法运行ESMFold")
        return False
    
    # 选择GPU设备
    device = torch.device('cuda:2')  # 使用GPU 2
    logger.info(f"🎯 使用设备: {device}")
    
    # 创建缓存管理器
    cache_manager = StructureFeatureCache(
        cache_dir="./structure_cache",
        max_memory_gb=15  # 为ESMFold预留足够内存
    )
    
    # 显示缓存统计
    stats = cache_manager.get_cache_stats()
    logger.info(f"📊 缓存统计: {stats}")
    
    try:
        # 初始化ESMFold
        logger.info("🔄 初始化ESMFold...")
        esmfold_wrapper = ESMFoldWrapper(device=device)
        
        if not esmfold_wrapper.available:
            logger.error("❌ ESMFold初始化失败")
            return False
        
        logger.info("✅ ESMFold初始化成功")
        logger.info(f"ESMFold内存使用: {torch.cuda.memory_allocated(device) / 1e9:.1f}GB")
        
        # 加载数据集
        datasets = [
            "/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
            "/home/qlyu/sequence/StructDiff-7.0.0/data/processed/val.csv"
        ]
        
        all_sequences = []
        for dataset_path in datasets:
            if os.path.exists(dataset_path):
                sequences = load_dataset_sequences(dataset_path)
                all_sequences.extend(sequences)
                logger.info(f"从 {dataset_path} 加载了 {len(sequences)} 个序列")
        
        if not all_sequences:
            logger.error("❌ 没有找到有效的序列数据")
            return False
        
        logger.info(f"📊 总共需要处理 {len(all_sequences)} 个序列")
        
        # 检查已缓存的序列
        cached_count = 0
        uncached_sequences = []
        
        for seq_id, sequence in all_sequences:
            if cache_manager.has_cached_structure(sequence):
                cached_count += 1
            else:
                uncached_sequences.append((seq_id, sequence))
        
        logger.info(f"📊 已缓存: {cached_count}, 需要计算: {len(uncached_sequences)}")
        
        if not uncached_sequences:
            logger.info("✅ 所有序列都已缓存，无需重新计算")
            return True
        
        # 批量处理未缓存的序列
        batch_size = 50  # 每批处理50个序列
        total_batches = (len(uncached_sequences) + batch_size - 1) // batch_size
        
        logger.info(f"🔄 开始批量预计算，共 {total_batches} 批")
        
        # 统计信息
        total_computed = 0
        total_failed = 0
        total_time = 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(uncached_sequences))
            batch = uncached_sequences[start_idx:end_idx]
            
            logger.info(f"📦 处理批次 {batch_idx + 1}/{total_batches} ({len(batch)} 个序列)")
            
            # 处理当前批次
            batch_start_time = time.time()
            results = precompute_batch_structures(esmfold_wrapper, batch, cache_manager)
            batch_time = time.time() - batch_start_time
            
            # 统计结果
            batch_computed = sum(1 for _, status, _ in results if status == "computed")
            batch_failed = sum(1 for _, status, _ in results if status in ["prediction_failed", "error", "cache_failed"])
            
            total_computed += batch_computed
            total_failed += batch_failed
            total_time += batch_time
            
            # 显示批次结果
            avg_time_per_seq = batch_time / len(batch)
            logger.info(f"  ✅ 批次完成: 计算 {batch_computed}, 失败 {batch_failed}, 平均 {avg_time_per_seq:.2f}s/序列")
            
            # 保存缓存索引
            if batch_idx % 5 == 0:  # 每5批保存一次
                cache_manager.save_disk_index()
            
            # 显示进度
            progress = (batch_idx + 1) / total_batches * 100
            remaining_batches = total_batches - batch_idx - 1
            estimated_remaining_time = remaining_batches * (total_time / (batch_idx + 1)) / 60
            
            logger.info(f"📈 进度: {progress:.1f}%, 预计剩余: {estimated_remaining_time:.1f}分钟")
        
        # 保存最终缓存索引
        cache_manager.save_disk_index()
        
        # 最终统计
        logger.info("🎉 结构特征预计算完成!")
        logger.info(f"📊 最终统计:")
        logger.info(f"  总序列数: {len(all_sequences)}")
        logger.info(f"  已缓存: {cached_count}")
        logger.info(f"  新计算: {total_computed}")
        logger.info(f"  失败: {total_failed}")
        logger.info(f"  总耗时: {total_time / 60:.1f}分钟")
        
        # 显示最终缓存统计
        final_stats = cache_manager.get_cache_stats()
        logger.info(f"📊 最终缓存统计: {final_stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 预计算过程失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理资源
        if 'cache_manager' in locals():
            cache_manager.save_disk_index()
            cache_manager.clear_memory_cache()

def verify_cache_integrity():
    """验证缓存完整性"""
    logger.info("🔍 验证缓存完整性...")
    
    cache_manager = StructureFeatureCache()
    stats = cache_manager.get_cache_stats()
    
    logger.info(f"📊 缓存统计: {stats}")
    
    # 验证几个随机样本
    datasets = [
        "/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
        "/home/qlyu/sequence/StructDiff-7.0.0/data/processed/val.csv"
    ]
    
    sample_count = 0
    valid_count = 0
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            sequences = load_dataset_sequences(dataset_path)
            
            # 检查前10个序列
            for seq_id, sequence in sequences[:10]:
                sample_count += 1
                if cache_manager.has_cached_structure(sequence):
                    structure = cache_manager.get_cached_structure(sequence)
                    if structure is not None:
                        valid_count += 1
                        logger.info(f"✅ 序列 {seq_id}: 缓存有效")
                    else:
                        logger.warning(f"⚠️ 序列 {seq_id}: 缓存损坏")
                else:
                    logger.warning(f"❌ 序列 {seq_id}: 未缓存")
    
    logger.info(f"🔍 验证完成: {valid_count}/{sample_count} 有效")
    return valid_count == sample_count

if __name__ == "__main__":
    try:
        # 预计算所有结构特征
        success = precompute_all_structures()
        
        if success:
            # 验证缓存完整性
            verify_cache_integrity()
            logger.info("✅ 结构特征预计算和验证完成!")
        else:
            logger.error("❌ 结构特征预计算失败!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断预计算过程")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 预计算脚本失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 