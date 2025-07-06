# GPU利用率优化计划

## 🚨 当前问题分析

### 现状
- **GPU利用率**: 仅14% (远低于理想的80-95%)
- **功率使用**: 88W / 450W (仅19%)
- **内存使用**: 18GB / 24GB (75% - 合理)
- **训练速度**: ~2秒/批次 (过慢)

### 根本原因
1. **数据加载瓶颈** - 单线程加载，批次过小
2. **ESMFold计算瓶颈** - CPU密集型结构预测
3. **计算-内存不平衡** - 高内存占用但低计算强度
4. **未充分利用并行性** - 保守的并发设置

## 📈 优化策略与实施计划

### 🎯 目标
- GPU利用率: 14% → 80%+
- 功率使用: 88W → 300W+
- 训练速度: 2秒/批次 → 0.5秒/批次
- 保持内存使用在安全范围内

---

## 🚀 优化方案

### 1. **数据加载优化** (预期提升: 30-50%)

#### 1.1 多进程数据加载
```python
# 当前设置
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    num_workers=0,  # ❌ 单线程
    pin_memory=False  # ❌ 未启用内存固定
)

# 优化后设置
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # 增加批次大小
    num_workers=4,  # ✅ 多进程加载
    pin_memory=True,  # ✅ 启用内存固定
    prefetch_factor=2,  # ✅ 预取缓冲
    persistent_workers=True  # ✅ 持久化工作进程
)
```

#### 1.2 结构特征缓存策略
```python
# 实现智能缓存系统
class StructureCacheManager:
    def __init__(self, cache_dir, max_cache_size=1000):
        self.cache_dir = cache_dir
        self.cache = {}
        self.max_size = max_cache_size
    
    def get_or_predict(self, sequence, esmfold_wrapper):
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()
        cache_file = f"{self.cache_dir}/{seq_hash}.pkl"
        
        if os.path.exists(cache_file):
            # 从缓存加载
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            # 预测并缓存
            structure = esmfold_wrapper.predict_structure(sequence)
            with open(cache_file, 'wb') as f:
                pickle.dump(structure, f)
            return structure
```

### 2. **批次大小优化** (预期提升: 40-60%)

#### 2.1 动态批次大小调整
```python
def optimize_batch_size():
    """动态寻找最优批次大小"""
    base_batch_size = 2
    max_batch_size = 16
    
    for batch_size in [2, 4, 6, 8, 12, 16]:
        try:
            # 测试批次大小
            test_batch = create_test_batch(batch_size)
            start_time = time.time()
            
            with torch.cuda.amp.autocast():
                outputs = model(test_batch)
                loss = outputs['total_loss']
                loss.backward()
            
            batch_time = time.time() - start_time
            memory_used = torch.cuda.memory_allocated() / 1e9
            
            if memory_used < 22:  # 保持2GB安全边界
                optimal_batch_size = batch_size
                logger.info(f"批次大小 {batch_size}: 时间={batch_time:.2f}s, 内存={memory_used:.1f}GB")
            else:
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
    
    return optimal_batch_size
```

#### 2.2 梯度累积优化
```python
# 当前设置
batch_size = 2
gradient_accumulation_steps = 8
effective_batch_size = 16

# 优化设置
batch_size = 8  # 增加到8
gradient_accumulation_steps = 2  # 减少到2
effective_batch_size = 16  # 保持不变

# 预期效果: 减少梯度累积次数，增加GPU计算密度
```

### 3. **混合精度训练** (预期提升: 20-30%)

#### 3.1 启用AMP (Automatic Mixed Precision)
```python
# 添加到训练循环
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def optimized_training_step(model, batch, optimizer):
    with autocast():  # ✅ 自动混合精度
        outputs = model(
            sequences=batch['sequences'],
            attention_mask=batch['attention_mask'],
            timesteps=timesteps,
            structures=batch.get('structures'),
            conditions=batch.get('conditions'),
            return_loss=True
        )
        loss = outputs['total_loss'] / gradient_accumulation_steps
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    
    return outputs, loss.item() * gradient_accumulation_steps
```

#### 3.2 模型优化
```python
# 启用优化编译 (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")

# 或者使用更激进的优化
model = torch.compile(model, mode="max-autotune")
```

### 4. **并行计算优化** (预期提升: 25-40%)

#### 4.1 ESMFold并行预测
```python
class ParallelESMFoldWrapper:
    def __init__(self, device, num_parallel=2):
        self.device = device
        self.num_parallel = num_parallel
        self.esmfold_pool = []
        
        # 创建多个ESMFold实例
        for i in range(num_parallel):
            esmfold = ESMFoldWrapper(device=f"cuda:{device}")
            self.esmfold_pool.append(esmfold)
    
    def predict_batch_parallel(self, sequences):
        """并行预测多个序列的结构"""
        from concurrent.futures import ThreadPoolExecutor
        
        def predict_single(seq_and_model):
            seq, model_idx = seq_and_model
            return self.esmfold_pool[model_idx].predict_structure(seq)
        
        # 分配序列到不同的ESMFold实例
        seq_model_pairs = [
            (seq, i % self.num_parallel) 
            for i, seq in enumerate(sequences)
        ]
        
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            results = list(executor.map(predict_single, seq_model_pairs))
        
        return results
```

#### 4.2 异步数据预处理
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncDataProcessor:
    def __init__(self, esmfold_wrapper, max_workers=4):
        self.esmfold_wrapper = esmfold_wrapper
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = {}
    
    async def preprocess_batch_async(self, sequences):
        """异步预处理批次数据"""
        loop = asyncio.get_event_loop()
        
        # 并行处理所有序列
        tasks = [
            loop.run_in_executor(
                self.executor,
                self.process_single_sequence,
                seq
            ) for seq in sequences
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def process_single_sequence(self, sequence):
        """处理单个序列"""
        if sequence in self.cache:
            return self.cache[sequence]
        
        structure = self.esmfold_wrapper.predict_structure(sequence)
        self.cache[sequence] = structure
        return structure
```

### 5. **内存-计算平衡优化** (预期提升: 15-25%)

#### 5.1 梯度检查点 (Gradient Checkpointing)
```python
# 在模型初始化时启用
from torch.utils.checkpoint import checkpoint

class OptimizedStructDiff(StructDiff):
    def forward(self, *args, **kwargs):
        # 使用梯度检查点减少内存使用
        if self.training:
            return checkpoint(super().forward, *args, **kwargs)
        else:
            return super().forward(*args, **kwargs)
```

#### 5.2 动态内存管理
```python
def dynamic_memory_management():
    """动态内存管理策略"""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    
    # 如果内存使用率低于70%，可以增加批次大小
    if allocated / (24 * 1024**3) < 0.7:
        return "increase_batch_size"
    
    # 如果内存使用率高于90%，需要清理
    elif allocated / (24 * 1024**3) > 0.9:
        torch.cuda.empty_cache()
        return "decrease_batch_size"
    
    return "maintain"
```

## 🛠️ 实施计划

### 阶段1: 快速优化 (1-2小时实施)

#### 1.1 创建优化版本训练脚本
```python
# 文件名: full_train_optimized_gpu_utilization.py

def create_optimized_training_script():
    """创建GPU利用率优化版本"""
    
    # 1. 增加批次大小
    batch_size = 6  # 从2增加到6
    gradient_accumulation_steps = 3  # 相应调整
    
    # 2. 启用多进程数据加载
    num_workers = 4
    pin_memory = True
    prefetch_factor = 2
    
    # 3. 启用混合精度训练
    use_amp = True
    
    # 4. 优化数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=collator,
        drop_last=True
    )
    
    return train_loader
```

#### 1.2 批次大小测试脚本
```python
# 文件名: test_optimal_batch_size.py

def test_batch_sizes():
    """测试不同批次大小的性能"""
    
    batch_sizes = [2, 4, 6, 8, 10, 12]
    results = {}
    
    for bs in batch_sizes:
        try:
            start_time = time.time()
            
            # 创建测试批次
            test_batch = create_test_batch(bs)
            
            # 测试前向传播
            with torch.cuda.amp.autocast():
                outputs = model(test_batch)
                loss = outputs['total_loss']
            
            # 测试反向传播
            loss.backward()
            
            end_time = time.time()
            memory_used = torch.cuda.memory_allocated() / 1e9
            
            results[bs] = {
                'time': end_time - start_time,
                'memory': memory_used,
                'throughput': bs / (end_time - start_time)
            }
            
            print(f"批次大小 {bs}: 时间={end_time - start_time:.2f}s, "
                  f"内存={memory_used:.1f}GB, "
                  f"吞吐量={bs / (end_time - start_time):.2f} samples/s")
            
            # 清理梯度和内存
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"批次大小 {bs}: 内存不足")
                break
            else:
                raise e
    
    return results
```

### 阶段2: 中级优化 (半天实施)

#### 2.1 结构特征缓存系统
- 实现智能缓存管理
- 预计算常见序列的结构特征
- 减少ESMFold重复计算

#### 2.2 异步数据处理
- 实现异步数据预处理
- 并行结构预测
- 重叠计算和数据加载

### 阶段3: 高级优化 (1天实施)

#### 3.1 模型编译优化
- 使用PyTorch 2.0编译优化
- 自定义CUDA核心（如需要）
- 算子融合优化

#### 3.2 多GPU并行 (如果需要)
- 数据并行训练
- 模型并行（针对大模型）
- 流水线并行

## 📊 预期效果

### 性能提升预测
| 优化项目 | 当前状态 | 优化后 | 提升幅度 |
|---------|---------|--------|----------|
| GPU利用率 | 14% | 75-85% | 5-6倍 |
| 功率使用 | 88W | 280-320W | 3-4倍 |
| 训练速度 | 2.0s/批次 | 0.4-0.6s/批次 | 3-5倍 |
| 内存效率 | 中等 | 高 | 20-30% |
| 总训练时间 | ~300小时 | ~60-80小时 | 4-5倍 |

### 资源利用率目标
```
GPU利用率: 75-85% (理想范围)
内存使用: 20-22GB (留2GB安全边界)
功率使用: 280-320W (60-70%最大功率)
温度: <75°C (安全范围)
```

## 🚀 立即可执行的快速优化

### 1. 测试最优批次大小
```bash
cd /home/qlyu/sequence/StructDiff-7.0.0
python test_optimal_batch_size.py
```

### 2. 创建优化版本训练脚本
```bash
# 基于当前脚本创建优化版本
cp full_train_with_structure_features_fixed_v2.py full_train_optimized_v1.py
# 然后手动编辑优化参数
```

### 3. 监控优化效果
```bash
# 实时监控GPU利用率
watch -n 1 nvidia-smi

# 监控训练速度
tail -f outputs/structure_feature_training_optimized/training.log
```

通过这些优化，预计可以将GPU利用率从14%提升到75%以上，训练速度提升3-5倍，大大缩短训练时间！ 