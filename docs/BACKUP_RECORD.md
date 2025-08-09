# 备份记录

## 备份信息
- **备份时间**: 2025-08-04 20:22:12
- **备份原因**: 在进行架构改进和文件清理前创建完整备份
- **备份方式**: 手动备份关键文件

## 备份建议

### 方法1：使用Git（推荐）
```bash
# 确保所有更改已提交
git add .
git commit -m "备份：架构改进前的完整状态"
git tag -a "backup-before-cleanup-v8.0.0" -m "架构改进前备份"
```

### 方法2：手动备份
1. 复制整个项目文件夹到安全位置
2. 命名为：`StructDiff-8.0.0_backup_20250804`

### 方法3：使用提供的脚本
- Windows: 运行 `create_backup.bat`
- Linux/Mac: 运行 `bash create_backup.sh`

## 重要文件列表（必须备份）

### 核心代码
- `structdiff/` - 所有核心代码
- `scripts/train_separated.py` - 主训练脚本
- `scripts/cpldiff_standard_evaluation.py` - 评估脚本
- `configs/separated_training_production.yaml` - 生产配置

### 文档
- `docs/` - 所有文档
- `README.md` - 项目说明
- `requirements.txt` - 依赖列表

### 模型和数据（如果有）
- `checkpoints/` - 训练的模型检查点
- `outputs/` - 训练输出
- `structure_cache/` - 结构缓存

## 恢复方法

如果需要恢复某些文件：

### 从Git恢复
```bash
# 查看备份标签
git tag -l "*backup*"

# 恢复到备份点
git checkout backup-before-cleanup-v8.0.0

# 或恢复特定文件
git checkout backup-before-cleanup-v8.0.0 -- path/to/file
```

### 从手动备份恢复
直接从备份文件夹复制需要的文件

## 清理前检查清单

- [x] 创建备份记录文档
- [ ] 执行Git提交和标签
- [ ] 确认重要文件已备份
- [ ] 记录当前Git commit hash
- [ ] 准备开始清理

---

**注意**: 在开始清理前，请确保已经完成上述备份步骤之一。