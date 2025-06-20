# StructDiff Git 版本管理指南

本项目使用自动化的Git版本管理系统，支持语义化版本控制和便捷的回滚操作。

## 快速开始

### 上传代码

```bash
# 基本上传（patch版本）
./git_upload.sh -m "修复bug"

# 小版本更新
./git_upload.sh -t minor -m "添加新功能"

# 大版本更新
./git_upload.sh -t major -m "重大架构更新"

# 强制推送
./git_upload.sh -f -m "强制更新"
```

### 查看版本

```bash
# 查看当前版本
./git_rollback.sh -c

# 列出所有版本
./git_rollback.sh -l
```

### 回滚版本

```bash
# 回滚到指定版本（保留修改）
./git_rollback.sh v1.2.3

# 硬回滚（丢弃所有修改）
./git_rollback.sh -h v1.2.3

# 回滚到上一版本
./git_rollback.sh -p
```

## 版本控制策略

### 语义化版本号

版本号格式：`v主版本.次版本.修订版本`

- **主版本 (major)**：重大架构变更、API不兼容更新
- **次版本 (minor)**：新功能添加、功能增强
- **修订版本 (patch)**：bug修复、小改进

### 自动化特性

1. **自动版本递增**：根据更新类型自动计算新版本号
2. **标签管理**：每个版本自动创建Git标签
3. **详细提交信息**：包含版本号、时间戳、更新类型
4. **完整代码上传**：确保所有文件都被提交

## 常用命令

### 上传操作

```bash
# 查看帮助
./git_upload.sh -h

# 小改动上传
./git_upload.sh -m "修复训练脚本的内存泄漏问题"

# 功能更新
./git_upload.sh -t minor -m "添加ESMFold结构预测功能"

# 重大更新
./git_upload.sh -t major -m "重构扩散模型架构"

# 指定分支上传
./git_upload.sh -b develop -m "开发分支更新"
```

### 回滚操作

```bash
# 查看帮助
./git_rollback.sh --help

# 查看版本历史
./git_rollback.sh -l

# 安全回滚（保留工作区修改）
./git_rollback.sh v1.2.3

# 完全回滚（丢弃所有修改）
./git_rollback.sh -h v1.2.3

# 软回滚（只移动HEAD，保留暂存区）
./git_rollback.sh -s v1.2.3
```

## 版本管理最佳实践

### 1. 提交前检查

```bash
# 检查当前状态
git status

# 查看修改内容
git diff

# 检查分支
git branch
```

### 2. 合理的版本递增

- **日常bug修复**：使用 patch 版本
- **新功能开发**：使用 minor 版本
- **重大重构**：使用 major 版本

### 3. 清晰的提交信息

```bash
# 好的提交信息示例
./git_upload.sh -m "修复ESMFold内存溢出问题"
./git_upload.sh -t minor -m "添加多肽生成条件控制功能"
./git_upload.sh -t major -m "迁移到PyTorch 2.0和Flash Attention"

# 避免的提交信息
./git_upload.sh -m "更新"  # 太模糊
./git_upload.sh -m "fix"   # 不够具体
```

## 紧急情况处理

### 意外上传错误内容

```bash
# 1. 查看最近的版本
./git_rollback.sh -l

# 2. 回滚到上一个正确版本
./git_rollback.sh -p

# 3. 修复问题后重新上传
./git_upload.sh -m "修复错误上传的问题"
```

### 版本冲突解决

```bash
# 1. 拉取远程更新
git pull origin main

# 2. 解决冲突后重新上传
./git_upload.sh -m "解决合并冲突"
```

### 强制推送（谨慎使用）

```bash
# 只在确认无其他人协作时使用
./git_upload.sh -f -m "强制修复版本问题"
```

## 团队协作规范

### 分支管理

```bash
# 主分支：稳定版本
git checkout main

# 开发分支：新功能开发
git checkout -b develop

# 功能分支：特定功能开发
git checkout -b feature/new-sampling-method
```

### 上传流程

1. **功能开发**：在功能分支进行
2. **测试验证**：确保代码可运行
3. **合并到开发分支**：
   ```bash
   git checkout develop
   git merge feature/new-sampling-method
   ./git_upload.sh -t minor -m "添加新的采样方法"
   ```
4. **发布稳定版本**：
   ```bash
   git checkout main
   git merge develop
   ./git_upload.sh -t major -m "发布v2.0.0稳定版本"
   ```

## 监控和维护

### 定期检查

```bash
# 检查版本历史
./git_rollback.sh -l

# 检查远程同步状态
git fetch origin
git status

# 清理本地分支
git branch -d feature/completed-feature
```

### 备份重要版本

```bash
# 创建重要版本的备份分支
git checkout -b backup/v1.0.0 v1.0.0
git push origin backup/v1.0.0
```

## 故障排除

### 常见问题

1. **上传失败**：检查网络连接和GitHub权限
2. **版本冲突**：先拉取远程更新再上传
3. **回滚失败**：确认版本标签存在
4. **权限问题**：检查脚本执行权限

### 恢复命令

```bash
# 如果脚本损坏，手动恢复
git add .
git commit -m "手动提交"
git push origin main

# 如果版本标签丢失，重新创建
git tag -a v1.2.3 -m "恢复版本标签"
git push origin --tags
```

## 联系和支持

如果遇到Git版本管理问题，请：

1. 检查本文档的故障排除部分
2. 查看Git官方文档
3. 在项目Issue中提问

---

**记住**：版本管理是为了保护代码安全和便于协作，请养成良好的提交习惯！ 