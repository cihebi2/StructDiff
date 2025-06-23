# StructDiff Git 代码同步指南

## 快速同步（推荐）

如果你想快速同步 GitHub 上的最新代码，请按以下步骤操作：

### 方法1: 使用同步脚本

```bash
# 1. 进入项目目录
cd /home/qlyu/sequence/StructDiff

# 2. 给脚本添加执行权限
chmod +x git_sync.sh

# 3. 运行同步脚本
./git_sync.sh
```

### 方法2: 手动执行 Git 命令

```bash
# 1. 进入项目目录
cd /home/qlyu/sequence/StructDiff

# 2. 检查当前状态
git status

# 3. 如果有未提交的修改，先暂存
git stash push -m "暂存本地修改"

# 4. 获取远程最新代码
git fetch origin

# 5. 合并远程代码
git pull origin main

# 6. 如果之前暂存了修改，恢复暂存
git stash pop
```

## 详细步骤说明

### 步骤1: 检查当前状态

```bash
cd /home/qlyu/sequence/StructDiff
git status
```

这会显示：
- 当前分支
- 未提交的修改
- 未跟踪的文件

### 步骤2: 处理本地修改

**如果工作区干净（没有修改）:**
直接进行下一步

**如果有未提交的修改:**
```bash
# 选项A: 暂存修改（推荐）
git stash push -m "自动暂存 - $(date '+%Y-%m-%d %H:%M:%S')"

# 选项B: 提交修改
git add .
git commit -m "保存本地修改"

# 选项C: 丢弃修改（谨慎使用）
git reset --hard HEAD
```

### 步骤3: 同步远程代码

```bash
# 获取远程最新信息
git fetch origin

# 检查是否有更新
git log --oneline HEAD..origin/main

# 合并远程代码
git merge origin/main
```

或者一步完成：
```bash
git pull origin main
```

### 步骤4: 恢复本地修改

**如果之前使用了 stash:**
```bash
git stash pop
```

## 可能遇到的问题

### 1. 合并冲突

如果出现冲突，会看到类似信息：
```
Auto-merging some_file.py
CONFLICT (content): Merge conflict in some_file.py
Automatic merge failed; fix conflicts and then commit the result.
```

**解决方法:**
```bash
# 1. 查看冲突文件
git status

# 2. 编辑冲突文件，手动解决冲突
# 删除 <<<<<<< HEAD, =======, >>>>>>> origin/main 标记
# 保留需要的代码

# 3. 标记冲突已解决
git add <冲突文件>

# 4. 完成合并
git commit -m "解决合并冲突"
```

### 2. 恢复暂存时的冲突

```bash
# 如果 git stash pop 失败
git stash apply  # 应用暂存但不删除
# 手动解决冲突后
git stash drop   # 删除暂存
```

### 3. 回滚操作

```bash
# 回滚到上一个提交
git reset --hard HEAD~1

# 或使用项目提供的回滚脚本
./git_rollback.sh -l  # 列出所有版本
./git_rollback.sh -p  # 回滚到上一版本
```

## 验证同步结果

```bash
# 查看最新提交
git log --oneline -5

# 查看远程分支状态
git status

# 查看版本信息
cat VERSION
```

## 有用的 Git 命令

```bash
# 查看所有分支
git branch -a

# 查看远程仓库信息
git remote -v

# 查看提交历史
git log --oneline --graph -10

# 查看文件差异
git diff HEAD~5

# 查看暂存列表
git stash list

# 清理未跟踪文件
git clean -fd
```

## 自动化脚本

项目中已包含以下 Git 管理脚本：

- `git_sync.sh` - 代码同步脚本（新增）
- `git_upload.sh` - 代码上传脚本
- `git_rollback.sh` - 版本回滚脚本

使用方法：
```bash
chmod +x git_*.sh
./git_sync.sh      # 同步代码
./git_upload.sh -h # 查看上传帮助
./git_rollback.sh -l # 查看所有版本
```

## 注意事项

1. **备份重要修改**: 同步前建议备份重要的本地修改
2. **检查冲突**: 仔细检查并解决所有合并冲突
3. **测试代码**: 同步后运行测试确保代码正常
4. **版本管理**: 利用项目的版本管理脚本进行规范化操作

## 故障排除

如果遇到问题，可以：

1. 查看详细错误信息
2. 使用 `git status` 检查状态
3. 使用 `git log` 查看历史
4. 必要时使用 `git_rollback.sh` 回滚
5. 参考项目的其他文档 