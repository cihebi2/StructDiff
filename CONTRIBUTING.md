# 贡献指南

感谢您对 StructDiff 项目的关注！我们欢迎各种形式的贡献，包括但不限于：

- Bug 报告和修复
- 功能请求和实现
- 文档改进
- 性能优化
- 测试用例

## 如何贡献

### 1. Fork 项目

- 点击页面右上角的 "Fork" 按钮
- 将项目克隆到本地：`git clone https://github.com/你的用户名/StructDiff.git`

### 2. 创建分支

```bash
git checkout -b feature/your-feature-name
```

### 3. 开发环境设置

```bash
# 创建虚拟环境
conda env create -f environment.yml
conda activate structdiff

# 安装开发依赖
pip install -e .[dev]
```

### 4. 代码规范

我们使用以下工具来保持代码质量：

- **代码格式化**: `black` 和 `isort`
- **代码检查**: `flake8` 和 `mypy`
- **测试**: `pytest`

在提交前请运行：

```bash
# 格式化代码
black structdiff/ tests/
isort structdiff/ tests/

# 检查代码
flake8 structdiff/ tests/
mypy structdiff/

# 运行测试
pytest tests/
```

### 5. 提交更改

```bash
git add .
git commit -m "feat: 添加新功能的描述"
git push origin feature/your-feature-name
```

### 6. 创建 Pull Request

- 在 GitHub 上创建 Pull Request
- 详细描述您的更改
- 确保所有测试通过

## 代码风格

### Python 代码风格

- 遵循 PEP 8 标准
- 使用 type hints
- 编写清晰的文档字符串

```python
def process_sequence(sequence: str, max_length: int = 50) -> List[str]:
    """
    处理蛋白质序列。

    Args:
        sequence: 输入的蛋白质序列
        max_length: 最大长度限制

    Returns:
        处理后的序列列表

    Raises:
        ValueError: 当序列包含无效字符时
    """
    pass
```

### 提交信息格式

使用约定式提交格式：

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `style:` 代码格式化（不影响功能）
- `refactor:` 代码重构
- `test:` 添加或修改测试
- `chore:` 其他维护性更改

## 报告问题

使用 [GitHub Issues](https://github.com/yourusername/StructDiff/issues) 来报告问题。请包含：

- **环境信息**: 操作系统、Python 版本、PyTorch 版本等
- **问题描述**: 清晰描述问题
- **重现步骤**: 详细的重现步骤
- **预期行为**: 您期望的正确行为
- **实际行为**: 实际发生的情况
- **错误信息**: 完整的错误堆栈跟踪

## 功能请求

我们欢迎功能请求！请通过 GitHub Issues 提交，并包含：

- **需求描述**: 详细说明您需要的功能
- **使用场景**: 什么情况下会用到这个功能
- **建议实现**: 如果有的话，提供实现思路

## 开发指南

### 项目结构

```
StructDiff/
├── structdiff/           # 主要代码
│   ├── models/          # 模型定义
│   ├── data/            # 数据处理
│   ├── utils/           # 工具函数
│   └── evaluate/        # 评估模块
├── configs/             # 配置文件
├── scripts/             # 脚本文件
├── tests/               # 测试文件
└── notebooks/           # Jupyter notebooks
```

### 添加新功能

1. 在适当的模块中添加代码
2. 编写单元测试
3. 更新文档
4. 添加配置选项（如需要）

### 性能测试

对于性能相关的更改，请提供基准测试结果：

```bash
python scripts/benchmark.py
```

## 社区

- **讨论**: 使用 GitHub Discussions 进行技术讨论
- **聊天**: 加入我们的社群（如果有的话）

## 行为准则

请遵循我们的行为准则：

- 尊重他人
- 包容不同观点
- 专注于对项目有益的讨论
- 帮助新贡献者

感谢您的贡献！🎉 