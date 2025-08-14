# Contributing Guide

Welcome to contribute code to the Genesis deep learning framework! This guide will help you understand how to participate in project development.

## 🤝 Ways to Contribute

### Code Contributions
- Fix bugs
- Add new features
- Performance optimization
- Test improvements

### Documentation Contributions
- Improve existing documentation
- Add tutorials and examples
- Translate documentation
- API documentation improvements

### Community Contributions
- Answer questions
- Code review
- Issue reporting
- Feature suggestions

## 📋 Development Workflow

### 1. Preparation

```bash
# Fork the project to your GitHub account
# Clone你的fork
git clone https://github.com/YOUR_USERNAME/genesis.git
cd genesis

# 添加上游仓库
git remote add upstream https://github.com/phonism/genesis.git

# 创建开发分支
git checkout -b feature/your-feature-name
```

### 2. 开发环境搭建

详见[开发环境配置](development.md)文档。

### 3. 代码开发

- 遵循代码规范
- 添加单元测试
- 更新相关文档
- 提交清晰的commit消息

### 4. 测试验证

```bash
# 运行测试套件
python -m pytest tests/ -v

# 运行代码格式检查
black genesis/ tests/
flake8 genesis/ tests/

# 运行类型检查
mypy genesis/
```

### 5. 提交PR

- 确保所有测试通过
- 填写详细的PR描述
- 链接相关的Issue
- 等待代码审查

## 📝 代码规范

### Python风格指南

我们遵循[PEP 8](https://pep8.org/)规范：

```python
# 好的示例
def compute_attention_weights(query, key, scale_factor):
    """Compute scaled dot-product attention weights.
    
    Args:
        query: Query tensor of shape [batch, seq_len, hidden_dim]
        key: Key tensor of shape [batch, seq_len, hidden_dim] 
        scale_factor: Scaling factor for attention scores
        
    Returns:
        Attention weights of shape [batch, seq_len, seq_len]
    """
    scores = genesis.matmul(query, key.transpose(-2, -1))
    scaled_scores = scores * scale_factor
    return genesis.softmax(scaled_scores, dim=-1)
```

### 文档字符串

使用Google风格的docstring：

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """One line summary of the function.
    
    More detailed description if needed. Can span multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return param1 > 0
```

### 测试编写

```python
import pytest
import genesis

class TestAttention:
    """Test attention mechanisms."""
    
    def test_basic_attention(self):
        """Test basic attention computation."""
        batch_size, seq_len, hidden_dim = 2, 4, 8
        
        query = genesis.randn(batch_size, seq_len, hidden_dim)
        key = genesis.randn(batch_size, seq_len, hidden_dim)
        value = genesis.randn(batch_size, seq_len, hidden_dim)
        
        attention = genesis.nn.MultiHeadAttention(hidden_dim, num_heads=2)
        output = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
        
    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_different_head_counts(self, num_heads):
        """Test attention with different head counts."""
        # 测试实现
        pass
```

## 🚀 开发最佳实践

### 1. 分支管理

```bash
# 主要分支
main          # 稳定版本
develop       # 开发版本

# 功能分支
feature/xxx   # 新功能开发
bugfix/xxx    # bug修复
hotfix/xxx    # 紧急修复
```

### 2. Commit消息格式

```
type(scope): brief description

Detailed description (optional)

Fixes #123
```

类型说明：
- `feat`: 新功能
- `fix`: bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建工具等

### 3. 性能考虑

- 避免不必要的内存拷贝
- 使用in-place操作when可能
- 考虑CUDA kernel的内存访问模式
- 添加性能基准测试

## 🐛 Bug报告

提交bug时请包含：

1. **环境信息**
   - Genesis版本
   - Python版本
   - CUDA版本
   - 操作系统

2. **复现步骤**
   - 最小可复现代码
   - 预期行为
   - 实际行为
   - 错误信息

3. **相关日志**
   - 完整的错误堆栈
   - 相关配置信息

示例：
```python
# 最小复现案例
import genesis

model = genesis.nn.Linear(10, 5)
x = genesis.randn(3, 10)
y = model(x)  # 这里出现错误

# 错误信息：
# RuntimeError: CUDA kernel launch failed
```

## 🎯 贡献重点领域

当前我们特别欢迎以下领域的贡献：

### 高优先级
- [ ] 性能优化和基准测试
- [ ] CUDA算子实现
- [ ] 文档和教程完善
- [ ] 测试覆盖率提升

### 中优先级
- [ ] 新的神经网络层
- [ ] 数据加载器优化
- [ ] 分布式训练支持
- [ ] 混合精度训练

### 低优先级
- [ ] 可视化工具
- [ ] 模型部署支持
- [ ] 第三方框架集成

## 📞 联系我们

- **GitHub Issues**: 报告问题和功能请求
- **GitHub Discussions**: 技术讨论和问答
- **Email**: genesis-dev@example.com

## 🏆 贡献者认可

我们重视每一位贡献者的努力：

- 贡献者将列在项目README中
- 重大贡献者将获得维护者权限
- 定期发布贡献者通讯

## 📄 许可证

通过贡献代码，你同意你的贡献将在[MIT许可证](https://opensource.org/licenses/MIT)下发布。

---

!!! info "开始贡献"
    准备好开始贡献了吗？先从[开发环境配置](development.md)开始吧！

感谢你为Genesis项目的贡献！🎉