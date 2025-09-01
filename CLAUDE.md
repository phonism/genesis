# CLAUDE.md - Genesis项目关键指南

## ⚠️ 严重错误警告 - 永远不要忘记

### 🚫 绝对禁止在远程服务器使用 `pkill`
**2025-01-08的严重错误**: 在远程服务器执行`pkill -9 python`，杀死所有Python进程导致服务器完全断连。

**绝对不要执行**:
- `pkill python` 或 `pkill -9 python`
- `killall python`
- 任何按进程名批量杀进程的命令

**正确做法**:
1. 找具体进程: `ps aux | grep <specific>`
2. 只按PID杀: `kill <PID>`
3. 执行前必须确认影响范围

## 📝 代码规范 - 每次编码必须遵守

### 🔥 **[强制] PY033 - Docstring规范**
**每个模块、函数、类、方法都必须有docstring！**

```python
"""
Module docstring describing the purpose.
"""

def function_name():
    """
    Function docstring describing what it does.
    
    Args:
        param: Description
        
    Returns:
        Description of return value
    """
    pass

class ClassName:
    """Class docstring describing the class."""
    
    def method_name(self):
        """Method docstring describing what it does."""
        pass
```

### 🔥 **代码风格强制要求**
1. **双引号**: 字符串使用双引号，f-string内部用单引号
2. **行长度**: 最大120字符
3. **英文**: 所有代码和文档英文（除用户界面）
4. **Git提交**: 绝对不包含Claude相关信息
5. **Docstring**: 必须用三重双引号 `"""` 包围
6. **❌ 禁止局部import**: 永远不在函数内部import，所有import必须在文件顶部

### ⚡ **检查清单 - 代码提交前必查**
- [ ] 所有函数/类/方法都有docstring
- [ ] 使用双引号
- [ ] 行长度 < 120字符
- [ ] 提交信息无Claude相关内容

## 项目核心信息

### 远程执行关键路径
- **本地**: `/Users/luqi03/workspace/online_github/code/genesis/`
- **远程**: `/root/workspace/code/genesis/`
- **GPU服务器**: 2张A800 (使用`CUDA_VISIBLE_DEVICES=0`或`1`)
- **同步**: 执行`./upload.sh`将本地代码同步到远程

### 核心API
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
from genesis import float32, float16, bfloat16

# 创建张量 - 支持tensor和Tensor两种API
x = genesis.tensor(data, device=genesis.device('cuda'), dtype=genesis.float32)  # PyTorch风格
x = genesis.Tensor(data, device=genesis.device('cuda'))  # 原始API

# 基础操作
z = genesis.matmul(x, y)
loss.backward()
optimizer.step()

# 新特性
genesis.save_checkpoint(model_state, "checkpoint.pth")
state = genesis.load_checkpoint("checkpoint.pth")
```

### 架构要点
- **双后端**: CPU用PyTorch，GPU用CUDA+Triton
- **核心文件**:
  - `genesis/__init__.py`: 主API入口和功能函数
  - `genesis/autograd.py`: Tensor类和自动微分引擎
  - `genesis/ndarray/cuda_storage.py`: 纯CUDA存储后端实现
  - `genesis/nn/modules/`: 模块化神经网络层目录
    - `module.py`: 基础Module和Parameter类
    - `linear.py`: Linear, Flatten层
    - `loss.py`: CrossEntropyLoss, MSELoss, BCELoss等损失函数
    - `activation.py`: 激活函数（ReLU, Softmax, SiLU等）
    - `normalization.py`: 归一化层（LayerNorm, BatchNorm等）
    - `transformer.py`: Attention机制和Transformer组件
  - `genesis/nn/functional.py`: 函数式神经网络操作
  - `genesis/optim/`: 优化器实现（Adam, AdamW, SGD）
  - `genesis/models/qwen.py`: Qwen大模型完整实现
  - `genesis/dtypes.py`: 数据类型系统（支持float16/bfloat16）
  - `genesis/utils/`: 工具模块（数据加载、性能分析等）
- **应用层**:
  - `apps/llm/`: LLM训练和推理应用
  - `benchmark/`: 性能测试和基准比较
  - `tests/`: 完整测试套件
  - `docs/`: MkDocs文档系统
- **新特性**:
  - 专业文档系统（MkDocs Material）
  - 混合精度训练（AMP）支持
  - 梯度裁剪和学习率调度器
  - 模型检查点保存/加载系统
  - Qwen模型完整训练和推理支持
  - 增强的基准测试和性能分析工具
  - GPU远程开发和调试工具集成

### 关键测试命令
```bash
# 同步代码到远程服务器
./upload.sh

# 远程GPU测试和开发
mcp__gpu-remote__exec "cd /root/workspace/code/genesis && CUDA_VISIBLE_DEVICES=0 python test.py"
mcp__gpu-remote__list_dir "/root/workspace/code/genesis"  # 查看远程文件
mcp__gpu-remote__get_file "/root/workspace/code/genesis/somefile.py"  # 下载远程文件

# 性能测试基准
python benchmark/bench_matmul.py          # 矩阵乘法性能
python benchmark/bench_functional.py      # 函数式操作性能
python benchmark/bench_ops.py             # 基础操作性能
python benchmark/bench_qwen.py            # Qwen模型性能
python benchmark/simple_qwen_bench.py     # 简化Qwen基准
python benchmark/profile_qwen.py          # Qwen性能分析
python benchmark/compare_perf.py          # 性能对比分析

# 运行测试套件
python -m pytest tests/ -v               # 完整测试套件
python -m pytest tests/test_qwen.py -v   # Qwen模型测试
python tests/test_functional.py          # 功能测试
python tests/test_nn.py                  # 神经网络测试
python tests/test_autograd.py            # 自动微分测试

# LLM训练和推理
cd apps/llm
python train_sft_qwen.py                 # Qwen SFT训练
python train_sft.py                      # 通用SFT训练
python chat_qwen.py                      # Qwen推理聊天
python torch_qwen.py                     # PyTorch对比测试

# 文档构建和部署
mkdocs serve                             # 本地预览文档
mkdocs build                             # 构建静态文档
./deploy_docs.sh                         # 部署文档到GitHub Pages

# 代码质量检查
python -m pytest --cov=genesis --cov-report=html  # 测试覆盖率
black genesis/                           # 代码格式化
isort genesis/                           # 导入排序
mypy genesis/                            # 类型检查
```

### 当前性能状态与优化
- **元素级操作效率**:
  - 中型张量(4M): 29.6%效率 (0.32x vs PyTorch)
  - 小型张量(64K): 18.9%效率 (持续优化中)
  - 大型张量(16M+): 4.7%效率 (需要重大优化)
- **矩阵乘法**: 0.25-0.37x速度比 vs PyTorch (持续优化中)
- **内存管理**: 已简化CUDA内存管理，提高稳定性
- **自动微分**: 开销从86.3%降至~50%

### 最近更新
- ✅ **v0.5.0** - 神经网络模块重构和损失函数扩展（2025-08-28）
  - **🏗️ 模块化重构**: 将monolithic的`genesis/nn/modules.py`按PyTorch模式重构为模块化目录结构
    - `genesis/nn/modules/` - 模块化目录
    - `module.py` - 基础Module和Parameter类
    - `linear.py` - Linear, Flatten层
    - `activation.py` - ReLU, Softmax, SiLU, Residual激活函数
    - `normalization.py` - BatchNorm1d, LayerNorm, RMSNorm等
    - `loss.py` - CrossEntropyLoss, MSELoss, L1Loss, BCELoss等损失函数
    - `container.py` - Sequential, ModuleList容器
    - `dropout.py` - Dropout正则化
    - `sparse.py` - Embedding, RotaryEmbedding
    - `transformer.py` - MultiheadAttention, FeedForwardSwiGLU
  - **💯 PyTorch兼容**: 新增CrossEntropyLoss, MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss
  - **🔧 函数增强**: 添加log_softmax, maximum, randint等functional API
  - **✅ 完全兼容**: 所有现有测试通过，API保持向后兼容
- ✅ **v0.4.0** - 专业文档系统和性能分析工具（95dfebc）
  - 完整的英文文档和API参考
  - 增强的基准测试套件，支持CUDA事件计时
  - 简化CUDA内存管理，提高稳定性
- ✅ **修复reduce操作精度问题**（24d594f）
- ✅ **v0.3.0** - Qwen模型和混合精度训练支持
  - 完整Qwen LLM架构实现
  - 自动混合精度（AMP）支持
  - 高级优化器（AdamW, 学习率调度器）
  - 模型检查点保存/加载系统

### 文档系统
- **完整文档**: 中英双语MkDocs文档系统
- **API参考**: 自动生成的完整API文档
- **技术博客**: `blog/`目录包含技术深度文章
- **文档部署**: 使用`./deploy_docs.sh`部署到GitHub Pages
- **本地预览**: `mkdocs serve`启动本地文档服务器

### 调试和开发工具
```bash
# 常用调试命令
python -c "import genesis; print(genesis.__version__)"  # 检查版本
python -c "import genesis; print(genesis.device('cuda').is_available())"  # 检查CUDA
python simple_debug_v3.py                              # 调试脚本

# 监控脚本（如果需要）
python claude_monitor.py                               # Claude监控工具

# 清理和重置
./clear.sh                                             # 清理临时文件
./kill.sh                                              # 安全终止进程（限本地）
```

### 故障排除指南
- **CUDA初始化失败**: 使用`CUDA_VISIBLE_DEVICES=0`指定GPU
- **内存不足**: 减小batch_size或使用混合精度训练
- **性能问题**: 运行benchmark/目录下的性能测试工具
- **测试失败**: 检查dependencies和CUDA环境配置
- **远程连接问题**: 确认GPU服务器状态和upload.sh同步

## 重要原则
1. **永远不用pkill/killall** - 使用具体PID终止进程
2. **修改后必须upload.sh** - 确保代码同步到远程
3. **GPU测试用CUDA_VISIBLE_DEVICES=0或1** - 指定GPU避免冲突
4. **谨慎操作远程服务器** - 确认命令影响范围
5. **优先使用Triton后端** - 避免CUDA初始化问题
6. **文档优先** - 重大特性必须同步更新文档
7. **测试驱动** - 新功能必须有对应测试用例
8. **性能基准** - 重要优化需要benchmark验证效果