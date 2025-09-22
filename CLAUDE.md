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

### 架构要点 (v2.0 - 清理后的新架构)
- **模块化后端系统**:
  - `genesis/backends/`: 设备特定实现
    - `cpu.py`: CPU后端（使用PyTorch）
    - `cuda.py`: CUDA张量存储
    - `cuda_memory.py`: 高性能CUDA内存管理
    - `cuda_kernels.py`: 优化的CUDA内核
- **核心组件**:
  - `genesis/tensor.py`: Tensor类和自动微分支持
  - `genesis/function.py`: 自动微分Function基类
  - `genesis/device.py`: 统一设备抽象
  - `genesis/storage.py`: 存储接口层
  - `genesis/ops/`: 操作分发系统
    - `dispatcher.py`: 中央操作路由器
    - `cpu/`: CPU操作实现
    - `cuda/`: CUDA操作实现
- **神经网络层**:
  - `genesis/nn/modules/`: 模块化神经网络层
    - `module.py`: 基础Module和Parameter类
    - `linear.py`: Linear, Flatten层
    - `loss.py`: 完整的损失函数集合
    - `activation.py`: 激活函数
    - `normalization.py`: LayerNorm, BatchNorm, RMSNorm
    - `transformer.py`: Multi-head Attention, Transformer组件
  - `genesis/nn/functional.py`: 函数式神经网络操作
  - `genesis/nn/triton_ops/`: Triton加速操作
- **训练工具**:
  - `genesis/optim/`: 优化器（Adam, AdamW, SGD）
  - `genesis/distributed/`: 分布式训练支持（DDP）
  - `genesis/models/qwen.py`: Qwen LLM完整实现
- **其他特性**:
  - CUDA懒初始化确保可靠性
  - 清晰的模块边界和依赖关系
  - 删除了旧的ndarray和autograd模块
  - 统一的操作分发机制

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
- ✅ **v2.0.0** - 架构清理和模块化重构（2025-09-16）
  - **🏗️ 完全移除ndarray老架构**: 删除整个ndarray模块，功能迁移到backends/
  - **📦 模块化后端系统**: CPU和CUDA后端分离在backends/目录
  - **🎯 统一设备抽象**: 新增genesis.device模块集中管理设备
  - **⚡ CUDA懒初始化**: 解决初始化问题，提高稳定性
  - **🔧 清理循环依赖**: 修复nn.moe和triton_ops的导入问题
  - **✨ 更清晰的代码结构**: tensor.py, function.py, storage.py核心文件
  - **🚀 操作分发优化**: ops/dispatcher.py统一路由机制
- ✅ **v1.0.0** - 神经网络模块重构和损失函数扩展
  - **🏗️ 模块化重构**: nn/modules/按功能分离
  - **💯 PyTorch兼容**: 完整的损失函数集合
  - **🔧 函数增强**: 添加log_softmax, maximum, randint等API
- ✅ **v0.4.0** - 专业文档系统和性能分析工具
  - 完整的英文文档和API参考
  - 增强的基准测试套件
- ✅ **v0.3.0** - Qwen模型和混合精度训练支持
  - 完整Qwen LLM架构实现
  - 自动混合精度（AMP）支持

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
