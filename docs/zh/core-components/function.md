# 函数系统

Genesis的函数系统为自动微分提供基础，定义了操作在前向传播中如何执行以及在反向传播中如何计算梯度。

## 📋 概述

函数系统围绕`Function`基类构建，封装了：
- 前向计算逻辑
- 反向梯度计算
- 用于存储中间值的上下文管理
- 与自动微分引擎的集成

## 🏗️ 架构

```mermaid
graph TB
    subgraph "函数系统"
        A[Function基类] --> B[apply()方法]
        A --> C[forward()方法]
        A --> D[backward()方法]
        E[Context] --> F[save_for_backward()]
        E --> G[saved_tensors]
    end

    subgraph "自动微分集成"
        B --> H[计算图]
        H --> I[梯度流]
        I --> J[反向传播]
    end

    subgraph "内置函数"
        K[AddFunction] --> A
        L[MulFunction] --> A
        M[MatMulFunction] --> A
        N[ReluFunction] --> A
    end

    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style H fill:#e8f5e8
```

## 🎯 核心概念

### Function基类
`Function`类为所有操作提供接口：

```python
class Function:
    """所有自动微分函数的基类。"""

    @staticmethod
    def apply(*args):
        """应用具有自动微分支持的函数。"""
        ctx = Context()

        # 前向传播
        result = cls.forward(ctx, *args)

        # 如果任何输入需要梯度，设置反向传播
        if any(tensor.requires_grad for tensor in args if isinstance(tensor, Tensor)):
            result.set_creator(ctx, cls.backward)

        return result

    @staticmethod
    def forward(ctx, *args):
        """计算前向传播。必须由子类实现。"""
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """计算反向传播。必须由子类实现。"""
        raise NotImplementedError
```

### 上下文管理
`Context`类管理反向计算所需的信息：

```python
class Context:
    """用于存储反向传播期间所需信息的上下文。"""

    def __init__(self):
        self.saved_tensors = []
        self.saved_variables = {}

    def save_for_backward(self, *tensors):
        """保存张量以供反向传播使用。"""
        self.saved_tensors.extend(tensors)

    def save_variable(self, name, value):
        """保存变量以供反向传播使用。"""
        self.saved_variables[name] = value
```

## 💻 实现示例

### 基本算术函数
```python
class AddFunction(Function):
    """支持梯度的加法函数。"""

    @staticmethod
    def forward(ctx, a, b):
        """前向传播：计算a + b。"""
        # 加法不需要保存输入
        return genesis.ops.add(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：梯度不变流动。"""
        return grad_output, grad_output

# 使用
add = AddFunction.apply
c = add(a, b)  # 等价于支持自动微分的 a + b
```

### 矩阵乘法函数
```python
class MatMulFunction(Function):
    """支持梯度的矩阵乘法。"""

    @staticmethod
    def forward(ctx, a, b):
        """前向传播：计算 a @ b。"""
        ctx.save_for_backward(a, b)
        return genesis.ops.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：使用链式法则计算梯度。"""
        a, b = ctx.saved_tensors

        grad_a = genesis.ops.matmul(grad_output, b.transpose(-2, -1))
        grad_b = genesis.ops.matmul(a.transpose(-2, -1), grad_output)

        return grad_a, grad_b

# 使用
matmul = MatMulFunction.apply
c = matmul(a, b)  # 等价于支持自动微分的 a @ b
```

### 带上下文的激活函数
```python
class ReluFunction(Function):
    """支持梯度的ReLU激活。"""

    @staticmethod
    def forward(ctx, input):
        """前向传播：计算 max(0, input)。"""
        output = genesis.ops.maximum(input, 0)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：负输入的梯度为0。"""
        input, = ctx.saved_tensors
        mask = input > 0
        return grad_output * mask

# 使用
relu = ReluFunction.apply
activated = relu(x)
```

## 🚀 高级特性

### 原地操作
```python
class AddInplaceFunction(Function):
    """原地加法函数。"""

    @staticmethod
    def forward(ctx, a, b):
        """前向传播：原地修改a。"""
        ctx.save_variable('original_a', a.clone())
        a.add_(b)
        return a

    @staticmethod
    def backward(ctx, grad_output):
        """原地操作的反向传播。"""
        return grad_output, grad_output
```

### 多输出函数
```python
class SplitFunction(Function):
    """返回多个输出的函数。"""

    @staticmethod
    def forward(ctx, input, split_sizes):
        """将输入张量分割成多个部分。"""
        ctx.save_variable('split_sizes', split_sizes)
        return genesis.ops.split(input, split_sizes)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """从多个输出连接梯度。"""
        grad_input = genesis.ops.cat(grad_outputs, dim=0)
        return grad_input, None  # split_sizes没有梯度
```

### 自定义上下文变量
```python
class ScaleFunction(Function):
    """通过常数因子缩放张量。"""

    @staticmethod
    def forward(ctx, input, scale_factor):
        """通过常数因子缩放输入。"""
        ctx.save_variable('scale_factor', scale_factor)
        return input * scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        """通过相同因子缩放梯度。"""
        scale_factor = ctx.saved_variables['scale_factor']
        return grad_output * scale_factor, None
```

## 🔧 与操作集成

### 向调度器注册函数
```python
# 向操作调度器注册函数
genesis.ops.register_function('add', AddFunction.apply)
genesis.ops.register_function('matmul', MatMulFunction.apply)
genesis.ops.register_function('relu', ReluFunction.apply)

# 现在操作自动使用注册的函数
x = genesis.tensor([1, 2, 3], requires_grad=True)
y = genesis.tensor([4, 5, 6], requires_grad=True)
z = x + y  # 自动使用AddFunction
```

### 自定义操作定义
```python
def custom_operation(input, param):
    """使用Function定义自定义操作。"""
    return CustomFunction.apply(input, param)

# 注册为操作
genesis.ops.register_operation('custom_op', custom_operation)

# 像任何其他操作一样使用
result = genesis.custom_op(tensor, param)
```

## 📊 性能考虑

### 内存效率
```python
class EfficientFunction(Function):
    """内存高效的函数实现。"""

    @staticmethod
    def forward(ctx, input):
        # 只保存反向所需的内容
        ctx.save_for_backward(input.detach())  # 分离以避免递归梯度

        # 高效计算结果
        result = efficient_computation(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 高效计算梯度
        return efficient_gradient_computation(input, grad_output)
```

### 数值稳定性
```python
class StableFunction(Function):
    """数值稳定的函数实现。"""

    @staticmethod
    def forward(ctx, input):
        # 使用数值稳定的计算
        output = stable_computation(input)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        # 使用稳定的梯度计算
        return stable_gradient(input, output, grad_output)
```

## 🔍 调试和测试

### 函数测试
```python
def test_function_gradients():
    """测试函数梯度计算。"""
    x = genesis.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # 测试前向传播
    y = CustomFunction.apply(x)

    # 测试反向传播
    y.backward(genesis.tensor([1.0, 1.0, 1.0]))

    # 检查梯度
    assert x.grad is not None
    print(f"梯度：{x.grad}")

# 数值梯度检查
def numerical_gradient_check(func, input, eps=1e-5):
    """使用数值微分检查梯度。"""
    # 数值梯度检查的实现
    pass
```

### 调试上下文
```python
class DebugFunction(Function):
    """带调试信息的函数。"""

    @staticmethod
    def forward(ctx, input):
        print(f"前向：输入形状 = {input.shape}")
        ctx.save_for_backward(input)
        result = computation(input)
        print(f"前向：输出形状 = {result.shape}")
        return result

    @staticmethod
    def backward(ctx, grad_output):
        print(f"反向：grad_output形状 = {grad_output.shape}")
        input, = ctx.saved_tensors
        grad_input = gradient_computation(input, grad_output)
        print(f"反向：grad_input形状 = {grad_input.shape}")
        return grad_input
```

## 🔗 参见

- [张量系统](tensor.md) - 张量类和自动微分集成
- [核心组件概述](index.md) - 整体系统架构
- [自动微分](autograd.md) - 详细的自动微分系统
- [自定义操作指南](../tutorials/custom-ops.md) - 创建自定义操作