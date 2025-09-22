# 随机数生成 API

Genesis 提供与 PyTorch 兼容的随机数生成 API，具有线程安全的状态管理和可重复性保证。

## 函数

### 全局随机状态管理

#### `genesis.manual_seed(seed)`

设置全局随机种子以获得可重复的结果。

**参数:**
- `seed` (int): 随机种子值

**示例:**
```python
import genesis

# 设置种子以获得可重复结果
genesis.manual_seed(42)
x = genesis.rand(100, 100)  # 可重复的随机张量
```

#### `genesis.seed()`

从当前时间或系统熵设置随机种子。

**示例:**
```python
genesis.seed()  # 从系统获取随机种子
```

#### `genesis.initial_seed()`

获取使用的初始随机种子。

**返回值:**
- int: 初始种子值

#### `genesis.get_rng_state()`

获取当前随机数生成器状态。

**返回值:**
- Tensor: 当前随机数生成器状态

**示例:**
```python
# 保存当前状态
state = genesis.get_rng_state()
# ... 执行随机操作 ...
genesis.set_rng_state(state)  # 恢复状态
```

#### `genesis.set_rng_state(new_state)`

设置随机数生成器状态。

**参数:**
- `new_state` (Tensor): 要恢复的随机数生成器状态

### 线程安全的随机生成

#### `genesis.fork_rng(devices=None, enabled=True)`

线程安全随机数生成的上下文管理器。

**参数:**
- `devices` (list, 可选): 要分叉随机数生成器状态的设备
- `enabled` (bool): 是否实际分叉 (默认: True)

**示例:**
```python
with genesis.fork_rng():
    genesis.manual_seed(999)
    # 这里的随机操作不会影响全局状态
    x = genesis.rand(10, 10)
# 全局状态在这里恢复
```

## Generator 类

### `genesis.Generator(device='cpu')`

用于控制随机状态的随机数生成器。

**参数:**
- `device` (str): 生成器的设备 (默认: 'cpu')

**方法:**

#### `generator.manual_seed(seed)`

设置生成器的随机种子。

**参数:**
- `seed` (int): 随机种子值

**示例:**
```python
gen = genesis.Generator()
gen.manual_seed(12345)

# 为特定操作使用生成器
x = genesis.rand(100, 100, generator=gen)
```

#### `generator.seed()`

从系统熵设置随机种子。

#### `generator.initial_seed()`

获取生成器的初始种子。

#### `generator.get_state()`

获取生成器的当前状态。

#### `generator.set_state(new_state)`

设置生成器的状态。

## 全局生成器

#### `genesis.default_generator`

默认的全局随机数生成器实例。

**示例:**
```python
# 访问默认生成器
gen = genesis.default_generator
state = gen.get_state()
```

## 使用示例

### 基础随机生成

```python
import genesis

# 设置种子以获得可重复结果
genesis.manual_seed(42)

# 生成随机张量
x = genesis.rand(100, 100, device='cuda')
y = genesis.randn(50, 50, device='cpu')
z = genesis.randint(0, 10, (20, 20))
```

### 高级状态管理

```python
import genesis

# 保存全局状态
saved_state = genesis.get_rng_state()

# 执行一些随机操作
genesis.manual_seed(123)
x = genesis.rand(10, 10)

# 恢复之前的状态
genesis.set_rng_state(saved_state)
y = genesis.rand(10, 10)  # 如同从未调用 seed(123)
```

### 线程安全使用

```python
import genesis
import threading

def worker():
    with genesis.fork_rng():
        genesis.manual_seed(42)
        # 每个线程都有独立的随机状态
        return genesis.rand(100, 100)

# 多个线程不会相互干扰
threads = [threading.Thread(target=worker) for _ in range(4)]
```

### 自定义生成器使用

```python
import genesis

# 创建自定义生成器
gen1 = genesis.Generator()
gen1.manual_seed(111)

gen2 = genesis.Generator() 
gen2.manual_seed(222)

# 不同的生成器产生不同的序列
x1 = genesis.rand(10, 10, generator=gen1)
x2 = genesis.rand(10, 10, generator=gen2)  # 与 x1 不同
```

## 实现说明

- 随机数生成器 API 设计为与 PyTorch 兼容
- 通过适当的状态隔离保证线程安全
- 状态管理支持全局和每个生成器的控制
- 所有随机函数都接受可选的 `generator` 参数
- 实现内部使用 NumPy 的随机数生成器
- CUDA 设备的随机生成继承自 CPU 状态管理

## 另请参阅

- [张量创建函数](genesis.md#张量创建) - 使用随机生成的函数
- [内存管理](memory.md) - 设备特定的注意事项