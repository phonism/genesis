# 存储层

Genesis的存储层提供了一个抽象接口，用于管理不同设备上的张量数据存储。

## 📋 概述

存储层是Genesis v2.0架构的关键组件，它：
- 抽象了设备特定的存储实现
- 管理内存生命周期
- 提供设备间的高效数据传输
- 支持各种数据类型和内存布局

## 🏗️ 架构

```mermaid
graph TB
    subgraph "存储抽象"
        A[Storage接口] --> B[create_storage()]
        A --> C[storage.to()]
        A --> D[storage.copy_()]
    end

    subgraph "设备实现"
        E[CPUStorage] --> A
        F[CUDAStorage] --> A
        G[FutureStorage] --> A
    end

    subgraph "内存管理"
        H[内存分配] --> I[内存池]
        H --> J[引用计数]
        H --> K[垃圾回收]
    end

    style A fill:#e1f5fe
    style E fill:#e8f5e9
    style F fill:#ffeb3b
```

## 🎯 核心概念

### Storage接口
所有存储实现的基础接口：

```python
class Storage:
    """张量数据存储的抽象接口。"""

    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._data = None

    def to(self, device):
        """传输存储到另一个设备。"""
        raise NotImplementedError

    def copy_(self, other):
        """从另一个存储原地复制。"""
        raise NotImplementedError

    def clone(self):
        """创建存储的深拷贝。"""
        raise NotImplementedError

    @property
    def data_ptr(self):
        """获取底层数据指针。"""
        return self._data
```

### 存储创建
创建适合设备的存储：

```python
def create_storage(data, device, dtype=None):
    """为给定设备创建存储。"""
    if device.type == 'cpu':
        return CPUStorage(data, dtype)
    elif device.type == 'cuda':
        return CUDAStorage(data, dtype)
    else:
        raise ValueError(f"不支持的设备类型：{device.type}")

# 使用
storage = create_storage([1, 2, 3], genesis.device("cuda"))
```

## 💻 存储实现

### CPU存储
CPU设备的存储实现：

```python
class CPUStorage(Storage):
    """CPU张量存储。"""

    def __init__(self, data, dtype=None):
        # 转换数据为PyTorch张量
        if isinstance(data, torch.Tensor):
            self._data = data.cpu()
        else:
            self._data = torch.tensor(data, dtype=dtype)

        super().__init__(
            shape=self._data.shape,
            dtype=self._data.dtype,
            device=genesis.device("cpu")
        )

    def to(self, device):
        """传输到另一个设备。"""
        if device.is_cpu:
            return self

        # 传输到GPU
        cuda_data = self._data.cuda(device.id)
        return CUDAStorage.from_tensor(cuda_data)

    def copy_(self, other):
        """从另一个存储复制。"""
        if isinstance(other, CPUStorage):
            self._data.copy_(other._data)
        else:
            # 从GPU复制
            self._data.copy_(other.to_cpu()._data)
```

### CUDA存储
GPU设备的存储实现：

```python
class CUDAStorage(Storage):
    """CUDA张量存储。"""

    def __init__(self, data, dtype=None, device_id=0):
        self.device_id = device_id
        self._allocate_memory(data, dtype)

        super().__init__(
            shape=self._shape,
            dtype=self._dtype,
            device=genesis.device(f"cuda:{device_id}")
        )

    def _allocate_memory(self, data, dtype):
        """分配CUDA内存。"""
        # 使用内存池分配
        size = self._compute_size(data, dtype)
        self._data_ptr = CUDAMemoryPool.allocate(size)

        # 复制数据到GPU
        self._copy_to_gpu(data)

    def to(self, device):
        """传输到另一个设备。"""
        if device.is_cuda and device.id == self.device_id:
            return self

        if device.is_cpu:
            # 传输到CPU
            cpu_data = self._copy_to_cpu()
            return CPUStorage(cpu_data)
        else:
            # 传输到另一个GPU
            return self._transfer_to_gpu(device.id)
```

## 🚀 高级特性

### 内存视图
高效的内存视图无需复制：

```python
class StorageView(Storage):
    """原始存储的视图。"""

    def __init__(self, base_storage, offset, shape, stride):
        self.base = base_storage
        self.offset = offset
        self.stride = stride

        super().__init__(
            shape=shape,
            dtype=base_storage.dtype,
            device=base_storage.device
        )

    @property
    def data_ptr(self):
        """获取带偏移的数据指针。"""
        return self.base.data_ptr + self.offset

    def is_contiguous(self):
        """检查视图是否连续。"""
        expected_stride = self._compute_contiguous_stride(self.shape)
        return self.stride == expected_stride
```

### 共享内存
进程间共享的存储：

```python
class SharedStorage(Storage):
    """跨进程共享内存存储。"""

    def __init__(self, shape, dtype, shared_memory_name=None):
        # 创建或连接到共享内存
        if shared_memory_name:
            self.shm = SharedMemory(name=shared_memory_name)
        else:
            size = self._compute_size(shape, dtype)
            self.shm = SharedMemory(create=True, size=size)

        super().__init__(shape, dtype, genesis.device("cpu"))

    def close(self):
        """关闭共享内存连接。"""
        self.shm.close()

    def unlink(self):
        """移除共享内存。"""
        self.shm.unlink()
```

### 内存映射存储
大型数据集的内存映射文件：

```python
class MappedStorage(Storage):
    """内存映射文件存储。"""

    def __init__(self, filename, shape, dtype, mode='r'):
        self.filename = filename
        self.mode = mode

        # 创建内存映射
        self.mmap = np.memmap(
            filename,
            dtype=dtype,
            mode=mode,
            shape=shape
        )

        super().__init__(shape, dtype, genesis.device("cpu"))

    def flush(self):
        """将更改刷新到磁盘。"""
        if self.mode != 'r':
            self.mmap.flush()

    def close(self):
        """关闭内存映射。"""
        del self.mmap
```

## 💾 内存管理

### 引用计数
自动内存管理通过引用计数：

```python
class RefCountedStorage(Storage):
    """带引用计数的存储。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_count = 1

    def incref(self):
        """增加引用计数。"""
        self.ref_count += 1

    def decref(self):
        """减少引用计数并在需要时释放。"""
        self.ref_count -= 1
        if self.ref_count == 0:
            self._deallocate()

    def _deallocate(self):
        """释放底层内存。"""
        if self.device.is_cuda:
            CUDAMemoryPool.deallocate(self.data_ptr)
        else:
            del self._data
```

### 内存池集成
与内存池的高效集成：

```python
class PooledStorage(Storage):
    """使用内存池的存储。"""

    def __init__(self, shape, dtype, device):
        super().__init__(shape, dtype, device)

        # 从池中分配
        size = self._compute_size()
        if device.is_cuda:
            self.pool = CUDAMemoryPool.get_instance()
        else:
            self.pool = CPUMemoryPool.get_instance()

        self._block = self.pool.allocate(size)

    def __del__(self):
        """返回内存到池。"""
        if hasattr(self, '_block'):
            self.pool.deallocate(self._block)
```

## 🔧 存储操作

### 数据类型转换
在存储之间转换数据类型：

```python
def convert_dtype(storage, target_dtype):
    """转换存储到不同的数据类型。"""
    if storage.dtype == target_dtype:
        return storage

    # 创建具有新数据类型的新存储
    if storage.device.is_cuda:
        # GPU上转换
        converted_data = cuda_convert_dtype(
            storage.data_ptr,
            storage.dtype,
            target_dtype,
            storage.shape
        )
        return CUDAStorage.from_data(converted_data, target_dtype)
    else:
        # CPU上转换
        converted_data = storage._data.to(target_dtype)
        return CPUStorage(converted_data)
```

### 存储复制
高效的存储复制操作：

```python
def copy_storage(src, dst):
    """高效复制存储。"""
    # 同设备复制
    if src.device == dst.device:
        if src.device.is_cuda:
            cuda_copy(src.data_ptr, dst.data_ptr, src.size)
        else:
            dst._data.copy_(src._data)
    else:
        # 跨设备复制
        if src.device.is_cuda and dst.device.is_cpu:
            # GPU到CPU
            cuda_to_cpu(src.data_ptr, dst._data.data_ptr(), src.size)
        elif src.device.is_cpu and dst.device.is_cuda:
            # CPU到GPU
            cpu_to_cuda(src._data.data_ptr(), dst.data_ptr, src.size)
        else:
            # GPU到GPU
            cuda_to_cuda(src.data_ptr, dst.data_ptr, src.size)
```

## 📊 性能优化

### 内存对齐
确保最佳性能的内存对齐：

```python
def aligned_storage(shape, dtype, device, alignment=64):
    """创建对齐的存储以获得最佳性能。"""
    size = compute_size(shape, dtype)

    # 对齐大小到边界
    aligned_size = (size + alignment - 1) // alignment * alignment

    # 分配对齐的内存
    if device.is_cuda:
        ptr = cuda_aligned_alloc(aligned_size, alignment)
        return CUDAStorage.from_ptr(ptr, shape, dtype)
    else:
        # 为CPU使用对齐的分配器
        data = torch.empty(aligned_size // dtype.itemsize, dtype=dtype)
        return CPUStorage(data.view(shape))
```

### 批量操作
优化的批量存储操作：

```python
class BatchStorage:
    """多个存储的批量操作。"""

    def __init__(self, storages):
        self.storages = storages

    def batch_copy(self, targets):
        """批量复制到目标存储。"""
        if all(s.device.is_cuda for s in self.storages):
            # 使用CUDA流进行并行复制
            streams = [cuda.Stream() for _ in self.storages]

            for storage, target, stream in zip(self.storages, targets, streams):
                with cuda.stream(stream):
                    copy_storage(storage, target)

            # 同步所有流
            for stream in streams:
                stream.synchronize()
        else:
            # 顺序CPU复制
            for storage, target in zip(self.storages, targets):
                copy_storage(storage, target)
```

## 🔍 调试和监控

### 存储检查
用于调试的存储检查工具：

```python
def inspect_storage(storage):
    """检查存储属性。"""
    info = {
        'shape': storage.shape,
        'dtype': storage.dtype,
        'device': storage.device,
        'size_bytes': storage.size_bytes(),
        'is_contiguous': storage.is_contiguous(),
        'data_ptr': hex(storage.data_ptr) if storage.data_ptr else None,
    }

    if storage.device.is_cuda:
        info['memory_pool'] = storage.pool_info()

    return info

# 使用
storage = create_storage([1, 2, 3], genesis.device("cuda"))
print(inspect_storage(storage))
```

### 内存泄漏检测
检测存储内存泄漏：

```python
class StorageTracker:
    """跟踪存储分配和释放。"""

    def __init__(self):
        self.active_storages = {}
        self.total_allocated = 0

    def register(self, storage):
        """注册新存储。"""
        self.active_storages[id(storage)] = {
            'storage': weakref.ref(storage),
            'size': storage.size_bytes(),
            'timestamp': time.time()
        }
        self.total_allocated += storage.size_bytes()

    def unregister(self, storage):
        """注销存储。"""
        storage_id = id(storage)
        if storage_id in self.active_storages:
            info = self.active_storages.pop(storage_id)
            self.total_allocated -= info['size']

    def check_leaks(self):
        """检查潜在的内存泄漏。"""
        leaks = []
        for storage_id, info in self.active_storages.items():
            if info['storage']() is None:
                # 存储已被垃圾回收但未注销
                leaks.append(info)
        return leaks
```

## 🔗 参见

- [设备抽象](device.md) - 设备管理系统
- [内存管理](../backends/memory.md) - 高级内存管理
- [后端系统](../backends/index.md) - 后端实现
- [张量系统](tensor.md) - 张量类和存储集成