# Qwen模型实现

## 概述

Genesis框架内置了Qwen（通义千问）大语言模型的实现，支持完整的训练和推理流程。

## 模型架构

Qwen模型基于Transformer架构，具有以下特点：

- **注意力机制**: Multi-Head Attention with RoPE (Rotary Position Embedding)
- **激活函数**: SwiGLU activation
- **层归一化**: RMSNorm
- **位置编码**: Rotary Position Embedding (RoPE)

## 快速使用

### 基础推理

```python
import genesis
from genesis.models.qwen import QwenModel, QwenConfig

# 创建模型配置
config = QwenConfig(
    vocab_size=32000,
    n_layer=24,
    n_head=16,
    n_embd=2048,
    max_seq_len=2048
)

# 创建模型
model = QwenModel(config)

# 推理
input_ids = genesis.tensor([[1, 2, 3, 4, 5]])  # [batch_size, seq_len]
output = model(input_ids)
print(f"输出形状: {output.shape}")  # [1, 5, 32000]
```

### 训练示例

```python
import genesis.optim as optim
import genesis.nn as nn

# 创建优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 训练循环
for batch in dataloader:
    input_ids, labels = batch
    
    # 前向传播
    logits = model(input_ids)
    
    # 计算损失
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 参数更新
    optimizer.step()
```

## 配置参数

### QwenConfig

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `vocab_size` | int | 32000 | 词汇表大小 |
| `n_layer` | int | 24 | Transformer层数 |
| `n_head` | int | 16 | 注意力头数 |
| `n_embd` | int | 2048 | 隐藏层维度 |
| `max_seq_len` | int | 2048 | 最大序列长度 |
| `dropout` | float | 0.1 | Dropout概率 |
| `bias` | bool | False | 是否使用偏置 |

## 性能优化

### 混合精度训练

```python
# 启用混合精度
genesis.enable_autocast = True

with genesis.autocast():
    logits = model(input_ids)
    loss = criterion(logits, labels)

# 梯度缩放
scaler = genesis.GradScaler()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 梯度检查点

```python
# 启用梯度检查点节省显存
model.gradient_checkpointing = True
```

## 应用示例

### 文本生成

```python
def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt)
    input_tensor = genesis.tensor([input_ids])
    
    with genesis.no_grad():
        for _ in range(max_length):
            logits = model(input_tensor)
            next_token = logits[0, -1].argmax()
            input_tensor = genesis.cat([input_tensor, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_tensor[0].tolist())

# 使用示例
generated = generate_text(model, tokenizer, "今天天气")
print(generated)
```

### 微调训练

参考 `apps/llm/train_sft_qwen.py` 了解完整的SFT (Supervised Fine-tuning) 实现。

## 文件结构

- `genesis/models/qwen.py` - 模型实现
- `apps/llm/qwen_model.py` - 训练配置和工具
- `apps/llm/train_sft_qwen.py` - SFT训练脚本
- `apps/llm/chat_qwen.py` - 推理聊天脚本

## 相关资源

- [Qwen官方论文](https://arxiv.org/abs/2309.16609)
- [RoPE位置编码详解](../tutorials/rope-attention.md)
- [大模型训练最佳实践](../training/llm-training.md)