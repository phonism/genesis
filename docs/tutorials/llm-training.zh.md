# 大语言模型训练指南

本全面指南涵盖了使用Genesis训练大语言模型（LLM）的全过程，从基础设置到高级优化技术。我们将以Qwen模型作为主要示例。

## 概述

Genesis为训练基于Transformer的语言模型提供了完整的框架，具有以下特性：
- Qwen模型架构实现
- 混合精度训练（FP16/BF16）
- 梯度裁剪和学习率调度
- 分布式训练支持
- 高效的检查点管理

## 前置要求

- 至少16GB显存的GPU（推荐A100/A800）
- CUDA 11.8+和相应的驱动程序
- Python 3.8+并已安装Genesis

## 快速开始

### 1. 基础Qwen模型设置

```python
import genesis
import genesis.nn as nn
from genesis.models.qwen import QwenConfig, QwenModel

# 配置模型
config = QwenConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_attention_heads=16,
    num_hidden_layers=24,
    intermediate_size=5632,
    max_position_embeddings=2048,
    dtype=genesis.float16  # 使用混合精度
)

# 创建模型
model = QwenModel(config)
print(f"模型参数量: {model.num_parameters() / 1e6:.1f}M")
```

### 2. 数据准备

```python
import genesis
from torch.utils.data import DataLoader

class TextDataset:
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 标记化并填充/截断
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # 转换为Genesis张量
        input_ids = genesis.tensor(tokens[:-1], dtype=genesis.int64)
        labels = genesis.tensor(tokens[1:], dtype=genesis.int64)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

# 加载你的数据集
dataset = TextDataset(train_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 3. 训练设置

```python
import genesis.optim as optim
import genesis.nn as nn

# 将模型移动到GPU
device = genesis.cuda()
model = model.to(device)

# 设置优化器和权重衰减
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8
)

# 带预热的学习率调度器
total_steps = len(dataloader) * num_epochs
warmup_steps = total_steps // 10

scheduler = optim.get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 损失函数
criterion = nn.CrossEntropyLoss()
```

## 训练循环实现

### 基础训练循环

```python
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将批次移动到设备
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(input_ids)
        logits = outputs.logits
        
        # 计算损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步骤
        optimizer.step()
        scheduler.step()
        
        # 累积损失
        total_loss += loss.item()
        num_batches += 1
        
        # 日志记录
        if batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()
            print(f'批次 {batch_idx}: loss={loss.item():.4f}, lr={current_lr:.2e}')
    
    return total_loss / num_batches

# 训练循环
for epoch in range(num_epochs):
    avg_loss = train_epoch(model, dataloader, optimizer, scheduler, criterion, device)
    print(f'轮次 {epoch}: 平均损失 = {avg_loss:.4f}')
    
    # 保存检查点
    if epoch % 10 == 0:
        genesis.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            f'qwen_checkpoint_epoch_{epoch}.pth'
        )
```

### 混合精度训练

```python
import genesis

def train_epoch_mixed_precision(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    
    # 启用混合精度
    genesis.enable_autocast = True
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 使用autocast进行前向传播
        with genesis.autocast():
            outputs = model(input_ids)
            logits = outputs.logits
            
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步骤
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'批次 {batch_idx}: loss={loss.item():.4f}')
    
    return total_loss / len(dataloader)
```

## 高级训练技术

### 1. 梯度累积

```python
def train_with_gradient_accumulation(model, dataloader, optimizer, scheduler, 
                                   criterion, device, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(input_ids)
        logits = outputs.logits
        
        # 计算损失并按累积步数缩放
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ) / accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 每accumulation_steps更新一次
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        if batch_idx % (100 * accumulation_steps) == 0:
            print(f'批次 {batch_idx}: loss={loss.item() * accumulation_steps:.4f}')
    
    return total_loss / len(dataloader)
```

### 2. 动态损失缩放

```python
class DynamicLossScaler:
    def __init__(self, init_scale=2**16, scale_factor=2.0, scale_window=1000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._growth_tracker = 0
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale
    
    def step(self, optimizer, has_overflow=False):
        if has_overflow:
            self.scale = max(self.scale / self.scale_factor, 1.0)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.scale_window:
                self.scale *= self.scale_factor
                self._growth_tracker = 0
        
        return not has_overflow

# 在训练循环中使用
scaler = DynamicLossScaler()

for batch in dataloader:
    # 前向传播
    loss = compute_loss(model, batch)
    scaled_loss = scaler.scale_loss(loss)
    
    # 反向传播
    optimizer.zero_grad()
    scaled_loss.backward()
    
    # 检查溢出
    has_overflow = check_gradient_overflow(model.parameters())
    
    # 反缩放和步进
    scaler.unscale_gradients(optimizer)
    should_step = scaler.step(optimizer, has_overflow)
    
    if should_step:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
```

### 3. 检查点保存和恢复训练

```python
class TrainingManager:
    def __init__(self, model, optimizer, scheduler, save_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.current_epoch = 0
        self.best_loss = float('inf')
    
    def save_checkpoint(self, epoch, loss, metrics=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'metrics': metrics or {}
        }
        
        # 保存常规检查点
        checkpoint_path = f'{self.save_dir}/checkpoint_epoch_{epoch}.pth'
        genesis.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = f'{self.save_dir}/best_model.pth'
            genesis.save(checkpoint, best_path)
            print(f"保存新的最佳模型，损失: {loss:.4f}")
        
        # 保存最新检查点
        latest_path = f'{self.save_dir}/latest_checkpoint.pth'
        genesis.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = genesis.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"从轮次 {self.current_epoch} 恢复，最佳损失: {self.best_loss:.4f}")
        return checkpoint

# 使用方法
training_manager = TrainingManager(model, optimizer, scheduler)

# 如果存在检查点则恢复
try:
    training_manager.load_checkpoint('checkpoints/latest_checkpoint.pth')
except FileNotFoundError:
    print("从头开始训练")

# 带检查点保存的训练循环
for epoch in range(training_manager.current_epoch, num_epochs):
    avg_loss = train_epoch(model, dataloader, optimizer, scheduler, criterion, device)
    
    # 保存检查点
    training_manager.save_checkpoint(epoch, avg_loss)
    
    print(f'轮次 {epoch}: 平均损失 = {avg_loss:.4f}')
```

## 模型评估和推理

### 1. 模型评估

```python
def evaluate_model(model, eval_dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with genesis.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids)
            logits = outputs.logits
            
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item() * shift_labels.numel()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = genesis.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity.item()
    }

# 评估模型
eval_metrics = evaluate_model(model, eval_dataloader, criterion, device)
print(f"评估损失: {eval_metrics['loss']:.4f}, 困惑度: {eval_metrics['perplexity']:.2f}")
```

### 2. 文本生成

```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    
    # 标记化提示词
    input_ids = tokenizer.encode(prompt)
    input_tensor = genesis.tensor([input_ids], dtype=genesis.int64).to(device)
    
    generated_ids = input_ids.copy()
    
    with genesis.no_grad():
        for _ in range(max_length):
            # 前向传播
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]  # 最后一个token的logits
            
            # 应用温度
            logits = logits / temperature
            
            # 应用top-p过滤
            sorted_logits, sorted_indices = genesis.sort(logits, descending=True)
            cumulative_probs = genesis.cumsum(genesis.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过阈值的tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # 从过滤后的分布中采样
            probs = genesis.softmax(logits, dim=-1)
            next_token = genesis.multinomial(probs, 1).item()
            
            # 添加到生成序列
            generated_ids.append(next_token)
            
            # 更新输入张量
            input_tensor = genesis.tensor([generated_ids], dtype=genesis.int64).to(device)
            
            # 检查结束token
            if next_token == tokenizer.eos_token_id:
                break
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

# 生成文本
prompt = "人工智能的未来是"
generated = generate_text(model, tokenizer, prompt, max_length=50)
print(f"生成结果: {generated}")
```

## 生产部署

### 1. 推理优化

```python
def optimize_for_inference(model, save_path):
    """为生产推理优化模型。"""
    model.eval()
    
    # 创建推理优化状态
    inference_state = {
        'model_state_dict': model.state_dict(),
        'model_config': model.config.__dict__,
        'inference_optimized': True,
        'genesis_version': genesis.__version__
    }
    
    genesis.save(inference_state, save_path)
    print(f"推理优化模型已保存到 {save_path}")

def load_for_inference(model_path, device=None):
    """加载推理优化的模型。"""
    checkpoint = genesis.load(model_path)
    config = QwenConfig(**checkpoint['model_config'])
    
    # 创建模型
    model = QwenModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if device:
        model = model.to(device)
    
    return model

# 优化并保存
optimize_for_inference(model, 'qwen_inference.pth')

# 加载用于推理
inference_model = load_for_inference('qwen_inference.pth', device=genesis.cuda())
```

### 2. 推理服务器设置

```python
class LLMInferenceServer:
    def __init__(self, model_path, tokenizer, device=None):
        self.tokenizer = tokenizer
        self.device = device or genesis.cuda()
        self.model = load_for_inference(model_path, self.device)
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_p=0.9):
        """从提示词生成文本。"""
        return generate_text(
            self.model, self.tokenizer, prompt,
            max_length=max_length, temperature=temperature, top_p=top_p
        )
    
    def batch_generate(self, prompts, max_length=100, temperature=0.8, top_p=0.9):
        """为多个提示词生成文本。"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_length, temperature, top_p)
            results.append(result)
        return results

# 创建推理服务器
server = LLMInferenceServer('qwen_inference.pth', tokenizer)

# 生成响应
responses = server.batch_generate([
    "生命的意义是什么？",
    "用简单的语言解释量子计算。",
    "写一个关于AI的短故事。"
])
```

## 性能优化技巧

### 1. 内存优化
- 对大模型使用梯度检查点
- 启用混合精度训练（FP16/BF16）
- 使用梯度累积实现大的有效批量大小
- 定期清理GPU缓存

### 2. 训练速度
- 为你的GPU使用适当的批量大小
- 如果可用，启用编译模式
- 使用多进程进行高效的数据加载
- 对训练进行性能分析以识别瓶颈

### 3. 模型质量
- 使用适当的学习率调度
- 应用梯度裁剪来稳定训练
- 密切监控训练指标
- 使用验证集防止过拟合

本指南为使用Genesis训练LLM提供了全面的基础。请根据你的具体模型大小、数据集和计算资源来调整这些技术。