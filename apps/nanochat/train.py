"""
Training script for nanochat base model from scratch.

Supports both single-GPU and multi-GPU distributed training.

Single GPU usage:
    python train.py

Multi-GPU usage:
    torchrun --nproc_per_node=2 train.py --ddp
    # or
    python -m torch.distributed.launch --nproc_per_node=2 train.py --ddp
"""

import sys
sys.path.append("../../")
import os
import time
import random
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# Backend selection: set NANOCHAT_BACKEND=torch to use PyTorch
BACKEND = os.environ.get("NANOCHAT_BACKEND", "genesis")

if BACKEND == "torch":
    print("Using PyTorch backend")
    import torch
    from torch import amp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
    genesis = torch
    nn = torch.nn
    F = torch.nn.functional
    genesis.optim = torch.optim
    dist = torch.distributed
else:
    print("Using Genesis backend")
    import genesis
    import genesis.distributed as dist
    from genesis import nn, amp
    from genesis.nn.parallel import DistributedDataParallel as DDP
    import genesis.nn.functional as F
    # Import torch profiler for Genesis profiling
    import torch
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from tokenizer import NanoChatBPETokenizer
from dataloader import tokenizing_distributed_data_loader

# Import model from genesis.models.transformers (supports both backends)
from genesis.models.transformers.dense import QwenModel
from genesis.models.transformers.config import TransformerConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train nanochat model")
    parser.add_argument("--ddp", action="store_true", help="Enable distributed data parallel training")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--block-size", type=int, default=2048, help="Sequence length")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval in steps")
    parser.add_argument("--save-interval", type=int, default=100, help="Checkpoint save interval in steps")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (None for unlimited)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision training")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                        help="Mixed precision dtype (fp16 or bf16)")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--profile-steps", type=int, default=5, help="Number of steps to profile")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    genesis.manual_seed(seed)


def setup_ddp() -> tuple[bool, int, int, int]:
    """Setup distributed data parallel training.

    Returns:
        Tuple of (ddp_enabled, rank, local_rank, world_size)
    """
    # Check if launched with torchrun/distributed launcher
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device for this process FIRST (required before init_process_group)
        # This prevents hangs and excessive memory utilization on GPU:0
        genesis.cuda.set_device(local_rank)

        # Initialize process group
        dist.init_process_group(backend="nccl")

        print(f"DDP initialized: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


@genesis.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: "Iterator",
    device: str,
    num_batches: int = 20
) -> float:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        data_loader: Data loader for validation set
        device: Device to run evaluation on
        num_batches: Number of batches to evaluate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    for i, (inputs_np, targets_np) in enumerate(data_loader):
        if i >= num_batches:
            break

        # Convert numpy arrays to genesis tensors
        inputs = genesis.tensor(inputs_np, device=genesis.device(device), dtype=genesis.int64)
        targets = genesis.tensor(targets_np, device=genesis.device(device), dtype=genesis.int64)

        # Forward pass
        logits = model(inputs)

        # Compute loss
        B, T, C = logits.shape
        logits = genesis.reshape(logits, (B * T, C))
        targets = genesis.reshape(targets, (B * T,))
        loss = nn.CrossEntropyLoss()(logits, targets)

        total_loss += float(loss.item() if hasattr(loss, "item") else loss.data)

    model.train()
    return total_loss / num_batches


# Parse arguments
args = parse_args()
set_seed(args.seed)

# Setup distributed training if requested
ddp_enabled, rank, local_rank, world_size = setup_ddp() if args.ddp else (False, 0, 0, 1)

# Training hyperparameters
batch_size = args.batch_size
block_size = args.block_size
learning_rate = args.learning_rate
accumulation_steps = args.accumulation_steps
tokenizer_batch_size = 128
eval_interval = args.eval_interval
eval_batches = 20
save_interval_steps = args.save_interval  # Save every N steps

# Set device based on DDP configuration
if ddp_enabled:
    device = f"cuda:{local_rank}"
else:
    device = "cuda"

# Load tokenizer
tokenizer_path = Path(__file__).parent / "checkpoints" / "tokenizer" / "tokenizer.json"
if not tokenizer_path.exists():
    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please train tokenizer first.")

tokenizer = NanoChatBPETokenizer.load(tokenizer_path)
if rank == 0:
    print(f"Loaded tokenizer with vocab_size: {tokenizer.vocab_size}")

# Initialize 0.5B model config using TransformerConfig
# Modified to use head_dim=64 for Triton FusedAttention support
config = TransformerConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=block_size,
    num_hidden_layers=28,
    num_attention_heads=24,
    hidden_size=1536,  # 64 * 16 for head_dim=64
    intermediate_size=8960,  # Standard 4x hidden size
    num_key_value_heads=24,  # Dense attention (no GQA)
    head_dim=64,  # Triton FusedAttention requires head_dim in [16, 32, 64, 128]
    norm_eps=1e-6,
    rope_theta=10000.0
)

# Create model and move to device
model = QwenModel(config)
model.to(genesis.device(device))

# Wrap model with DDP if enabled
if ddp_enabled:
    model = DDP(model, device_ids=[local_rank])
    if rank == 0:
        print(f"Model wrapped with DDP on {world_size} GPUs")

    # CRITICAL: Barrier to ensure all ranks finish initialization before training
    # This is standard PyTorch DDP practice to avoid race conditions
    dist.barrier()
    if rank == 0:
        print("All ranks synchronized after DDP initialization")

optimizer = genesis.optim.AdamW(model.parameters(), lr=learning_rate)

# Convert dtype string to genesis dtype
amp_dtype = genesis.float16 if args.dtype == "fp16" else genesis.bfloat16

# Initialize GradScaler for AMP
# Note: bfloat16 doesn't need gradient scaling (large dynamic range)
use_scaler = args.amp and args.dtype == "fp16"
scaler = amp.GradScaler() if use_scaler else None

if rank == 0:
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training config: batch_size={batch_size}, block_size={block_size}, lr={learning_rate}")
    if args.amp:
        print(f"Mixed precision training: enabled ({args.dtype})")
    if ddp_enabled:
        print(f"Global batch size: {batch_size * world_size * accumulation_steps}")

# Setup data directories
data_dir = Path(__file__).parent / "base_data"
checkpoint_dir = Path(__file__).parent / "checkpoints" / "model"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Create data loaders
train_loader = tokenizing_distributed_data_loader(
    B=batch_size,
    T=block_size,
    tokenizer_path=tokenizer_path,
    data_dir=data_dir,
    split="train",
    tokenizer_batch_size=tokenizer_batch_size,
    device=device
)

val_loader = tokenizing_distributed_data_loader(
    B=batch_size,
    T=block_size,
    tokenizer_path=tokenizer_path,
    data_dir=data_dir,
    split="val",
    tokenizer_batch_size=tokenizer_batch_size,
    device=device
)

# Define training step function
def train_step(model, inputs, targets):
    """Training step."""
    logits = model(inputs)
    B, T, C = logits.shape
    logits = genesis.reshape(logits, (B * T, C))
    targets = genesis.reshape(targets, (B * T,))
    loss = nn.CrossEntropyLoss()(logits, targets)
    return loss

# Training loop
start_time = time.time()
total_cnt = 0
batch_loss = 0.0
save_interval = accumulation_steps * save_interval_steps

# Initialize profiler if requested
profiler_ctx = None
if args.profile and rank == 0:
    print(f"Profiling enabled: will profile {args.profile_steps} steps")
    profiler_ctx = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,  # Disable to reduce file size
        profile_memory=False,  # Disable to reduce file size
        with_stack=False  # Disable to reduce file size
    )
    profiler_ctx.__enter__()

if rank == 0:
    print("Starting training...")
    print(f"Evaluating every {eval_interval} steps")
    print(f"Saving checkpoints every {save_interval // accumulation_steps} steps")
    if args.max_steps is not None:
        print(f"Training will stop after {args.max_steps} steps")

for inputs_np, targets_np in train_loader:
    # Convert numpy arrays to genesis tensors
    inputs = genesis.tensor(inputs_np, device=genesis.device(device), dtype=genesis.int64)
    targets = genesis.tensor(targets_np, device=genesis.device(device), dtype=genesis.int64)

    # Forward pass with optional mixed precision
    if args.amp:
        with amp.autocast('cuda', dtype=amp_dtype):
            loss = train_step(model, inputs, targets)
            loss = loss / accumulation_steps

        # Backward pass
        if scaler is not None:
            # Float16: use gradient scaling
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
        else:
            # BFloat16: no scaling needed
            loss.backward()
    else:
        # Standard fp32 training
        loss = train_step(model, inputs, targets)
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

    # Accumulate loss (convert to Python float for printing)
    batch_loss += float(loss.item() if hasattr(loss, "item") else loss.data)

    # Update parameters after accumulation steps
    if (total_cnt + 1) % accumulation_steps == 0:
        if scaler is not None:
            # Float16: use GradScaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # BFloat16 or FP32: standard optimizer step
            optimizer.step()
            optimizer.zero_grad()

        step_num = (total_cnt + 1) // accumulation_steps

        # Print progress (only rank 0)
        if rank == 0:
            elapsed = time.time() - start_time
            # Account for world_size in tokens/sec calculation for DDP
            total_tokens = batch_size * block_size * accumulation_steps * (world_size if ddp_enabled else 1)
            tokens_per_sec = total_tokens / elapsed
            print(f"Step {step_num}: train_loss={batch_loss:.4f}, time={elapsed:.2f}s, tokens/s={tokens_per_sec:.0f}")

        batch_loss = 0.0
        start_time = time.time()

        # Evaluate on validation set periodically (only rank 0)
        if step_num % eval_interval == 0 and rank == 0:
            print(f"Running validation at step {step_num}...")
            val_loss = evaluate_model(model, val_loader, device, eval_batches)
            print(f"Step {step_num}: val_loss={val_loss:.4f}")

        # Save checkpoint periodically (only rank 0)
        if (total_cnt + 1) % save_interval == 0 and rank == 0:
            # Get underlying model for checkpoint saving
            # DDP wraps the model, access it via .module attribute
            checkpoint_model = model.module if ddp_enabled else model
            state_dict = checkpoint_model.state_dict()
            save_path = checkpoint_dir / f"model_step_{step_num}.pth"
            genesis.save(state_dict, str(save_path))
            print(f"Checkpoint saved to {save_path}")

        # Check if we've reached max_steps
        if args.max_steps is not None and step_num >= args.max_steps:
            if rank == 0:
                print(f"Reached max_steps={args.max_steps}, stopping training.")
            break

        # Profiler step (after optimizer update)
        if profiler_ctx is not None:
            profiler_ctx.step()
            # Stop profiling after specified steps
            if step_num >= args.profile_steps:
                break

    total_cnt += 1

# Export profiler results
if profiler_ctx is not None:
    profiler_ctx.__exit__(None, None, None)
    # Export Chrome trace
    trace_file = f"genesis_{args.dtype}_trace.json"
    profiler_ctx.export_chrome_trace(trace_file)
    print(f"\n{'='*60}")
    print(f"Profiling complete! Results exported to:")
    print(f"  - Chrome trace: {trace_file}")
    print(f"    View at: chrome://tracing")
    print(f"{'='*60}\n")

if rank == 0:
    print("Training completed!")

# Cleanup DDP
if ddp_enabled:
    dist.destroy_process_group()
