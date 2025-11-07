"""Streaming dataloader with online tokenization for nanochat.

This module provides an efficient streaming data loader that:
- Reads parquet files containing text data
- Tokenizes text on-the-fly using the nanochat tokenizer
- Yields training batches of (inputs, targets) for language modeling
- Supports distributed training by sharding data across ranks
"""

import os
from collections import deque
from pathlib import Path
from typing import Iterator, List, Tuple, Optional

import numpy as np
import pyarrow.parquet as pq

from tokenizer import NanoChatBPETokenizer


def get_dist_info() -> Tuple[bool, int, int, int]:
    """Get distributed training information from genesis.distributed.

    Returns:
        Tuple of (ddp_enabled, rank, local_rank, world_size)
    """
    try:
        import genesis.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            return True, rank, local_rank, world_size
    except:
        pass
    return False, 0, 0, 1


def discover_parquet_files(data_dir: Path, split: str = "train") -> List[Path]:
    """Discover parquet files for the given split.
    
    Args:
        data_dir: Directory containing parquet files
        split: Either 'train' or 'val'
    
    Returns:
        List of parquet file paths
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    if split == "train":
        if len(parquet_files) == 1:
            return parquet_files
        return parquet_files[:-1]
    else:
        return [parquet_files[-1]]


def parquets_iter_batched(
    data_dir: Path,
    split: str = "train",
    start: int = 0,
    step: int = 1
) -> Iterator[List[str]]:
    """Iterate over parquet files yielding text batches.
    
    Args:
        data_dir: Directory containing parquet files
        split: Either 'train' or 'val'
        start: Starting index for distributed training (rank)
        step: Step size for distributed training (world_size)
    
    Yields:
        Batches of text strings
    """
    parquet_files = discover_parquet_files(data_dir, split)
    
    file_idx = 0
    while True:
        for parquet_path in parquet_files:
            if file_idx % step != start:
                file_idx += 1
                continue
            
            parquet_file = pq.ParquetFile(parquet_path)
            
            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_idx, columns=["text"])
                texts = table.column("text").to_pylist()
                
                batch = []
                for text in texts:
                    if text is None:
                        continue
                    
                    if isinstance(text, bytes):
                        try:
                            text = text.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                    else:
                        text = str(text)
                    
                    text = text.strip()
                    if text:
                        batch.append(text)
                
                if batch:
                    yield batch
            
            file_idx += 1


def tokenizing_distributed_data_loader(
    B: int,
    T: int,
    tokenizer_path: Path,
    data_dir: Path,
    split: str = "train",
    tokenizer_batch_size: int = 128,
    device: str = "cuda"
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stream pretraining text from parquet files, tokenize, yield training batches.
    
    Args:
        B: Batch size
        T: Sequence length
        tokenizer_path: Path to tokenizer.json file
        data_dir: Directory containing parquet files
        split: Either 'train' or 'val'
        tokenizer_batch_size: Batch size for tokenization
        device: Target device (cuda or cpu)
    
    Yields:
        Tuple of (inputs, targets) as numpy arrays of shape (B, T)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1
    
    tokenizer = NanoChatBPETokenizer.load(tokenizer_path)
    bos_id = tokenizer._bos_id if tokenizer._bos_id is not None else 0
    
    token_buffer = deque()
    
    def document_batches():
        """Infinite iterator over document batches."""
        while True:
            for batch in parquets_iter_batched(
                data_dir=data_dir,
                split=split,
                start=ddp_rank,
                step=ddp_world_size
            ):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    
    batches = document_batches()
    
    while True:
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            
            for doc in doc_batch:
                tokens = tokenizer.encode(doc, add_special_tokens=True)
                token_buffer.extend(tokens)
        
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        
        tokens_array = np.array(tokens, dtype=np.int32)
        inputs = tokens_array[:-1].reshape(B, T)
        targets = tokens_array[1:].reshape(B, T)
        
        yield inputs, targets
