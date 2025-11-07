"""Command line entry point for training the nanochat tokenizer."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import pyarrow.parquet as pq

from tokenizer import NanoChatBPETokenizer, NanoChatTokenizerConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for nanochat using tiktoken.")
    default_data_dir = Path(__file__).parent / "base_data"
    default_output_dir = Path(__file__).parent / "checkpoints" / "tokenizer"
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing parquet shards with a 'text' column.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split to use for tokenizer training.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_000,
        help="Desired vocabulary size for the tokenizer.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000_000_000,
        help="Maximum number of documents to draw for tokenizer training (-1 means unlimited).",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Use every Nth sample to reduce correlation (default: 1, use every sample).",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Enable reservoir sampling instead of taking the first max-samples documents.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reservoir sampling or shuffling.",
    )
    parser.add_argument(
        "--regex-pattern",
        type=str,
        default=None,
        help="Optional custom regex pattern for the tokenizer pre-tokenizer.",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"],
        help="Special tokens reserved at the front of the vocabulary.",
    )
    parser.add_argument(
        "--no-add-bos",
        action="store_true",
        help="Disable automatic BOS injection during encoding.",
    )
    parser.add_argument(
        "--no-add-eos",
        action="store_true",
        help="Disable automatic EOS injection during encoding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where the trained tokenizer and config will be saved.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="tokenizer.json",
        help="Filename used when saving the tokenizer json.",
    )
    return parser.parse_args()


def discover_parquet_files(data_dir: Path) -> List[Path]:
    """Return sorted parquet shard paths under the provided directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    parquet_paths = sorted(data_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards (*.parquet) found in {data_dir}.")
    return parquet_paths


def select_split(parquet_paths: Sequence[Path], split: str) -> List[Path]:
    """Return shard paths for the requested split."""
    if split == "train":
        if len(parquet_paths) == 1:
            return list(parquet_paths)
        return list(parquet_paths[:-1])
    return [parquet_paths[-1]]


def iter_parquet_texts(parquet_paths: Sequence[Path]) -> Iterator[str]:
    """Yield text entries from a list of parquet shards."""
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index, columns=["text"])
            texts = table.column("text").to_pylist()
            for text in texts:
                if text is None:
                    continue
                if isinstance(text, bytes):
                    try:
                        decoded = text.decode("utf-8", errors="ignore")
                    except Exception:  # pragma: no cover - extremely unlikely
                        continue
                else:
                    decoded = str(text)
                stripped = decoded.strip()
                if stripped:
                    yield stripped


def gather_training_samples(
    parquet_paths: Sequence[Path],
    max_samples: Optional[int],
    sample_stride: int,
    random_sample: bool,
    seed: int,
) -> List[str]:
    """Collect text samples for tokenizer training."""
    if sample_stride <= 0:
        raise ValueError("sample_stride must be a positive integer.")
    rng = random.Random(seed)
    samples: List[str] = []
    seen = 0
    for index, text in enumerate(iter_parquet_texts(parquet_paths)):
        if index % sample_stride != 0:
            continue
        if not random_sample:
            samples.append(text)
            if max_samples is not None and len(samples) >= max_samples:
                break
            continue
        if max_samples is None:
            samples.append(text)
            continue
        seen += 1
        if len(samples) < max_samples:
            samples.append(text)
            continue
        replacement_index = rng.randrange(seen)
        if replacement_index < max_samples:
            samples[replacement_index] = text
    if max_samples is not None and not random_sample:
        samples = samples[:max_samples]
    if random_sample and max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]
    return samples


def main() -> None:
    """Entrypoint for tokenizer training."""
    args = parse_args()
    max_samples = None if args.max_samples < 0 else args.max_samples
    parquet_paths = discover_parquet_files(args.data_dir)
    split_paths = select_split(parquet_paths, args.split)
    print(f"Found {len(split_paths)} parquet shard(s) for split '{args.split}'.")
    samples = gather_training_samples(
        split_paths,
        max_samples=max_samples,
        sample_stride=args.sample_stride,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    if not samples:
        raise RuntimeError("No training samples collected for tokenizer training.")
    print(f"Collected {len(samples)} samples for tokenizer training.")

    config = NanoChatTokenizerConfig(
        vocab_size=args.vocab_size,
        special_tokens=tuple(args.special_tokens),
        regex_pattern=args.regex_pattern,
        add_bos_token=not args.no_add_bos,
        add_eos_token=not args.no_add_eos,
    )
    tokenizer = NanoChatBPETokenizer.train(samples, config=config)
    print(f"Trained tokenizer with vocab size: {tokenizer.vocab_size}")
    save_path = tokenizer.save(args.output_dir, filename=args.output_name)
    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":
    main()
