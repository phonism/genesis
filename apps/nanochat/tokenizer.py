"""BPE tokenizer utilities for the nanochat project using HuggingFace tokenizers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@dataclass
class NanoChatTokenizerConfig:
    """Configuration values used when training the nanochat tokenizer.

    Attributes:
        vocab_size: Desired size of the tokenizer vocabulary.
        special_tokens: Ordered list of special tokens that must be reserved in
            the vocabulary. They will be appended to the learned tokens in the
            provided order.
        regex_pattern: Optional regular expression used for the pre-tokenizer
            during BPE training. When omitted, tiktoken uses its default GPT-2
            style pattern.
        add_bos_token: Whether ``encode`` should automatically prepend the BOS
            token when the token is known.
        add_eos_token: Whether ``encode`` should automatically append the EOS
            token when the token is known.
    """

    vocab_size: int = 32_000
    special_tokens: Sequence[str] = (
        "<|unk|>",
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
    )
    regex_pattern: Optional[str] = None
    add_bos_token: bool = True
    add_eos_token: bool = True


class NanoChatBPETokenizer:
    """Wrapper around HuggingFace tokenizers.Tokenizer providing convenience helpers."""

    def __init__(self, tokenizer: Tokenizer, config: NanoChatTokenizerConfig):
        """Store tokenizer backend and metadata."""
        self._tokenizer = tokenizer
        self._config = config
        self._bos_id = self._tokenizer.token_to_id("<|bos|>")
        self._eos_id = self._tokenizer.token_to_id("<|eos|>")
        special_tokens = list(config.special_tokens)
        self._special_token_ids = {
            self._tokenizer.token_to_id(token) for token in special_tokens if self._tokenizer.token_to_id(token) is not None
        }

    @property
    def tokenizer(self) -> Tokenizer:
        """Return the underlying tokenizers.Tokenizer instance."""
        return self._tokenizer

    @property
    def config(self) -> NanoChatTokenizerConfig:
        """Return the configuration associated with this tokenizer."""
        return self._config

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls,
        text_iterator: Iterable[str],
        *,
        config: Optional[NanoChatTokenizerConfig] = None,
    ) -> "NanoChatBPETokenizer":
        """Train a new tokenizer from an iterable of text samples."""
        tokenizer_config = config or NanoChatTokenizerConfig()
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=tokenizer_config.vocab_size,
            special_tokens=list(tokenizer_config.special_tokens),
        )
        texts = list(text_iterator)
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer, tokenizer_config)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> List[int]:
        """Convert text into token ids."""
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids
        if not add_special_tokens:
            return list(ids)
        augmented = list(ids)
        if (
            self._config.add_bos_token
            and self._bos_id is not None
            and (not augmented or augmented[0] != self._bos_id)
        ):
            augmented.insert(0, self._bos_id)
        if (
            self._config.add_eos_token
            and self._eos_id is not None
            and (not augmented or augmented[-1] != self._eos_id)
        ):
            augmented.append(self._eos_id)
        return augmented

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """Encode a batch of texts into token id sequences."""
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        if not add_special_tokens:
            return [list(enc.ids) for enc in encodings]
        result: List[List[int]] = []
        for enc in encodings:
            augmented = list(enc.ids)
            if (
                self._config.add_bos_token
                and self._bos_id is not None
                and (not augmented or augmented[0] != self._bos_id)
            ):
                augmented.insert(0, self._bos_id)
            if (
                self._config.add_eos_token
                and self._eos_id is not None
                and (not augmented or augmented[-1] != self._eos_id)
            ):
                augmented.append(self._eos_id)
            result.append(augmented)
        return result

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Convert token ids back into text."""
        if skip_special_tokens:
            filtered = [idx for idx in ids if idx not in self._special_token_ids]
            return self._tokenizer.decode(filtered, skip_special_tokens=False)
        return self._tokenizer.decode(list(ids), skip_special_tokens=False)

    def decode_batch(
        self,
        batch_ids: Sequence[Sequence[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token sequences."""
        decoded: List[str] = []
        for ids in batch_ids:
            decoded.append(self.decode(ids, skip_special_tokens=skip_special_tokens))
        return decoded

    def save(self, output_dir: Path, filename: str = "tokenizer.json") -> Path:
        """Persist tokenizer and configuration to disk."""
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        tokenizer_path = output_dir_path / filename
        self._tokenizer.save(str(tokenizer_path))
        config_path = tokenizer_path.with_suffix(".config.json")
        config_path.write_text(json.dumps(asdict(self._config), indent=2), encoding="utf-8")
        return tokenizer_path

    @classmethod
    def load(cls, tokenizer_path: Path) -> "NanoChatBPETokenizer":
        """Load a previously saved tokenizer."""
        tokenizer_path = Path(tokenizer_path)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config_path = tokenizer_path.with_suffix(".config.json")
        if config_path.exists():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            config = NanoChatTokenizerConfig(**raw)
        else:
            config = NanoChatTokenizerConfig()
        return cls(tokenizer, config)


def iter_file_lines(files: Sequence[Path]) -> Iterator[str]:
    """Yield non-empty lines from a collection of text files."""
    for file_path in files:
        with Path(file_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield stripped


__all__ = ["NanoChatBPETokenizer", "NanoChatTokenizerConfig", "iter_file_lines"]
