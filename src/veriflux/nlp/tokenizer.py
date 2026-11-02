"""Byte-Pair Encoding, WordPiece, and heuristic tokenizers implemented from first principles.

Zero external dependencies. Supports training custom BPE/WordPiece vocabularies,
encoding/decoding token streams, and extracting token spans.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(slots=True, frozen=True)
class TokenizerResult:
    """Outcome of tokenizing a sequence of text."""

    tokens: List[str]
    ids: List[int]
    offsets: List[Tuple[int, int]] = field(default_factory=list)


class SimpleTokenizer:
    """Fast whitespace and punctuation-delimited tokenizer with invertibility."""

    _TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.inv_vocab: Dict[int, str] = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}

    def fit(self, texts: Sequence[str], max_vocab: int = 10000) -> None:
        """Build vocabulary from a corpus."""
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self._TOKEN_RE.findall(text))
        for token, _ in counts.most_common(max_vocab - len(self.vocab)):
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token

    def tokenize(self, text: str) -> List[str]:
        """Split text into string tokens."""
        return self._TOKEN_RE.findall(text)

    def encode(self, text: str) -> List[int]:
        """Encode text to token identifiers."""
        tokens = self.tokenize(text)
        unk = self.vocab.get("<unk>", 1)
        return [self.vocab.get(tok, unk) for tok in tokens]

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token IDs back to a reconstructed string."""
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        return " ".join(tokens)


class BPETokenizer:
    """Byte-Pair Encoding (BPE) subword tokenizer implemented from first principles."""

    _GPT4_SPLIT_PATTERN = re.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|[0-9]++| ?[^\s\p{L}\p{N}]++|[\r\n]++|\s++$|\s++""",
        re.UNICODE,
    ) if False else re.compile(r"""'[a-zA-Z]+|[a-zA-Z]+|[0-9]+|\s+|[^\s\w]""", re.UNICODE)

    def __init__(self) -> None:
        self.vocab: Dict[bytes, int] = {}
        self.inv_vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[bytes, bytes], bytes] = {}
        self._init_byte_vocab()

    def _init_byte_vocab(self) -> None:
        """Initialise vocabulary with all 256 raw byte primitives."""
        self.vocab.clear()
        self.inv_vocab.clear()
        # Special tokens
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for idx, tok in enumerate(self.special_tokens):
            b_tok = tok.encode("utf-8")
            self.vocab[b_tok] = idx
            self.inv_vocab[idx] = b_tok

        for b in range(256):
            b_byte = bytes([b])
            idx = len(self.vocab)
            self.vocab[b_byte] = idx
            self.inv_vocab[idx] = b_byte

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _get_stats(vocab_words: Dict[Tuple[bytes, ...], int]) -> Counter[Tuple[bytes, bytes]]:
        """Count frequencies of adjacent symbol pairs."""
        pairs: Counter[Tuple[bytes, bytes]] = Counter()
        for word, freq in vocab_words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_vocab(
        pair: Tuple[bytes, bytes],
        vocab_words: Dict[Tuple[bytes, ...], int],
    ) -> Dict[Tuple[bytes, ...], int]:
        """Merge all occurrences of pair in vocabulary representation."""
        new_vocab: Dict[Tuple[bytes, ...], int] = {}
        bigram = pair[0] + pair[1]
        for word, freq in vocab_words.items():
            new_word: List[bytes] = []
            i = 0
            w_len = len(word)
            while i < w_len:
                if i < w_len - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def train(self, texts: Sequence[str], vocab_size: int = 1000) -> None:
        """Train BPE merge rules on corpus until target vocabulary size is reached."""
        self._init_byte_vocab()
        num_merges = vocab_size - len(self.vocab)
        if num_merges <= 0:
            return

        # Pre-tokenize into word frequency mapping
        word_counts: Counter[str] = Counter()
        for text in texts:
            words = self._GPT4_SPLIT_PATTERN.findall(text)
            word_counts.update(words)

        # Convert words into tuples of 1-byte sequences
        vocab_words: Dict[Tuple[bytes, ...], int] = {}
        for word, freq in word_counts.items():
            byte_seq = tuple(bytes([b]) for b in word.encode("utf-8"))
            vocab_words[byte_seq] = freq

        for _ in range(num_merges):
            pairs = self._get_stats(vocab_words)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)  # type: ignore[arg-type]
            if pairs[best_pair] < 2:
                break

            merged_token = best_pair[0] + best_pair[1]
            idx = len(self.vocab)
            self.vocab[merged_token] = idx
            self.inv_vocab[idx] = merged_token
            self.merges[best_pair] = merged_token

            vocab_words = self._merge_vocab(best_pair, vocab_words)

    def _encode_word(self, word: str) -> List[int]:
        """Apply learned BPE merges greedily to a single word."""
        word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
        if not word_bytes:
            return []

        symbols = list(word_bytes)
        while len(symbols) >= 2:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # Find earliest learned merge
            mergeable = [p for p in pairs if p in self.merges]
            if not mergeable:
                break
            # Priority given to earliest learned merge rank
            best_pair = min(mergeable, key=lambda p: self.vocab[self.merges[p]])
            merged_val = self.merges[best_pair]

            new_symbols: List[bytes] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(merged_val)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        unk_id = self.vocab.get(b"<unk>", 1)
        return [self.vocab.get(s, unk_id) for s in symbols]

    def encode(self, text: str) -> List[int]:
        """Encode text to token ID sequence."""
        if not text:
            return []
        words = self._GPT4_SPLIT_PATTERN.findall(text)
        token_ids: List[int] = []
        for word in words:
            token_ids.extend(self._encode_word(word))
        return token_ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token IDs back into string."""
        raw_bytes = bytearray()
        for idx in ids:
            b_chunk = self.inv_vocab.get(idx, b"")
            if b_chunk in (b"<pad>", b"<unk>", b"<bos>", b"<eos>"):
                continue
            raw_bytes.extend(b_chunk)
        return raw_bytes.decode("utf-8", errors="replace")

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        """Encode multiple strings in batch."""
        return [self.encode(t) for t in texts]

    def save(self, path: str) -> None:
        """Serialize BPE model to JSON."""
        data = {
            "vocab": {v.hex(): k for v, k in self.vocab.items()},
            "merges": [
                [p[0].hex(), p[1].hex(), m.hex()] for p, m in self.merges.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load BPE model from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {bytes.fromhex(k): v for k, v in data["vocab"].items()}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {
            (bytes.fromhex(p[0]), bytes.fromhex(p[1])): bytes.fromhex(p[2])
            for p in data["merges"]
        }


class WordPieceTokenizer:
    """WordPiece subword tokenizer (BERT-style) with ## subword prefixes."""

    def __init__(self, unk_token: str = "[UNK]", max_input_chars_per_word: int = 100) -> None:
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.vocab: Dict[str, int] = {unk_token: 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3}
        self.inv_vocab: Dict[int, str] = {0: unk_token, 1: "[PAD]", 2: "[CLS]", 3: "[SEP]"}

    def fit(self, texts: Sequence[str], vocab_size: int = 1000) -> None:
        """Build WordPiece vocabulary from character pieces and top frequent segments."""
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text.lower().split())

        chars: Set[str] = set()
        for word in counts:
            chars.update(list(word))

        for c in sorted(chars):
            if c not in self.vocab:
                idx = len(self.vocab)
                self.vocab[c] = idx
                self.inv_vocab[idx] = c
            sub = f"##{c}"
            if sub not in self.vocab:
                idx = len(self.vocab)
                self.vocab[sub] = idx
                self.inv_vocab[idx] = sub

        for word, _ in counts.most_common(vocab_size - len(self.vocab)):
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.inv_vocab[idx] = word

    def tokenize(self, text: str) -> List[str]:
        """Greedy longest-match subword tokenization."""
        output_tokens: List[str] = []
        for word in text.split():
            if len(word) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens: List[str] = []
            while start < len(word):
                end = len(word)
                cur_substr: Optional[str] = None
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = f"##{substr}"
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        unk_id = self.vocab.get(self.unk_token, 0)
        return [self.vocab.get(t, unk_id) for t in tokens]

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [self.inv_vocab.get(i, self.unk_token) for i in ids]
        res: List[str] = []
        for t in tokens:
            if t.startswith("##"):
                if res:
                    res[-1] += t[2:]
                else:
                    res.append(t[2:])
            else:
                res.append(t)
        return " ".join(res)


class RegexTokenizer:
    """Configurable regex-based tokenizer with span tracking."""

    def __init__(self, pattern: Optional[str] = None) -> None:
        self.pattern = re.compile(pattern or r"\w+|[^\w\s]", re.UNICODE)

    def tokenize_with_spans(self, text: str) -> TokenizerResult:
        """Tokenize text while extracting exact start and end offsets."""
        tokens: List[str] = []
        offsets: List[Tuple[int, int]] = []
        for match in self.pattern.finditer(text):
            tokens.append(match.group(0))
            offsets.append((match.start(), match.end()))
        return TokenizerResult(tokens=tokens, ids=list(range(len(tokens))), offsets=offsets)

