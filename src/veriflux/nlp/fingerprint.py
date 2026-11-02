"""Text fingerprinting algorithms: MinHash and SimHash.

Enables near-duplicate text detection, document deduplication, and O(1) set similarity
estimation without external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re
from typing import List, Sequence, Set, Tuple


@dataclass(slots=True, frozen=True)
class MinHashSignature:
    """Compact signature of minimum hash values."""

    values: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.values)


class MinHasher:
    """MinHash locality-sensitive hashing for estimating Jaccard set similarities."""

    # Mersenne prime (2^61 - 1) for modulo arithmetic
    _PRIME: int = (1 << 61) - 1

    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        rng = random.Random(seed)
        # Generate deterministic linear congruential hash parameters: (a * x + b) % PRIME
        self._a = [rng.randint(1, self._PRIME - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, self._PRIME - 1) for _ in range(num_perm)]

    def _shingle(self, text: str, k: int = 3) -> Set[str]:
        words = re.findall(r"\w+", text.lower())
        if len(words) < k:
            return set(words) or {text.lower()}
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

    def fingerprint(self, text: str, k_shingle: int = 3) -> MinHashSignature:
        """Compute the MinHash signature for text."""
        shingles = self._shingle(text, k=k_shingle)
        if not shingles:
            return MinHashSignature(tuple([0] * self.num_perm))

        sig = [self._PRIME] * self.num_perm

        for shingle in shingles:
            # 64-bit integer hash from MD5
            raw_hash = int(hashlib.md5(shingle.encode("utf-8")).hexdigest()[:16], 16)
            for i in range(self.num_perm):
                h = (self._a[i] * raw_hash + self._b[i]) % self._PRIME
                if h < sig[i]:
                    sig[i] = h

        return MinHashSignature(tuple(sig))

    def similarity(self, sig_a: MinHashSignature, sig_b: MinHashSignature) -> float:
        """Estimate the Jaccard similarity between two MinHash signatures."""
        if len(sig_a) != len(sig_b):
            raise ValueError("MinHash signatures must have the same permutation count")
        if len(sig_a) == 0:
            return 0.0

        matches = sum(1 for x, y in zip(sig_a.values, sig_b.values) if x == y)
        return matches / len(sig_a)


class SimHasher:
    """SimHash algorithm for calculating Hamming distance based document similarity."""

    def __init__(self, hashbits: int = 64) -> None:
        if hashbits not in (32, 64, 128):
            raise ValueError("hashbits must be 32, 64, or 128")
        self.hashbits = hashbits

    def _hash_token(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        if self.hashbits == 32:
            return int(h[:8], 16)
        elif self.hashbits == 64:
            return int(h[:16], 16)
        else:
            return int(h, 16)

    def fingerprint(self, text: str) -> int:
        """Compute integer SimHash representation of text."""
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return 0

        v = [0] * self.hashbits
        for token in tokens:
            token_hash = self._hash_token(token)
            for i in range(self.hashbits):
                bitmask = 1 << i
                if token_hash & bitmask:
                    v[i] += 1
                else:
                    v[i] -= 1

        fingerprint = 0
        for i in range(self.hashbits):
            if v[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    def distance(self, hash_a: int, hash_b: int) -> int:
        """Compute the Hamming distance (number of bit differences) between two fingerprints."""
        x = (hash_a ^ hash_b) & ((1 << self.hashbits) - 1)
        # Brian Kernighan's algorithm for set bit count
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count

    def similarity(self, hash_a: int, hash_b: int) -> float:
        """Return normalised similarity in [0.0, 1.0]."""
        dist = self.distance(hash_a, hash_b)
        return max(0.0, 1.0 - (dist / self.hashbits))

