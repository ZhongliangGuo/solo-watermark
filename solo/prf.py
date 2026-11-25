"""Pseudorandom function shared by generation and detection.

A keyed context (a key and a tuple of token ids) is hashed into a 32-bit seed; the seed and a token id are hashed again into a
uniform value u in (0, 1). Generation evaluates u for the whole vocabulary at once (``uniforms``); detection evaluates it for one
token (``token_uniform``). Both return identical values.
"""
import torch

M32 = 0xFFFFFFFF


def mix32_int(x: int) -> int:
    """32-bit integer hash (lowbias32 variant) of a Python int."""
    x &= M32
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & M32
    x = ((x ^ (x >> 15)) * 0x846CA68B) & M32
    return x ^ (x >> 16)


def mix32(x: torch.Tensor) -> torch.Tensor:
    """The same hash on int64 tensors holding values in [0, 2^32)."""
    x = x & M32
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & M32
    x = ((x ^ (x >> 15)) * 0x846CA68B) & M32
    return x ^ (x >> 16)


def context_seed(key: int, context) -> int:
    """32-bit seed of a keyed context: a key and a tuple of token ids (-1 stands for 'no token')."""
    s = mix32_int(key * 0x9E3779B1 + 0x1234567)
    s = mix32_int(s ^ mix32_int(0x51ED270B))  # salt slot of the experimental code base, unused (salt = 0)
    for t in context:
        s = mix32_int(s ^ mix32_int((t + 2) * 0x85EBCA77 + 0x2C1B3C6D))
    return s


_VOCAB_CACHE = {}


def uniforms(seeds: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """u[b, v] in (0, 1) for every token v < vocab_size, one row per 32-bit seed (float64, 64 random bits)."""
    dev = seeds.device
    if (vocab_size, dev) not in _VOCAB_CACHE:
        v = torch.arange(vocab_size, device=dev, dtype=torch.int64)
        _VOCAB_CACHE[(vocab_size, dev)] = (mix32(v * 0x9E3779B1 + 0x7F4A7C15), mix32(v * 0xC2B2AE35 + 0x165667B1))
    hv1, hv2 = _VOCAB_CACHE[(vocab_size, dev)]
    s = seeds.to(torch.int64).unsqueeze(1)
    a = mix32(s ^ hv1).to(torch.float64)
    b = mix32(mix32(s + 0x27D4EB2F) ^ hv2).to(torch.float64)
    return (a * 4294967296.0 + b + 0.5) / 18446744073709551616.0


def token_uniform(seed: int, token: int) -> float:
    """u for one token: equal to uniforms(seed)[token]."""
    hv1 = mix32_int(token * 0x9E3779B1 + 0x7F4A7C15)
    hv2 = mix32_int(token * 0xC2B2AE35 + 0x165667B1)
    a = mix32_int(seed ^ hv1)
    b = mix32_int(mix32_int(seed + 0x27D4EB2F) ^ hv2)
    return (a * 4294967296.0 + b + 0.5) / 18446744073709551616.0
