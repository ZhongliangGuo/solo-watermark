"""SOLO generation: Gumbel-max sampling with single-owner contexts.

All responses of a request are decoded together. A response that reaches a keyed context which no response of the request has used
yet claims it and decodes with the Gumbel-max rule argmax_v log p_v - log(-log u_v), where u is the PRF value of (key, context, v);
a response that reaches a context already claimed (by itself or by a sibling) samples that token with true randomness. No
pseudorandom value is therefore used twice inside a request, and the responses of a request are jointly distributed as independent
unwatermarked responses (under an ideal PRF), while every response can be detected with the single provider key.
"""
import math

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .context import SoloConfig, context
from .prf import context_seed, uniforms


class SoloSampler:
    """Chooses the next token of a batch of responses from their next-token probabilities.

    groups[b] is the request of row b; rows with the same group share one set of claimed contexts.
    """

    def __init__(self, key: int, groups, cfg: SoloConfig = SoloConfig(), generator: torch.Generator = None):
        self.key, self.cfg, self.gen = key, cfg, generator
        self.groups = list(groups)
        self.tokens = [[] for _ in self.groups]          # response tokens of every row
        self.seen = [set() for _ in self.groups]         # contexts already scored in this response
        self.claimed = {}                                # request -> contexts claimed by any of its responses
        self._cur = [None] * len(self.groups)            # (position, seed) of the current step

    def step(self, probs: torch.Tensor, active) -> torch.Tensor:
        """probs: [B, V] next-token probabilities (temperature applied). active: bool per row. Returns token ids [B]."""
        chosen = torch.multinomial(probs.float().clamp_min(0), 1, generator=self.gen).squeeze(1)  # true randomness
        rows, seeds = [], []
        for b in range(probs.shape[0]):
            if not active[b]:
                continue
            t = len(self.tokens[b])
            sd = context_seed(self.key, context(self.tokens[b], t, self.cfg))
            self._cur[b] = (t, sd)
            if sd in self.seen[b]:                       # repeated context within the response: not watermarked
                continue
            if self.cfg.request_scoped:
                claimed = self.claimed.setdefault(self.groups[b], set())
                if sd in claimed:                        # owned by a sibling (or by this response): true randomness
                    continue
                claimed.add(sd)                          # claim now, so simultaneous claims in this step lose
            rows.append(b); seeds.append(sd)
        for c0 in range(0, len(rows), 96):              # row chunks bound the memory of the [rows, V] PRF matrix
            r = torch.tensor(rows[c0:c0 + 96], device=probs.device)
            u = uniforms(torch.tensor(seeds[c0:c0 + 96], device=probs.device, dtype=torch.int64), probs.shape[1])
            p = probs[r].double()
            g = torch.where(p > 0, torch.log(p) - torch.log(-torch.log(u)), torch.full_like(p, -math.inf))
            chosen[r] = g.argmax(dim=1)
        return chosen

    def commit(self, chosen: torch.Tensor, active):
        """Append the chosen tokens and mark the current contexts as seen by their responses."""
        for b, tok in enumerate(chosen.tolist()):
            if not active[b]:
                continue
            t = len(self.tokens[b])
            sd = self._cur[b][1] if self._cur[b] is not None and self._cur[b][0] == t else \
                context_seed(self.key, context(self.tokens[b], t, self.cfg))
            self.seen[b].add(sd)
            self.tokens[b].append(tok)


class SoloLogitsProcessor(LogitsProcessor):
    """Hugging Face logits processor: selects every token with SoloSampler and makes greedy decoding emit it.

    Use with model.generate(..., do_sample=False, logits_processor=LogitsProcessorList([proc]), repetition_penalty=1.0).
    Transformers applies processors from the model's generation_config (for example its repetition penalty) before this one, so
    override them to sample from the model distribution itself.
    """

    def __init__(self, sampler: SoloSampler, eos_token_ids, temperature: float = 1.0):
        self.s, self.T = sampler, temperature
        self.eos = set(eos_token_ids)
        self.active = np.ones(len(sampler.groups), bool)

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores.float() / self.T, dim=-1)
        chosen = self.s.step(probs, self.active)
        self.s.commit(chosen, self.active)
        for b, t in enumerate(chosen.tolist()):
            if self.active[b] and t in self.eos:
                self.active[b] = False
        out = torch.full_like(scores, -math.inf)
        out[torch.arange(scores.shape[0]), chosen] = 0.0
        return out


def eos_token_ids(model, tokenizer):
    ids = {tokenizer.eos_token_id}
    ge = getattr(model.generation_config, "eos_token_id", None)
    ids |= set(ge if isinstance(ge, (list, tuple)) else [ge])
    for name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        i = tokenizer.convert_tokens_to_ids(name)
        if isinstance(i, int) and i != tokenizer.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


@torch.no_grad()
def generate(model, tokenizer, prompts, k: int, key: int, cfg: SoloConfig = SoloConfig(), max_new_tokens: int = 256,
             temperature: float = 1.0, seed: int = 0, chat: bool = True, repetition_penalty: float = 1.0):
    """Draw k watermarked responses for every prompt (one request per prompt).

    The default repetition_penalty = 1.0 overrides the penalty of the model's generation_config, so that the responses are drawn
    from the model distribution at the given temperature without truncation.

    Returns a list (one entry per prompt) of k dicts with the response token ids ('ids') and text ('text').
    """
    dev = next(model.parameters()).device
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             if chat else p for p in prompts]
    rows = [t for t in texts for _ in range(k)]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(rows, return_tensors="pt", padding=True, add_special_tokens=not chat).to(dev)
    eos = eos_token_ids(model, tokenizer)
    gen = torch.Generator(device=dev).manual_seed(seed)
    sampler = SoloSampler(key, groups=[i // k for i in range(len(rows))], cfg=cfg, generator=gen)
    proc = SoloLogitsProcessor(sampler, eos, temperature)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, logits_processor=LogitsProcessorList([proc]),
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=eos, temperature=None, top_p=None, top_k=None,
                         repetition_penalty=repetition_penalty)
    new = out[:, enc["input_ids"].shape[1]:].tolist()
    res = []
    for i in range(len(prompts)):
        group = []
        for ids in new[i * k:(i + 1) * k]:
            for j, t in enumerate(ids):
                if t in eos:
                    ids = ids[:j]; break
            group.append(dict(ids=ids, text=tokenizer.decode(ids, skip_special_tokens=True)))
        res.append(group)
    return res
