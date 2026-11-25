"""Unit tests (CPU, no model download): reference values of the paper's implementation, single ownership, and detection."""
import random

import numpy as np
import torch

from solo import SoloConfig, SoloSampler, detect
from solo.context import context
from solo.prf import context_seed, token_uniform, uniforms


def test_prf_reference_values():
    # values of the implementation used for the paper's experiments
    sd = context_seed(12345, (17, 4, 901, 33))
    assert sd == 4006655709
    assert token_uniform(sd, 1000) == 0.21159736223554293
    assert uniforms(torch.tensor([sd]), 1001)[0, 1000].item() == token_uniform(sd, 1000)


def test_detection_reference_value():
    rng = random.Random(2026)
    seq = [rng.randrange(151643) for _ in range(200)]
    d = detect(seq, key=12345)
    assert d.n_scored == 200
    assert abs(d.p_value - 0.33379396445152804) < 1e-12


def test_context_map():
    cfg = SoloConfig()
    toks = list(range(100, 140))
    assert context(toks, 0, cfg) == (-1, -1, -1, -1)            # no history yet
    c = context(toks, 35, cfg)
    assert c[:2] == (133, 134)                                   # last h = 2 tokens
    assert c[2] in toks[23:33] and c[3] in toks[3:23]            # anchors from [t-12, t-2) and [t-32, t-12)


def _toy_run(cfg, key=777, k=8, steps=150, vocab=300, seed=0):
    """k responses of one request decoded from random next-token distributions."""
    gen = torch.Generator().manual_seed(seed)
    pg = torch.Generator().manual_seed(seed + 1)
    s = SoloSampler(key, groups=[0] * k, cfg=cfg, generator=gen)
    active = np.ones(k, bool)
    for _ in range(steps):
        p = torch.softmax(torch.randn(k, vocab, generator=pg) * 1.5, dim=-1)
        s.commit(s.step(p, active), active)
    return s


def test_no_pseudorandom_value_is_used_twice_in_a_request(monkeypatch):
    import solo.sampler as sm
    used = []
    real = sm.uniforms
    monkeypatch.setattr(sm, "uniforms", lambda seeds, V: (used.extend(seeds.tolist()), real(seeds, V))[1])
    s = _toy_run(SoloConfig(), k=10, vocab=40)                   # small vocabulary: siblings often reach the same context
    assert len(used) == len(set(used))                            # every keyed context decoded with the PRF at most once
    assert set(used) == s.claimed[0]
    n_pos = sum(len(t) for t in s.tokens)
    assert len(used) < n_pos                                      # some positions fell back to true randomness

    used.clear()
    s = _toy_run(SoloConfig(request_scoped=False), k=10, vocab=40)
    assert len(used) > len(set(used))                             # a fixed-key watermark reuses values across siblings


def test_watermarked_responses_are_detected_and_others_are_not():
    s = _toy_run(SoloConfig(), steps=300)
    p_wm = [detect(t, key=s.key).p_value for t in s.tokens]
    assert max(p_wm) < 1e-3
    rng = random.Random(1)
    p_null = [detect([rng.randrange(300) for _ in range(300)], key=s.key).p_value for _ in range(200)]
    assert 0.35 < float(np.mean(p_null)) < 0.65                  # roughly uniform under no watermark
    assert detect(s.tokens[0], key=s.key + 1).p_value > 1e-3     # wrong key: no detection


def test_generate_samples_from_the_model_distribution():
    # a checkpoint's generation_config (here a repetition penalty) must not reshape the distribution SOLO samples from
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    import solo.sampler as sm

    words = ["<pad>", "<eos>", "<unk>"] + [f"w{i}" for i in range(61)]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, eos_token_id=1, pad_token_id=0)).eval()
    model.generation_config.repetition_penalty = 1.3
    seen = []
    call = sm.SoloLogitsProcessor.__call__
    def spy(self, input_ids, scores):
        with torch.no_grad():
            raw = model(input_ids).logits[:, -1, :]
        seen.append(float((scores.float() - raw.float()).abs().max()))
        return call(self, input_ids, scores)
    sm.SoloLogitsProcessor.__call__ = spy
    try:
        out = sm.generate(model, tok, ["w1 w2 w3"], k=3, key=5, max_new_tokens=12, chat=False)[0]
    finally:
        sm.SoloLogitsProcessor.__call__ = call
    assert len(out) == 3 and len(seen) > 1
    assert max(seen) < 1e-4
