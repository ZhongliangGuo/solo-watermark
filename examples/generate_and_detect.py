"""Draw K watermarked responses for a prompt with SOLO, detect each one, and compare with unwatermarked sampling.

usage: python examples/generate_and_detect.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--k 10] [--key 12345] [--device cpu]
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without installing
from solo import SoloConfig, detect, generate
from solo.sampler import eos_token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="Write a short blog post with three tips for staying productive when working from home.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--key", type=int, default=12345)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    cfg = SoloConfig()  # h = 2, anchor windows [t-12, t-2) and [t-32, t-12), single-owner contexts per request

    responses = generate(model, tok, [args.prompt], k=args.k, key=args.key, cfg=cfg, max_new_tokens=args.max_new_tokens)[0]
    print(f"== {args.k} SOLO responses (key {args.key})")
    for i, r in enumerate(responses):
        d = detect(r["ids"], key=args.key, cfg=cfg)
        print(f"[{i}] p = {d.p_value:.2e} ({d.n_scored} scored tokens) | {r['text'][:70]!r}")

    # Unwatermarked reference: plain sampling, scored with the same detector (p-values are roughly uniform)
    enc = tok([tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)] * args.k,
              return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                         max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    print("== unwatermarked responses, same detector")
    eos = set(eos_token_ids(model, tok))
    for i, ids in enumerate(out[:, enc["input_ids"].shape[1]:].tolist()):
        ids = ids[:next((j for j, t in enumerate(ids) if t in eos), len(ids))]  # cut at the end of the turn
        print(f"[{i}] p = {detect(ids, key=args.key, cfg=cfg).p_value:.2e}")
    print("Note: for a decision at a target false positive rate, calibrate the threshold on unwatermarked responses of the same "
          "model and prompt template (solo.calibrated_threshold).")


if __name__ == "__main__":
    main()
