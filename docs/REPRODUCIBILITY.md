# Details

## Configuration

| `SoloConfig` field | default | meaning |
|---|---|---|
| `h` | `2` | number of most recent tokens in the keyed context |
| `windows` | `((2, 12), (12, 32))` | anchor windows `[t-hi, t-lo)`, and `()` gives the request-scoped context history over `h`-token contexts |
| `request_scoped` | `True` | single-owner contexts per request, and `False` gives a fixed-key Gumbel-max watermark with the same contexts |

The defaults are the configuration evaluated in the paper. The anchor of a window is its token with the smallest value of the
public bijection `anchor_hash` in `solo/context.py`. A deployment should key this hash and the pseudorandom function with a
secret of at least 128 bits (for example HMAC-SHA256). The 32-bit hashes here match the implementation used for the paper's
experiments.

## Generation

`generate(model, tokenizer, prompts, k, key, cfg)` draws `k` responses for every prompt (one request per prompt) and returns,
for every prompt, `k` dicts with the response token ids (`ids`) and text (`text`). By default it samples at temperature 1 without
top-p or top-k truncation and with `repetition_penalty=1.0`, which overrides the repetition penalty that some checkpoints set in
their `generation_config` (Transformers applies it before any custom logits processor).

To watermark your own decoding loop, use `SoloSampler` directly: `step(probs, active)` chooses the next token of every row from
its next-token probabilities and `commit(tokens, active)` records them. Alternatively, plug `SoloLogitsProcessor` into
`model.generate` with `do_sample=False` and `repetition_penalty=1.0`. Rows that belong to the same request must share a group id.

`examples/generate_and_detect.py` draws 10 SOLO responses for one prompt and compares them with 10 unwatermarked responses under
the same detector (a few minutes on a CPU):

```bash
python examples/generate_and_detect.py --model Qwen/Qwen2.5-0.5B-Instruct --k 10 --device cpu
```

It prints the p-value and the number of scored tokens of every SOLO response, then the p-values of the unwatermarked
responses under the same detector, which are roughly uniform on (0, 1).

## Detection threshold

`detect(ids, key, cfg)` scores the first occurrence of every keyed context of a response and returns the p-value of the nominal
Gamma test, the score and the number of scored tokens. It needs neither the prompt nor the other responses of the request.

For one fixed key, unwatermarked responses that pass through the same answer template evaluate the same keyed contexts, so the
nominal null can be miscalibrated on templated outputs. For decisions at a target false positive rate, calibrate the threshold on
unwatermarked responses of the same model and template, as in the paper:

```python
from solo import calibrated_threshold
thr = calibrated_threshold(null_p_values, fpr=0.01)   # p-values of unwatermarked responses under your key
is_watermarked = detect(ids, key=key).p_value <= thr
```

## Tests

```bash
pip install pytest
pytest tests
```

The tests run on a CPU without downloading a model. They check the pseudorandom function and the detector against reference
values of the implementation used for the paper, the context map, single ownership of keyed contexts within a request (no
pseudorandom value is used twice, whereas a fixed key reuses values across sibling responses), the detection of watermarked
versus unwatermarked token sequences, and that `generate` samples from the model's own next-token distribution even when the
checkpoint's `generation_config` sets a repetition penalty.

Tested with Python 3.12, PyTorch 2.13, Transformers 5.18, NumPy 2.5 and SciPy 1.18.
