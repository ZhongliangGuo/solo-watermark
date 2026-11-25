# SOLO: Sample-Faithful and Detectable LLM Watermarking under Repeated Sampling

Reference implementation of SOLO, a generation-time watermark that stays faithful when an application draws several responses per prompt.

Self-consistency, best-of-n selection and sampling-based uncertainty estimation draw several responses for one prompt and assume that they are independent. SOLO decodes the responses of one request with single-owner keyed contexts: a response that reaches a context already claimed in the request samples that token with true randomness, so no pseudorandom value is reused and the responses are jointly distributed as independent unwatermarked responses (under an ideal pseudorandom function). Every response is detected from its text alone with one key. The keyed context of a position is its last two tokens plus two anchors chosen by min-hash from the windows `[t-12, t-2)` and `[t-32, t-12)`, so that sibling responses rarely compete for a context and an edit moves the anchors only locally.

## Install

```bash
pip install -r requirements.txt   # torch, transformers, numpy, scipy
pip install -e .                   # installs the package `solo`
```

## Run

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from solo import SoloConfig, generate, detect

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok, model = AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name).eval()
responses = generate(model, tok, ["What is the capital of Australia?"], k=10, key=12345, cfg=SoloConfig())[0]
print([detect(r["ids"], key=12345).p_value for r in responses])   # response token ids only, without the prompt
```

```bash
python examples/generate_and_detect.py --model Qwen/Qwen2.5-0.5B-Instruct --k 10 --device cpu
pytest tests                       # CPU only, no model download
```

[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) describes the configuration, the detection threshold, the sampler interface and the tests.

## Layout

- `solo/prf.py`: pseudorandom function shared by generation and detection
- `solo/context.py`: content-anchored context map and `SoloConfig`
- `solo/sampler.py`: generation with single-owner contexts (`SoloSampler`, `SoloLogitsProcessor`, `generate`)
- `solo/detect.py`: single-key Gamma test and threshold calibration
- `examples/`: SOLO responses versus unwatermarked responses for one prompt
- `tests/`: unit tests

## License

MIT, see `LICENSE`.
