"""SOLO detection: a single-key Gumbel score test on the text alone.

The detector recomputes the keyed context of every position from the response tokens, skips repeated contexts (as the generator
does), and sums s_t = -log(1 - u_t) over the scored positions. Without the watermark the sum of n scores follows Gamma(n, 1);
positions that a sibling owned at generation time behave like unwatermarked ones and only add null-distributed noise. Neither the
prompt nor the other responses of the request are needed.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaincc

from .context import SoloConfig, context
from .prf import context_seed, token_uniform


@dataclass
class Detection:
    p_value: float   # one-sided p-value of the Gamma test (small = watermarked)
    score: float     # sum of -log(1 - u) over the scored positions
    n_scored: int    # number of scored (first-occurrence) positions


def detect(tokens, key: int, cfg: SoloConfig = SoloConfig()) -> Detection:
    """Test one response (a list of token ids, without the prompt) for the SOLO watermark of `key`."""
    seen, score, n = set(), 0.0, 0
    for t in range(len(tokens)):
        sd = context_seed(key, context(tokens, t, cfg))
        if sd in seen:
            continue
        seen.add(sd)
        score += -math.log1p(-token_uniform(sd, tokens[t]))
        n += 1
    return Detection(p_value=float(gammaincc(n, score)) if n > 0 else 1.0, score=score, n_scored=n)


def calibrated_threshold(null_p_values, fpr: float = 0.01) -> float:
    """Threshold on p-values for a target false positive rate, estimated on unwatermarked responses.

    The Gamma null is exact on average over keys, but for one fixed key, unwatermarked responses that pass through the same
    template evaluate the same keyed contexts and share a key-dependent shift. As in the paper, calibrate on unwatermarked responses
    of the same model and prompt template and flag a response when its p-value is <= the returned threshold.
    """
    return float(np.quantile(np.asarray(null_p_values, dtype=float), fpr))
