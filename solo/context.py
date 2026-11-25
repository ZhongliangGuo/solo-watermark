"""Content-anchored context map of SOLO.

The keyed context of position t of a response is its last ``h`` tokens and, for every anchor window [t - hi, t - lo), the token of
the window with the smallest anchor hash (a min-hash). The anchors summarize the preceding tokens, so that sibling responses of a
request rarely share a context, and an insertion or deletion only moves the anchors locally.
"""
from dataclasses import dataclass

from .prf import M32


@dataclass(frozen=True)
class SoloConfig:
    """Parameters of SOLO. The defaults are the configuration used in the paper.

    h: number of most recent tokens in the context.
    windows: anchor windows as (lo, hi) pairs, i.e. [t - hi, t - lo); () gives plain h-token contexts (the request-scoped context
        history of Hu et al.).
    request_scoped: claim each keyed context at most once per request (single-owner contexts). False gives an ordinary
        fixed-key Gumbel-max watermark with the same context map, which couples the responses of a request.
    """
    h: int = 2
    windows: tuple = ((2, 12), (12, 32))
    request_scoped: bool = True


def anchor_hash(token: int) -> int:
    """Anchor (min-hash) order of token ids: an affine bijection of the 32-bit token id."""
    return (token * 0x9E3779B1 + 0x7F4A7C15) & M32


def context(tokens, t: int, cfg: SoloConfig) -> tuple:
    """Keyed context of position t, given the response tokens tokens[:t] (identical at generation and at detection)."""
    local = tuple(([-1] * cfg.h + list(tokens[max(0, t - cfg.h):t]))[-cfg.h:]) if cfg.h > 0 else ()
    anchors = []
    for lo, hi in cfg.windows:
        a, b = max(0, t - hi), t - lo
        anchors.append(min(tokens[a:b], key=anchor_hash) if b > a else -1)
    return local + tuple(anchors)
