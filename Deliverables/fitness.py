"""
Fitness = mean speaker similarity over several sentences, penalized for
(a) degenerate/raspy audio that fools the embedding (intelligibility proxy)
(b) drifting off the stock-voice manifold (OOD proxy).
Caches by tensor hash since ES will re-evaluate elites.
"""
import hashlib
import numpy as np
import torch

from adapter import synthesize, speaker_similarity, EVAL_SENTENCES

_CACHE: dict[str, float] = {}


def _tensor_key(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().cpu().numpy().tobytes()).hexdigest()


def _intelligibility_proxy(wav: np.ndarray) -> float:
    """Cheap no-ASR proxy: penalize near-silence, clipping, or abnormally low
    energy variance (a common failure mode of embedding-only optimization).
    Returns a penalty in [0, 1], higher = worse."""
    if wav.size == 0 or not np.isfinite(wav).all():
        return 1.0
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-4:
        return 1.0
    clip_frac = np.mean(np.abs(wav) > 0.99)
    energy = wav ** 2
    frame = max(1, len(wav) // 50)
    frame_energy = np.array([energy[i:i + frame].mean() for i in range(0, len(energy), frame)])
    cv = frame_energy.std() / (frame_energy.mean() + 1e-8)  # low cv -> flat/raspy monotone
    flatness_penalty = max(0.0, 0.3 - cv) / 0.3  # penalize if too flat
    return float(np.clip(clip_frac * 2 + flatness_penalty * 0.5, 0, 1))


def _ood_penalty(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> float:
    """Normalized distance from the stock-voice distribution, per-dim."""
    z = (tensor.flatten() - mean) / (std + 1e-6)
    return float(z.pow(2).mean().sqrt())  # ~avg z-score magnitude


def evaluate(
    tensor: torch.Tensor,
    target_embed,
    stock_mean: torch.Tensor,
    stock_std: torch.Tensor,
    lam: float = 0.4,   # intelligibility weight
    mu: float = 0.05,   # OOD weight
    n_sentences: int = 2,
) -> float:
    key = _tensor_key(tensor)
    if key in _CACHE:
        return _CACHE[key]

    sims, penalties = [], []
    for text in EVAL_SENTENCES[:n_sentences]:
        wav = synthesize(text, tensor)
        sims.append(speaker_similarity(wav, target_embed))
        penalties.append(_intelligibility_proxy(wav))

    sim = float(np.mean(sims))
    intel_pen = float(np.mean(penalties))
    ood_pen = _ood_penalty(tensor, stock_mean, stock_std)

    score = sim - lam * intel_pen - mu * ood_pen
    _CACHE[key] = score
    return score


def cache_size() -> int:
    return len(_CACHE)