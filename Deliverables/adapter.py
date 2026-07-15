"""
It's the only place that assumes function
signatures from starter/synth.py, similarity.py, blend.py. Everything else
(fitness.py, search.py, dim_analysis.py) imports from here and never touches
those modules directly, so a signature mismatch only costs you one file.

Fill in the three functions below to match the real starter code.
"""
import numpy as np
import torch

# ---- adjust these imports to match the handout ----
# import sys; sys.path.insert(0, "starter")
# from synth import synthesize as _synthesize
# from similarity import speaker_similarity as _speaker_similarity, load_target as _load_target
# from blend import STOCK_VOICE_DIR


def synthesize(text: str, style_tensor: torch.Tensor) -> np.ndarray:
    """text + style tensor -> mono float32 waveform (matches synth.py wrapper)."""
    # return _synthesize(text, style_tensor)
    raise NotImplementedError("wire up to starter/synth.py")


def speaker_similarity(wav: np.ndarray, target_embed) -> float:
    """cosine similarity between synthesized wav and target speaker embedding."""
    # return _speaker_similarity(wav, target_embed)
    raise NotImplementedError("wire up to starter/similarity.py")


def load_target_embedding(reference_dir: str):
    """Build/load the target speaker's resemblyzer embedding from wavs in reference_dir."""
    # e.g. from resemblyzer import VoiceEncoder, preprocess_wav
    # import glob
    # wavs = [preprocess_wav(p) for p in glob.glob(f"{reference_dir}/*.wav")]
    # return VoiceEncoder().embed_speaker(wavs)
    raise NotImplementedError("wire up to reference/ wavs, likely via similarity.py")


def load_stock_voices() -> dict[str, torch.Tensor]:
    """name -> style tensor, from kokoro_assets/."""
    raise NotImplementedError("wire up to kokoro_assets/ loader (see blend.py)")


EVAL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please call me back when you get a chance.",
    "The weather today is unusually warm for October.",
]