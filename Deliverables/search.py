"""Search skeleton: random walk over voice-tensor space. Runs as-is, but
the fitness function and search strategy are deliberately naive — that is
your hour.

    python search.py --reference_dir ../reference --start blend_baseline.pt \
        --iters 150 --out voice.pt
"""
import argparse

import numpy as np
import torch

from tts_handout.starter import synth
from tts_handout.starter import similarity as sim

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your order number after the beep.",
    "I will call you back tomorrow at three thirty.",
]


def _naturalness_penalty(wav):
    """Cheap no-ASR guard against the embedding-loves-it/ears-hate-it
    failure mode: penalize near-silence, clipping, or flat/monotone energy."""
    if wav.size == 0 or not np.isfinite(wav).all():
        return 1.0
    rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2))
    if rms < 1e-4:
        return 1.0
    clip_frac = np.mean(np.abs(wav) > 0.99)
    frame = max(1, len(wav) // 50)
    energy = wav.astype(np.float64) ** 2
    frame_energy = np.array([energy[i:i + frame].mean()
                             for i in range(0, len(energy), frame)])
    cv = frame_energy.std() / (frame_energy.mean() + 1e-8)
    flatness_pen = max(0.0, 0.25 - cv) / 0.25
    return float(np.clip(clip_frac * 2 + flatness_pen * 0.5, 0, 1))


def rows_used(text, voice):
    """Which of the 510 rows does THIS text's phoneme count select?
    Kokoro indexes the voice tensor by phoneme length per chunk, so a
    fitness call on one text only ever exercises a handful of rows."""
    pipe = synth.get_pipeline()
    v = synth.load_voice(voice)
    idxs = set()
    for r in pipe(text, voice=v):
        n = len(r.phonemes) if hasattr(r, "phonemes") and r.phonemes else None
        if n:
            idxs.add(min(n, v.shape[0] - 1))
    return idxs


def fitness(voice, target_emb, texts, lam=0.4):
    """Mean similarity across several DIFFERENT sentences, minus a
    naturalness penalty so the search can't win by finding artifacts the
    embedding likes and ears don't."""
    sims, pens = [], []
    for t in texts:
        wav = synth.synthesize(t, voice)
        sims.append(sim.similarity_to_target(wav, target_emb))
        pens.append(_naturalness_penalty(wav))
    return float(np.mean(sims)) - lam * float(np.mean(pens))


def perturb(voice, step, active_rows=None):
    """Structured perturbation: only touch rows that ANY eval sentence
    actually indexes (else you're randomizing 500+ rows fitness never
    sees, and gains evaporate on held-out sentences). Falls back to
    perturbing everything if active_rows is unknown."""
    cand = voice.clone()
    if active_rows:
        idx = torch.tensor(sorted(active_rows))
        cand[idx] += step * torch.randn(len(idx), voice.shape[1])
    else:
        cand += step * torch.randn_like(voice)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--start", required=True, help="starting .pt tensor")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--step", type=float, default=0.03)
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--listen_every", type=int, default=5)
    ap.add_argument("--stall_patience", type=int, default=20,
                    help="iters with no improvement before step-size restart")
    args = ap.parse_args()

    target = sim.target_embedding(args.reference_dir)
    best = synth.load_voice(args.start).clone()

    # Which rows do our eval sentences actually touch? Restrict mutation
    # to those (+ a small neighborhood) so gains don't evaporate on
    # held-out grading sentences that hit nearby rows.
    active_rows = set()
    for t in SENTENCES:
        active_rows |= rows_used(t, best)
    for r in list(active_rows):
        active_rows.update({max(0, r - 1), min(best.shape[0] - 1, r + 1)})
    print(f"active rows from eval sentences: {sorted(active_rows)}")

    best_f = fitness(best, target, SENTENCES)   # all 3 sentences, not 1
    print(f"start fitness: {best_f:.4f}")

    step = args.step
    accepted = 0
    stall = 0
    for i in range(1, args.iters + 1):
        cand = perturb(best, step, active_rows)
        f = fitness(cand, target, SENTENCES)

        # accept on improvement, or sideways (within noise) to escape
        # local optima; anneal step size on stagnation, restart it on a
        # long stall.
        if f > best_f - 1e-4:
            improved = f > best_f
            best, best_f = cand, max(f, best_f)
            accepted += 1
            stall = 0
            if improved:
                step = min(step * 1.05, args.step * 2)
            print(f"iter {i:4d}  accepted #{accepted}  fitness {best_f:.4f}  step {step:.4f}")
            if accepted % args.listen_every == 0:
                import soundfile as sf
                sf.write(f"listen_{accepted}.wav",
                         synth.synthesize(SENTENCES[0], best), synth.SR)
                print(f"  -> wrote listen_{accepted}.wav — GO LISTEN")
        else:
            stall += 1
            step *= 0.97
            if stall >= args.stall_patience:
                step = args.step * 0.5   # restart with smaller step, not full reset
                stall = 0
                print(f"iter {i:4d}  stalled {args.stall_patience} — step restart to {step:.4f}")

    torch.save(best, args.out)
    import soundfile as sf
    sf.write("listen_final.wav", synth.synthesize(SENTENCES[0], best), synth.SR)
    print(f"final fitness {best_f:.4f} -> saved {args.out}")
    print("wrote listen_final.wav — LISTEN BEFORE YOU SUBMIT")


if __name__ == "__main__":
    main()