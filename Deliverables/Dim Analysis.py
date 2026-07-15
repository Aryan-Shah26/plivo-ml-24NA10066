"""
Run once at minute ~10. Answers the hint: 'the 256 dims may not all do the
same kind of work'. Ranks dims by cross-voice variance -> high-variance dims
likely encode identity/timbre; near-constant dims are probably structural
(shared across all voices) and safe to leave near the stock-voice mean
during search, shrinking effective search dimensionality.
"""
import torch
from adapter import load_stock_voices


def analyze(save_path: str = "dim_stats.pt") -> dict:
    voices = load_stock_voices()
    stack = torch.stack(list(voices.values()))  # [54, 256] (or [54, T, 256] — flatten if so)
    if stack.dim() > 2:
        stack = stack.flatten(1)

    mean = stack.mean(0)
    std = stack.std(0)
    var_rank = torch.argsort(std, descending=True)

    stats = {"mean": mean, "std": std, "var_rank": var_rank}
    torch.save(stats, save_path)

    print(f"dims: {stack.shape[1]}")
    print(f"top-10 highest-variance dims: {var_rank[:10].tolist()}")
    print(f"top-10 lowest-variance dims:  {var_rank[-10:].tolist()}")
    print(f"std range: {std.min().item():.4f} - {std.max().item():.4f}")
    return stats


if __name__ == "__main__":
    analyze()