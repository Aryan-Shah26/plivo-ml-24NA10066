# Execution Log

## Run 1: Baseline Establishment
* **Script:** `blend.py`
* **Settings:** Default reference comparison against all 54 stock Kokoro voices.
* **Score:** 0.6242 (Best single voice: `af_nova`). Naive 50/50 blending reduced score to 0.5983.
* **Audio Feedback:** The baseline voice sounded coherent and intelligible but lacked the specific target speaker's distinct acoustic identity.
* **Changes Made:** Selected `blend_baseline.pt` as the initialization point for the randomized search rather than starting from zero, providing a strong initial manifold.

## Run 2: Structured Perturbation Search
* **Script:** `search.py`
* **Settings:** 150 iterations, step size 0.03, stall patience 20. Naturalness penalty lambda = 0.4.
* **Score:** 0.7400 (Final).
* **Audio Feedback:** Iteration checkpoints (`listen_5.wav`, `listen_10.wav`) showed progressive identity matching. The final output (`listen_final.wav`) is crystal clear and retains a human feel. It is not robotic and comfortably passes the Intelligibility Gate.
* **Changes Made:** Search concluded successfully.