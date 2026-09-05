#!/usr/bin/env python3
"""Generate a deterministic synthetic corpus for golden/regression tests.

Needs >= 1000 surviving words. Regenerating this file invalidates every
expectation under tests/golden/expected/ -- see docs/words.md#tests.
"""
import random

random.seed(20260831)

VOCAB = 1400          # distinct words before min-count pruning
TOKENS = 400_000      # total tokens

# Words the default battery and the analogy checks look for, given the
# top (most frequent) ids so they always survive pruning.
SEEDED = [
    "the", "one", "king", "france", "computer", "water", "music", "red", "war",
    "man", "woman", "paris", "rome", "good", "better", "bad", "queen", "boy",
    "girl", "italy", "berlin", "germany", "big", "bigger", "small",
]
words = SEEDED + [f"w{i:04d}" for i in range(VOCAB - len(SEEDED))]

# Zipf-ish weights so subsampling and the ^0.75 flattening have something to do.
weights = [1.0 / (i + 1) ** 0.9 for i in range(len(words))]

out = random.choices(words, weights=weights, k=TOKENS)
with open("tests/golden/words/corpus.txt", "w") as f:
    f.write(" ".join(out))

print(f"wrote {TOKENS} tokens over {len(set(out))} distinct words")
