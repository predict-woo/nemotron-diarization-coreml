"""native/torch_topk.cpp (used by the Swift runtime) reproduces torch.topk on CPU, ties included."""

import numpy as np
import torch

from nemodiar.cache import topk_indices


def test_cpp_topk_matches_torch_on_ties():
    rng = np.random.default_rng(0)
    for trial in range(1500):
        n = int(rng.integers(30, 5000))
        k = int(rng.integers(1, min(n, 600)))
        kind = trial % 4
        if kind == 0:
            v = rng.integers(0, 5, n).astype(np.float32)
        elif kind == 1:
            v = rng.choice([-np.inf, 0.6931472, 1.0, np.inf, 0.2], n).astype(np.float32)
        elif kind == 2:
            v = np.where(rng.random(n) < 0.5, np.float32(0.6931472), rng.normal(size=n)).astype(np.float32)
        else:
            v = rng.normal(size=n).astype(np.float32)
        expected = torch.topk(torch.from_numpy(v), k, sorted=False)[1].numpy()
        got = topk_indices(v[None], k, "cpp")[0]
        np.testing.assert_array_equal(got, expected, err_msg=f"trial {trial} n={n} k={k}")
