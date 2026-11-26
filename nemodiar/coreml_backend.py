"""CoreML encoder-step backend.

Model layouts (scripts/convert.py):
  buckets - multifunction model, function `len_<L>` per bucket length L: `sequence` (1, L, 512) zero-padded +
            `key_bias` (1, 1, 1, L) key mask; the smallest bucket holding the step runs (default)
  fixed   - the same with a single function-less model
  range   - `sequence` (1, n, 512) of the exact step length (flexible shape), no padding, no mask
  ane     - Neural Engine formulation (nemodiar/ane_model.py, float16): `sequence` (1, 512, 1, L), `key_bias`
            (1, L, 1, 1), `valid` (1, 1, 1, L) -> `logits` (1, 8, 8, L), one function per bucket
"""

import numpy as np

from .pipeline import NumpyEmbedder

COMPUTE_UNITS = {
    "cpu": "CPU_ONLY",
    "cpu_and_gpu": "CPU_AND_GPU",
    "cpu_and_ne": "CPU_AND_NE",
    "all": "ALL",
}
NEG_BIAS = -1e4


def model_layout(path: str) -> tuple[str, list[int]]:
    import coremltools as ct

    spec = ct.utils.load_spec(path)
    if len(spec.description.functions):
        functions = spec.description.functions
        names = {i.name for i in functions[0].input}
        lengths = sorted(int(f.name.split("_")[1]) for f in functions)
        return ("ane" if "valid" in names else "buckets"), lengths
    seq = next(i for i in spec.description.input if i.name == "sequence").type.multiArrayType
    if seq.WhichOneof("ShapeFlexibility") == "shapeRange":
        return "range", [int(seq.shapeRange.sizeRanges[1].upperBound)]
    return "fixed", [int(seq.shape[1])]


class CoreMLBackend:
    def __init__(self, model_path: str, compute_units: str = "cpu_and_gpu"):
        import coremltools as ct

        units = getattr(ct.ComputeUnit, COMPUTE_UNITS[compute_units])
        self.layout, self.buckets = model_layout(model_path)
        self.max_len = self.buckets[-1]
        self.hidden = 512
        if self.layout in ("buckets", "ane"):
            self.models = {b: ct.models.MLModel(model_path, compute_units=units, function_name=f"len_{b}")
                           for b in self.buckets}
        else:
            self.models = {self.max_len: ct.models.MLModel(model_path, compute_units=units)}
        self._sequence = {b: np.zeros((1, b, self.hidden), np.float32) for b in self.buckets}
        self._bias = {b: np.zeros((1, 1, 1, b), np.float32) for b in self.buckets}
        self.embedder = NumpyEmbedder()

    def embed(self, features):
        return self.embedder(features)

    def bucket(self, n: int) -> int:
        for b in self.buckets:
            if b >= n:
                return b
        raise ValueError(f"step input of {n} frames exceeds the model's {self.max_len}")

    def encode(self, sequence):
        n = len(sequence)
        b = self.bucket(n)
        if self.layout == "ane":
            seq = np.zeros((1, self.hidden, 1, b), np.float16)
            seq[0, :, 0, :n] = sequence.T
            bias = np.full((1, b, 1, 1), NEG_BIAS, np.float16)
            bias[:, :n] = 0
            valid = np.zeros((1, 1, 1, b), np.float16)
            valid[..., :n] = 1
            out = self.models[b].predict({"sequence": seq, "key_bias": bias, "valid": valid})["logits"]
            return np.ascontiguousarray(out[0, :, :, :n].transpose(2, 1, 0).reshape(n * 8, 8), dtype=np.float32)
        if self.layout == "range":
            feeds = {"sequence": np.ascontiguousarray(sequence[None], np.float32)}
        else:
            seq, bias = self._sequence[b], self._bias[b]
            seq[0, :n] = sequence
            seq[0, n:] = 0.0
            bias[..., :n] = 0.0
            bias[..., n:] = NEG_BIAS
            feeds = {"sequence": seq, "key_bias": bias}
        out = self.models[b].predict(feeds)["logits"]
        return np.asarray(out[0, : n * 8], np.float32)
