"""Feasibility: one function, `cond` branches at different lengths sharing the same weight constants."""

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

N, H, F, LMAX = 31, 512, 2048, 576
BUCKETS = [384, 448, 512, 576]
rng = np.random.default_rng(0)
W1 = [(rng.normal(size=(F, H)) * 0.02).astype(np.float32) for _ in range(N)]
W2 = [(rng.normal(size=(H, F)) * 0.02).astype(np.float32) for _ in range(N)]


@mb.program(input_specs=[mb.TensorSpec(shape=(1, LMAX, H)), mb.TensorSpec(shape=(1,), dtype=types.int32)],
            opset_version=ct.target.iOS18)
def prog(x, bucket):
    w1 = [mb.const(val=w, name=f"w1_{i}") for i, w in enumerate(W1)]
    w2 = [mb.const(val=w, name=f"w2_{i}") for i, w in enumerate(W2)]
    sel = mb.squeeze(x=bucket)

    def body(L):
        def f():
            h = mb.slice_by_size(x=x, begin=[0, 0, 0], size=[1, L, H])
            for a, b in zip(w1, w2):
                h = mb.add(x=h, y=mb.linear(x=mb.gelu(x=mb.linear(x=h, weight=a), mode="EXACT"), weight=b))
            return mb.pad(x=h, pad=[0, 0, 0, LMAX - L, 0, 0])
        return f

    def chain(i):
        if i == len(BUCKETS) - 1:
            return body(BUCKETS[i])
        return lambda: mb.cond(pred=mb.equal(x=sel, y=np.int32(i)), _true_fn=body(BUCKETS[i]), _false_fn=chain(i + 1))

    return chain(0)()


model = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                   minimum_deployment_target=ct.target.iOS18)
model.save("build/exp_cond.mlpackage")
ct.utils.compile_model("build/exp_cond.mlpackage", destination_path="build/exp_cond.mlmodelc")
print("saved")
