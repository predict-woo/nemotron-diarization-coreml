import sys
sys.path.insert(0, "experiments")
import ablate  # noqa: E402
ablate.L = 576
from ablate import bench  # noqa: E402
from ablate3 import Body  # noqa: E402  (reuses its loaded model)
from nemodiar.export_model import EncoderStep  # noqa: E402
from nemodiar.reference_hf import load_hf  # noqa: E402
_, hf = load_hf()
step = EncoderStep.from_hf(hf, 576, fused=True)
for n in [1, 4, 8, 16, 24]:
    bench(Body(step, n).eval(), f"{n} layers, no head")
