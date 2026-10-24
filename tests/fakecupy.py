"""CPU stand-in for CuPy: numpy arrays, but the CuPy-only code paths in town_xp are taken
(legacy RandomState stream, asnumpy, cupyx.scatter_add). Used to validate the GPU path statistically."""
import sys, types, numpy as np
cp = types.ModuleType("cupy")
for k in dir(np):
    if not k.startswith("__"): setattr(cp, k, getattr(np, k))
cp.random = types.SimpleNamespace(RandomState=np.random.RandomState)
cp.asnumpy = lambda x: np.asarray(x)
cx = types.ModuleType("cupyx"); cx.scatter_add = lambda a, idx, v: np.add.at(a, idx, v)
sys.modules["cupyx"] = cx
