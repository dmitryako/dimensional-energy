# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : DistMaxAbsErr.py Created : 2025-06-27 at 8:55 am by Dmitry.A.Konovalov@gmail.com

# dist_max_abs_err.py  –– Port of math.vec.metric.DistMaxAbsErr
# (Java original: 10 Jul 2008, 10 : 24 : 14)
#
# • Provides the same two *static-style* methods:
#       distSLOW(Vec, Vec)            – accepts two Vec objects
#       distSLOW(array, array)        – accepts two sequences / NumPy arrays
#   returning the **maximum absolute difference** between elements.
# • Logging kept minimal: a tiny stub mimicking the original interface.
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Sequence, Union
import numpy as np

from javax.utilx.log.Log import Log
from qm_math.vec.Vec import Vec


# from vec import Vec


# ------------- minimal logger stub (replace with real logger later) -----
# class Log:
#     @staticmethod
#     def getLog(cls=None) -> "Log":
#         return Log(cls)
#
#     def __init__(self, cls=None):
#         self._name = getattr(cls, "__name__", "Log")
#
#     # dummy debug methods so callers compile
#     def setDbg(self): pass
#     def dbg(self, *msg): print("[DBG]", *msg)
# ------------------------------------------------------------------------


class DistMaxAbsErr:
    log = Log.getLog("DistMaxAbsErr")

    # ---- overloaded convenience wrapper (Vec inputs) ------------------
    @staticmethod
    def distSLOW(v: Vec, v2: Vec) -> float:
        """Maximum absolute error between two Vec objects."""
        DistMaxAbsErr.log.setDbg(False)
        return DistMaxAbsErr._dist_arrays(v.getArr(), v2.getArr())

    # ---- numeric array version ---------------------------------------
    @staticmethod
    def _dist_arrays(a: Union[Sequence[float], np.ndarray],
                     b: Union[Sequence[float], np.ndarray]) -> float:
        """
        Maximum |a[i] − b[i]| over all i.  Raises if lengths differ.
        """
        DistMaxAbsErr.log.setDbg(False)

        arr1 = np.asarray(a, dtype=float)
        arr2 = np.asarray(b, dtype=float)

        if arr1.size != arr2.size:
            raise ValueError(f"different sizes: {arr1.size=} vs {arr2.size=}")

        diff = np.abs(arr1 - arr2)
        max_err = float(np.max(diff))

        # optional verbose tracing (mirrors Java debug prints)
        idx_max = int(np.argmax(diff))
        DistMaxAbsErr.log.dbg(
            f"MAX dist[i={idx_max}] = {max_err:g}  "
            f"v={arr1[idx_max]:g}  v2={arr2[idx_max]:g}"
        )
        return max_err

    # keep original Java method name alias for direct import use
    distSLOW_arrays = _dist_arrays


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Example with simple Python lists
    a = [1.0, 2.5, -0.3, 4.2]
    b = [1.0, 2.7, -0.1, 4.25]
    print("max |a-b| =", DistMaxAbsErr._dist_arrays(a, b))

    # Example with Vec objects
    # from vec import Vec
    v1 = Vec(a)
    v2 = Vec(b)
    print("Vec max |a-b| =", DistMaxAbsErr.distSLOW(v1, v2))
