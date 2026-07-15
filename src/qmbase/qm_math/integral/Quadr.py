# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Quadr.py Created : 2025-06-26 at 3:30 pm by Dmitry.A.Konovalov@gmail.com

# quadr.py  –– Port of math.integral.Quadr
# (Java original: 10 Jul 2008, 16 : 27 : 21)
#
# • Extends the NumPy‐backed FuncVec (vector of y-values on an x-grid).
# • All dot-product evaluations are done with NumPy, no FastLoop needed.
# • Java had three overloaded  calc(...)  methods; here they are renamed
#   with explicit suffixes so you can instantly see which version is used:
#
#       calc_vec(f)                   ←  one extra Vec
#       calc_vec_vec(f, f2)           ←  two extra Vecs
#       calc_vec_vec_vec(f, f2, f3)   ←  three extra Vecs
#
#   The wrapper methods  calcInt(...)  keep their Java names and delegate
#   to the appropriate NumPy implementation.
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Any
import numpy as np

from javax.utilx.log.Log import Log
from qm_math.Mathx import Mathx
from qm_math.func.FuncVec import FuncVec
from qm_math.vec.Vec import Vec


# from func_vec import FuncVec
# from vec import Vec
# from mathx import Mathx


# very small log stub (replace with real logger later)
# class Log:
#     @staticmethod
#     def getLog(cls=None): return Log(cls)
#     def __init__(self, cls=None): self._name = getattr(cls, "__name__", "Log")
#     def dbg(self, *msg): print("[DBG]", *msg); return self


class Quadr(FuncVec):                                   # Numerical quadrature
    log = Log.getLog("Quadr")
    _START_IDX: int = 0
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Quadr):  # (FuncVec)
            #     public Quadr(Quadr from) {
            #         super(from);
            super().__init__(args[0])
        elif len(args) == 1 and isinstance(args[0], Vec):  # (x_vec)
            #     public Quadr(Vec x) {
            #         super(x);
            super().__init__(args[0])
        elif len(args) == 2 and isinstance(args[0], Vec) and isinstance(args[1], Vec):
            x, y = args
            super().__init__(x, y)
        else:
            raise TypeError("Unsupported constructor arguments for Quadr.")

    def calcInt(self, *vecs: Vec) -> float:
        # todo NOTE orig java code did NOT have calcInt(Vec f)??
        #     public double calcInt(Vec f, Vec f2) {
        #     public double calcInt(Vec f, Vec f2, Vec f3) {
        return self.calc(*vecs)

    def calc(self, *vecs: Vec) -> float:
        if len(vecs) == 1:
            return self.calc_vec(vecs[0])
        elif len(vecs) == 2:
            return self.calc_vec2(vecs[0], vecs[1])
        elif len(vecs) == 3:
            return self.calc_vec3(vecs[0], vecs[1], vecs[2])
        elif len(vecs) == 4:
            return self.calc_vec4(vecs[0], vecs[1], vecs[2], vecs[3])
        else:
            raise TypeError("calcInt supports 1–3 Vec arguments")

    # ------------------------------------------------------------------ #
    #  NumPy implementations                                             #
    # ------------------------------------------------------------------ #
    def calc_vec(self, f: Vec) -> float:
        """Dot-product of self · f  over the common slice length."""
        end = min(self.size(), f.size())
        a = self.getArr()[: end]
        b = f.getArr()[: end]
        return float(np.dot(a, b))

    def calc_vec2(self, f: Vec, f2: Vec) -> float:
        """Sum_i  self[i] * f[i] * f2[i]  over common length."""
        end = Mathx.min(self.size(), f.size(), f2.size())
        a = self.getArr()[: end]
        b = f.getArr()[: end]
        c = f2.getArr()[: end]
        return float(np.sum(a * b * c))

    def calc_vec3(self, f: Vec, f2: Vec, f3: Vec) -> float:
        """Sum_i  self[i] * f[i] * f2[i] * f3[i]  over common length."""
        # end = Mathx.min(self.size(), f.size(), f2.size(), f3.size())
        end = min(self.size(), f.size(), f2.size(), f3.size())
        a = self.getArr()[: end]
        b = f.getArr()[: end]
        c = f2.getArr()[: end]
        d = f3.getArr()[: end]
        return float(np.sum(a * b * c * d))

    def calc_vec4(self, f: Vec, f2: Vec, f3: Vec, f4: Vec) -> float:
        """Sum_i  self[i] * f[i] * f2[i] * f3[i]  * f4[i]  over common length."""
        # end = Mathx.min(self.size(), f.size(), f2.size(), f3.size(), f4.size())
        end = min(self.size(), f.size(), f2.size(), f3.size(), f4.size())
        w = self.getArr()[: end]
        f1 = f.getArr()[: end]
        f2 = f2.getArr()[: end]
        f3 = f3.getArr()[: end]
        f4 = f4.getArr()[: end]
        return float(np.sum(w * f1 * f2 * f3 * f4))


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # simple demo: integrate  y = x  against itself over 5 points
    import numpy as np
    x = Vec(np.linspace(0.0, 1.0, 5))
    y = FuncVec(x, x)              # y = x
    quad = Quadr(y)                # Quadr stores y; x grid is copied

    result1 = quad.calc_vec(y)     # ∑ x·x
    print("∑ x·x =", result1)

    # triple product ∑ x·x·x
    result2 = quad.calc_vec2(y, y)
    print("∑ x·x·x =", result2)
