# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToR.py Created : 2025-06-26 at 2:16 pm by Dmitry.A.Konovalov@gmail.com


# func_lcr_to_r.py  –– Port of atom.wf.lcr.func.FuncLcrToR
# (Java original: 11 Jul 2008, 15 : 24 : 15)
#
# • Extends the earlier-ported  FuncLcr  base class.
# • Implements   f(x) = r = exp(x) − c .
# • Method names stay identical to the Java source.
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcr import FuncLcr

class FuncLcrToR(FuncLcr):
    # r = exp(x) − c
    def __init__(self, c: float):
        super().__init__(c)           # store constant c
    def calc(self, x: float) -> float:
        return math.exp(x) - self.getC()


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToR(c=0.5)
    for x in (0.0, 1.0, 2.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected:
    # f(0.0) = 0.5
    # f(1.0) ≈ 1.21828
    # f(2.0) ≈ 6.38906
