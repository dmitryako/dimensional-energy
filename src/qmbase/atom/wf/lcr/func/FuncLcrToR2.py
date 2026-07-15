# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToR2.py Created : 2025-06-26 at 2:28 pm by Dmitry.A.Konovalov@gmail.com

# func_lcr_to_r2.py  –– Port of atom.wf.lcr.func.FuncLcrToR2
# (Java original: 11 Jul 2008, 15 : 28 : 09)
#
# • Extends FuncLcrToR (r = exp(x) − c).
# • Implements   f(x) = r² .
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcrToR import FuncLcrToR


# from func_lcr_to_r import FuncLcrToR


class FuncLcrToR2(FuncLcrToR):
    """
    Concrete function:

        f(x) = r²             where  r = exp(x) − c
    """

    # Java: public FuncLcrToR2(double c)
    def __init__(self, c: float):
        super().__init__(c)

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        r = super().calc(x)            # r = exp(x) − c
        return r * r                   # r²


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToR2(c=0.5)
    for x in (0.0, 0.5, 1.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected output:
    # f(0.0) = 0.25
    # f(0.5) ≈ 0.155257
    # f(1.0) ≈ 1.48577
