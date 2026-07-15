# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncRToLcr.py Created : 2025-06-26 at 2:32 pm by Dmitry.A.Konovalov@gmail.com

# func_r_to_lcr.py  –– Port of atom.wf.lcr.func.FuncRToLcr
# (Java original: 11 Jul 2008, 15 : 25 : 13)
#
# • Extends FuncLcr but is used as   f(r) = ln(C + r)   where
#       C = exp(x0) − r0   (computed once in the constructor).
# • Keeps Java method names exactly:  calc(r)
# -------------------------------------------------------------------------

import math

from _new25.dbg import dbg
from atom.wf.lcr.func.FuncLcr import FuncLcr


# from func_lcr import FuncLcr


class FuncRToLcr(FuncLcr):
    """
    Concrete function (variable is r, not x):

        f(r) = ln(C + r) ,
        with   C = exp(x0) − r0   (c > 0)
    """

    # Java: public FuncRToLcr(double x0, double r0)
    def __init__(self, x0: float, r0: float):
        c = math.exp(x0) - r0         # ensure c > 0
        dbg('c')
        super().__init__(c)

    # Java: public double calc(double r)
    def calc(self, r: float) -> float:
        return math.log(self.getC() + r)


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Example: x0 = 0  ⇒  exp(0) = 1;  r0 = 0.2  ⇒  c = 0.8
    f = FuncRToLcr(x0=0.0, r0=0.2)

    for r in (0.0, 0.5, 1.0):
        print(f"f({r}) = {f.calc(r):.6g}")
    # Expected output with c = 0.8:
    # f(0.0) = ln(0.8)  ≈ -0.223144
    # f(0.5) = ln(1.3)  ≈  0.262364
    # f(1.0) = ln(1.8)  ≈  0.587787
#     f(0.0) = -0.223144
# f(0.5) = 0.262364
# f(1.0) = 0.587787
