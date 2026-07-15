# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToDivR.py Created : 2025-06-26 at 2:22 pm by Dmitry.A.Konovalov@gmail.com
from atom.wf.lcr.func.FuncLcrToR import FuncLcrToR


# func_lcr_to_div_r.py  –– Port of atom.wf.lcr.func.FuncLcrToDivR
# (Java original: 11 Jul 2008, 15 : 39 : 26)
#
# • Extends FuncLcrToR (r = exp(x) − c).
# • Implements reciprocal:  f(x) = 1 / r , returning 0 when r == 0
#   to match the Java guard.
# -------------------------------------------------------------------------

# from func_lcr_to_r import FuncLcrToR


class FuncLcrToDivR(FuncLcrToR):
    """
    Concrete function:

        f(x) = 1 / r
             = 1 / (exp(x) − c)
    """

    # Java: public FuncLcrToDivR(double p)
    def __init__(self, p: float):
        super().__init__(p)

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        r = super().calc(x)           # r = exp(x) − c
        if r == 0.0:
            return 0.0
        return 1.0 / r


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToDivR(p=0.5)
    for x in (0.0, 0.5, 1.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected output:
    # f(0.0) = 2
    # f(0.5) = 1.29744
    # f(1.0) = 0.82025
