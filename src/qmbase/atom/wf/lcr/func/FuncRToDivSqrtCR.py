# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncRToDivSqrtCR.py Created : 2025-06-26 at 2:30 pm by Dmitry.A.Konovalov@gmail.com

# func_r_to_div_sqrt_cr.py  –– Port of atom.wf.lcr.func.FuncRToDivSqrtCR
# (Java original: 28 Oct 2008, 10 : 20 : 29)
#
# • Extends FuncLcr, but its independent variable is **r** (not x).
# • Two Java constructors → one __init__(src) that accepts either
#       • a float  c
#       • another FuncLcr instance to copy
#   and forwards to FuncLcr.
# • Implements f(r) = 1 / √(c + r)   where c = self.getC().
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcr import FuncLcr


# from func_lcr import FuncLcr


class FuncRToDivSqrtCR(FuncLcr):
    """
    Concrete function (variable is r, not x):

        f(r) = 1 / √(c + r)
    """

    # Combines both Java constructors
    def __init__(self, src):
        """
        Java equivalents:
            FuncRToDivSqrtCR(double c)
            FuncRToDivSqrtCR(FuncLcr from)
        """
        super().__init__(src)

    # Java: public double calc(double r)
    def calc(self, r: float) -> float:
        return 1.0 / math.sqrt(self.getC() + r)


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f1 = FuncRToDivSqrtCR(0.5)   # pass c directly
    f2 = FuncRToDivSqrtCR(f1)    # copy-constructor path

    for r in (0.0, 0.5, 1.0):
        print(f"f1({r}) = {f1.calc(r):.6g}")
        print(f"f2({r}) = {f2.calc(r):.6g}")
    # Expected (c = 0.5):
    # f1(0.0) = 1.41421
    # f2(0.0) = 1.41421
    # f1(0.5) = 1
    # f2(0.5) = 1
    # f1(1.0) = 0.816497
    # f2(1.0) = 0.816497
