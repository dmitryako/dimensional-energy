# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToSqrtCr.py Created : 2025-06-26 at 2:29 pm by Dmitry.A.Konovalov@gmail.com


# func_lcr_to_sqrt_cr.py  –– Port of atom.wf.lcr.func.FuncLcrToSqrtCr
# (Java original: 14 May 2010, 17 : 04 : 17)
#
# • Extends FuncLcrToCr (CR = exp(x)).
# • Implements  f(x) = √CR .
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcrToCr import FuncLcrToCr


# from func_lcr_to_cr import FuncLcrToCr


class FuncLcrToSqrtCr(FuncLcrToCr):
    """
    Concrete function:

        f(x) = √CR
             = √(exp(x))
             = exp(x / 2)
    """

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        cr = super().calc(x)           # exp(x)
        return math.sqrt(cr)


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToSqrtCr()
    for x in (0.0, 1.0, 2.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected output:
    # f(0.0) = 1
    # f(1.0) ≈ 1.64872       (exp(0.5))
    # f(2.0) ≈ 2.71828       (exp(1))
