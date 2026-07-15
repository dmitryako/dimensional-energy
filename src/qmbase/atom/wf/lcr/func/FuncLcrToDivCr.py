# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToDivCr.py Created : 2025-06-26 at 2:22 pm by Dmitry.A.Konovalov@gmail.com


# func_lcr_to_div_cr.py  –– Port of atom.wf.lcr.func.FuncLcrToDivCr
# (Java original: 11 Jul 2008, 15 : 42 : 52)
#
# • Extends FuncLcrToCr (f(x) = CR = exp(x)).
# • Implements reciprocal:  f(x) = 1 / CR.
# • Java would quietly return +Infinity when CR == 0.  In CPython
#   1.0/0.0 raises ZeroDivisionError, so we guard and return math.inf
#   to preserve Java’s behaviour.
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcrToCr import FuncLcrToCr


# from func_lcr_to_cr import FuncLcrToCr


class FuncLcrToDivCr(FuncLcrToCr):
    """
    Concrete function:

        f(x) = 1 / CR   ,  CR = exp(x)
    """

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        cr = super().calc(x)                 # exp(x)
        return math.inf if cr == 0.0 else 1.0 / cr


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToDivCr()
    for x in (-2.0, 0.0, 1.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected:
    # f(-2.0) = 7.38906
    # f(0.0)  = 1
    # f(1.0)  = 0.367879
