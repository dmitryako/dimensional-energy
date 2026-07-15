# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncExp.py Created : 2025-06-26 at 5:14 pm by Dmitry.A.Konovalov@gmail.com


# func_exp.py  –– Port of math.func.simple.FuncExp
# (Java original: 09 Jul 2008, 17 : 03 : 35)
#
# • Implements the Func interface:  f(x) = exp(c · x)
# • Constructor and method names are kept identical (`__init__`, `calc`).
# -------------------------------------------------------------------------

import math

from qm_math.func.Func import Func


# from func import Func


class FuncExp(Func):
    """Exponential function  f(x) = exp(c · x)."""

    def __init__(self, c: float):
        self._c: float = float(c)          # scale factor in exponent

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        return math.exp(self._c * x)


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncExp(c=1.0)          # e^x
    for x in (-1.0, 0.0, 1.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Output:
    # f(-1.0) = 0.367879
    # f(0.0)  = 1
    # f(1.0)  = 2.71828
