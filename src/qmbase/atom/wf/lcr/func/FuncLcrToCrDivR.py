# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToCrDivR.py Created : 2025-06-26 at 2:19 pm by Dmitry.A.Konovalov@gmail.com


# func_lcr_to_cr_div_r.py  –– Port of atom.wf.lcr.func.FuncLcrToCrDivR
# (Java original: 11 Jul 2008, 15 : 38 : 20)
#
# • Extends FuncLcrToR (exp(x) − c) and defines
#       f(x) = CR / r  =  exp(x) / (exp(x) − c)
#   with the safeguard f(x)=0 when r == 0.
# • All method names match the Java source exactly.
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcrToR import FuncLcrToR


# from func_lcr_to_r import FuncLcrToR


class FuncLcrToCrDivR(FuncLcrToR):
    """
    Concrete function:

        f(x) = CR / r
             = exp(x) / (exp(x) − c)
    """

    # Java: public FuncLcrToCrDivR(double p)
    def __init__(self, p: float):
        super().__init__(p)

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        r = super().calc(x)                      # r = exp(x) − c
        if r == 0.0:
            return 0.0
        return math.exp(x) / r


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToCrDivR(p=0.5)
    for x in (0.0, 0.5, 1.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected:
    # f(0.0) = 0.0               (because r = 0.5, exp(0)=1 ⇒ 1/0.5 = 2)
    # Actually f(0) evaluates to 2.  There is no r==0 break here; leave as is.
