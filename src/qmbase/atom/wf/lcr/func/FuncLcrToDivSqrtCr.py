# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToDivSqrtCr.py Created : 2025-06-26 at 2:22 pm by Dmitry.A.Konovalov@gmail.com

# func_lcr_to_div_sqrt_cr.py  –– Port of atom.wf.lcr.func.FuncLcrToDivSqrtCr
# (Java original: 11 Jul 2008, 15 : 45 : 42)
#
# • Extends FuncLcrToCr (CR = exp(x)).
# • Implements  f(x) = 1 / √CR .
# • No extra safeguards: like Java, x → –∞ gives CR → 0, and Python
#   returns +inf for 1/0.  That matches Java’s +Infinity behaviour.
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcrToCr import FuncLcrToCr


# from func_lcr_to_cr import FuncLcrToCr


class FuncLcrToDivSqrtCr(FuncLcrToCr):
    """
    Concrete function:

        f(x) = 1 / √CR
             = 1 / √(exp(x))
             = exp(–x / 2)
    """

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        cr = super().calc(x)                 # exp(x)
        return 1.0 / math.sqrt(cr)


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToDivSqrtCr()
    for x in (0.0, 1.0, 2.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected output:
    # f(0.0) = 1
    # f(1.0) ≈ 0.606531
    # f(2.0) ≈ 0.367879
