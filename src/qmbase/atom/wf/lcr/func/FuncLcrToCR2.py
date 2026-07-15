# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToCR2.py Created : 2025-06-26 at 2:10 pm by Dmitry.A.Konovalov@gmail.com

# func_lcr_to_cr2.py  –– Port of atom.wf.lcr.func.FuncLcrToCR2
# (Java original: 11 Jul 2008, 15 : 41 : 25)
#
# • Extends FuncLcrToCr and overrides calc(x) so that
#       f(x) = (exp(x))²  =  exp(2 x)
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcrToCr import FuncLcrToCr


# from func_lcr_to_cr import FuncLcrToCr


class FuncLcrToCR2(FuncLcrToCr):
    """Concrete Func:  f(x) = CR² = (exp(x))² = exp(2·x)."""

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        # reuse parent implementation for clarity, although
        # math.exp(2*x) would be slightly faster.
        cr = super().calc(x)          # exp(x)
        return cr * cr


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToCR2()
    print("f(0)   =", f.calc(0))      # 1.0
    print("f(1.5) =", f.calc(1.5))    # ≈ exp(3) = 20.0855369232
