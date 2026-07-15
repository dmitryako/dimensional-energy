# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToCr.py Created : 2025-06-26 at 2:05 pm by Dmitry.A.Konovalov@gmail.com

# func_lcr_to_cr.py  –– Port of atom.wf.lcr.func.FuncLcrToCr
# (Java original: 11 Jul 2008, 15 : 42 : 05)
#
# • Implements the Func interface: f(x) = CR = exp(x)
# -------------------------------------------------------------------------

import math

from qm_math.func.Func import Func


# from qm_math.vec.Vec import Func


# from func import Func


class FuncLcrToCr(Func):
    """Concrete implementation of Func:  f(x) = exp(x)."""

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        return math.exp(x)


if __name__ == "__main__":
    f = FuncLcrToCr()
    print(f.calc(0))      # 1.0
    print(f.calc(1.5))    # ≈ 4.481689
