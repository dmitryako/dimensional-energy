# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncConst.py Created : 2025-06-26 at 5:13 pm by Dmitry.A.Konovalov@gmail.com
from qm_math.func.Func import Func


# func_const.py  –– Port of math.func.simple.FuncConst
# (Java original: 10 Jul 2008, 16 : 54 : 10)
#
# • Implements the Func interface with a constant value c0.
# • Method names exactly match the Java source (`calc`).
# -------------------------------------------------------------------------

# from func import Func


class FuncConst(Func):
    """Constant function  f(x) = c₀  for all x."""

    def __init__(self, v: float):
        self._c0: float = float(v)

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        return self._c0


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncConst(3.14)
    for x in (0.0, -2.5, 10.0):
        print(f"f({x}) = {f.calc(x)}")
    # Output:
    # f(0.0) = 3.14
    # f(-2.5) = 3.14
    # f(10.0) = 3.14
