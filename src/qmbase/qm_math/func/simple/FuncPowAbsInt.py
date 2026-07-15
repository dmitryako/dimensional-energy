# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncPowInt.py Created : 2025-06-26 at 5:16 pm by Dmitry.A.Konovalov@gmail.com
from qm_math.Mathx import Mathx
from qm_math.func.Func import Func


# func_pow_int.py  –– Port of math.func.simple.FuncPowAbsInt
# (Java original: 11 Jul 2008, 16 : 23 : 50)
#
# • Power-law with **integer** exponent k, using the Mathx.pow shortcut
#   you ported earlier (handles small |k| without calling math.pow).
# • Keeps Java field names (a, k) and method name `calc`.
# -------------------------------------------------------------------------

# from func import Func
# from mathx import Mathx


class FuncPowAbsInt(Func):
    """
    Power-law with integer exponent:
        f(x) = a · xᵏ
    where  k ∈ ℤ.
    """

    def __init__(self, a: float, k: int):
        self._a: float = float(a)
        self._k: int   = int(k)

    def calc(self, x: float) -> float:
        return self._a * Mathx.pow(abs(x), self._k)


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncPowAbsInt(a=3.0, k=4)      # f(x) = 3·x⁴
    for x in (-1.0, 0.0, 2.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected:
    # f(-1.0) = 3
    # f(0.0)  = 0
    # f(2.0)  = 48
