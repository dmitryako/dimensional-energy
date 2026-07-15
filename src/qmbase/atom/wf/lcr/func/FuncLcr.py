# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcr.py Created : 2025-06-26 at 1:56 pm by Dmitry.A.Konovalov@gmail.com
from qm_math.func.Func import Func


# func_lcr.py  –– Port of atom.wf.lcr.func.FuncLcr
#
# • Keeps every Java method name (getC).
# • Two Java constructors → one Python __init__ that accepts EITHER
#     • a float  c,                       e.g.  FuncLcr(0.2)
#     • another FuncLcr instance to copy, e.g.  FuncLcr(existing)
#   The logic branches on isinstance(arg, FuncLcr).
# • The class is still “abstract”: it does not implement calc(x).

# from func import Func


class FuncLcr(Func):
    """
    Abstract base: concrete subclasses must implement calc(x).
    Mathematical context (same as Java comment):

        f(x) = r,
        x = ln(CR),   CR = c + r,   c > 0
        exp(x) = c + r  ⇒  r = exp(x) − c
    """

    def __init__(self, src):
        """
        Java ctor overloads:

            FuncLcr(double c)
            FuncLcr(FuncLcr from)

        Python version inspects *src*:

            • If *src* is a float (or int) → treat it as c.
            • If *src* is a FuncLcr      → copy its c.
        """
        if isinstance(src, FuncLcr):
            self._c: float = src._c
        else:
            self._c = float(src)

    # ---------- Java getter name kept intact -----------------------------
    def getC(self) -> float:
        return self._c

    # ---------- still abstract ------------------------------------------
    def calc(self, x: float) -> float:
        raise NotImplementedError("Sub-class must implement calc(x)")


if __name__ == "__main__":
    # demo.py
    # from func_lcr import FuncLcr

    class ExampleLcr(FuncLcr):
        # minimal concrete subclass for testing
        def calc(self, x: float) -> float:
            return x + self.getC()


    ex1 = ExampleLcr(0.5)
    print("c =", ex1.getC(), "calc(1) =", ex1.calc(1))  # c = 0.5 , calc(1) = 1.5

    ex2 = ExampleLcr(ex1)  # copy-constructor path
    print("c_copy =", ex2.getC())

