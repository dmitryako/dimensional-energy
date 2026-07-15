# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToCr2DivR.py Created : 2025-06-26 at 2:14 pm by Dmitry.A.Konovalov@gmail.com

# func_lcr_to_cr2_div_r.py  –– Port of atom.wf.lcr.func.FuncLcrToCr2DivR
# (Java original: 11 Jul 2008, 15 : 32 : 22)
#
# • Extends  FuncLcrToR  (not yet shown in the thread, but assumed to be
#   a concrete subclass of FuncLcr that implements  calc(x) = r ).
# • Python constructor forwards the parameter *p* to the parent.
# • calc(x) reproduces the Java algorithm:
#       r  = super().calc(x)
#       if r == 0 : return 0
#       cr = r + getC()
#       return cr² / r
# -------------------------------------------------------------------------

import math

from atom.wf.lcr.func.FuncLcrToR import FuncLcrToR


# from func_lcr_to_r import FuncLcrToR     # make sure this base class is ported


class FuncLcrToCr2DivR(FuncLcrToR):
    """
    Concrete function:

        f(x) = (CR)² / r
             = (r + c)² / r         ,  where
               r = super().calc(x)
               c = getC()
    """

    # Java: public FuncLcrToCr2DivR(double p)
    def __init__(self, p: float):
        super().__init__(p)

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        r = super().calc(x)            # r = f_r(x)
        if r == 0.0:
            return 0.0
        cr = r + self.getC()
        return (cr * cr) / r


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Quick demo with a *dummy* FuncLcrToR implementation
    # ---------------------------------------------------
    # If you already ported FuncLcrToR, remove the inner dummy class
    # and import your real version.

    class DummyFuncLcrToR(FuncLcrToR):
        """minimal stand-in: r = max(exp(x) − c, 0)"""
        def calc(self, x: float) -> float:
            r = math.exp(x) - self.getC()
            return r if r > 0.0 else 0.0

    # Monkey-patch the base class used above so the demo runs
    FuncLcrToCr2DivR.__bases__ = (DummyFuncLcrToR,)

    f = FuncLcrToCr2DivR(p=0.5)     # here “p” is forwarded to DummyFuncLcrToR
    for x in (0.0, 0.5, 1.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Expected output (with dummy r):
    # f(0.0) = 0
    # f(0.5) = ...
    # f(1.0) = ...
