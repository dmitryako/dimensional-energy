# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLcrToCr2DivR2.py Created : 2025-06-26 at 2:14 pm by Dmitry.A.Konovalov@gmail.com
from atom.wf.lcr.func.FuncLcrToCrDivR import FuncLcrToCrDivR


# func_lcr_to_cr2_div_r2.py  –– Port of atom.wf.lcr.func.FuncLcrToCr2DivR2
# (Java original: 11 Jul 2008, 15 : 31 : 20)
#
# • Extends FuncLcrToCrDivR and squares its value:
#       f(x) = [CR / r]²
# • Method names are identical to the Java source.
# -------------------------------------------------------------------------

# from func_lcr_to_cr_div_r import FuncLcrToCrDivR


class FuncLcrToCr2DivR2(FuncLcrToCrDivR):
    """
    Concrete function:

        f(x) = (CR / r)²
    """

    # Java: public FuncLcrToCr2DivR2(double p)
    def __init__(self, p: float):
        super().__init__(p)

    # Java: public double calc(double x)
    def calc(self, x: float) -> float:
        yr = super().calc(x)           # CR / r
        if yr == 0.0:
            return 0.0
        return yr * yr                 # (CR / r)²


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    f = FuncLcrToCr2DivR2(p=0.5)
    for x in (0.0, 0.5, 1.0):
        print(f"f({x}) = {f.calc(x):.6g}")
    # Example output:
    # f(0.0) = 4
    # f(0.5) = 3.53678
    # f(1.0) = 2.69988
