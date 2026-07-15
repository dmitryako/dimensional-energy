# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : TransLrToR.py Created : 2025-06-27 at 12:58 pm by Dmitry.A.Konovalov@gmail.com

# trans_lr_to_r.py  –– Port of atom.wf.lr.TransLrToR
# --------------------------------------------------------------------
# • Keeps the Java names (`getR2`, `getDivSqrtR`, …) so old call-sites
#   compile unchanged.
# • Extends the Python `FuncVec` class you already converted.
# --------------------------------------------------------------------

from __future__ import annotations
import math

from qm_math.func.Func import Func
from qm_math.func.FuncVec import FuncVec
from qm_math.func.simple.FuncExp import FuncExp
from qm_math.vec.Vec import Vec


# from vec      import Vec
# from func     import Func
# from func_vec import FuncVec
# from func_exp import FuncExp


class TransLrToR(FuncVec):
    """
    Transform functions defined on an equally-spaced **log-r** grid
    (x = ln r) into various r-weighted forms.

    On construction the object stores **exp(x)** itself (i.e. *r*),
    exactly like the Java original:
        super(x, new FuncExp(1))  →  y = eˣ = r
    """

    # ----------------------------------------------------------------
    def __init__(self, x: Vec):
        super().__init__(x, FuncExp(1.0))  # y = exp(x) = r
        self._R2: Vec | None = None
        self._R: Vec | None = None
        self._divSqrtR: Vec | None = None
        self._mapLogRToR2 = FuncVec(x, _FunctorLogRToR2())

    # ----------------------------------------------------------------
    # Java getters reproduced verbatim
    def getMapLogRToR2(self) -> FuncVec:
        return self._mapLogRToR2

    def getR2(self) -> Vec:
        if self._R2 is None:
            self._R2 = FuncVec(self.getX(), _FunctorLogRToR2())
            self._R2 = self._R2.getY()  # force vec
        return self._R2

    def getR(self) -> Vec:
        if self._R is None:
            self._R = FuncVec(self.getX(), _FuncLogRToR())
            self._R = self._R.getY()  # force vec
        return self._R

    def getDivSqrtR(self) -> Vec:
        if self._divSqrtR is None:
            self._divSqrtR = FuncVec(self.getX(), _FuncLogRToDivSqrtR())
            self._divSqrtR = self._divSqrtR.getY()
        return self._divSqrtR


# --------------------------------------------------------------------
# Helper functors — direct ports of the Java private classes
# --------------------------------------------------------------------
class _FuncLogRToR(Func):
    """f(x) = r = exp(x)"""
    def calc(self, x: float) -> float:
        return math.exp(x)

class _FunctorLogRToR2(_FuncLogRToR):
    """f(x) = r²"""

    def calc(self, x: float) -> float:
        r = super().calc(x)
        return r * r

class _FuncLogRToDivSqrtR(_FuncLogRToR):
    """f(x) = 1 / √r"""

    def calc(self, x: float) -> float:
        r = super().calc(x)
        return 1.0 / math.sqrt(r)
