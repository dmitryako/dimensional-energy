# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : TransLcrToR.py Created : 2025-06-26 at 2:38 pm by Dmitry.A.Konovalov@gmail.com
# trans_lcr_to_r.py  –– Port of atom.wf.lcr.TransLcrToR
# (Java original: 11 Jul 2008, 15 : 12 : 32)
# • Extends the NumPy-backed FuncVec you already ported.
# • Lazily builds and caches all auxiliary Vec objects on first access,
#   exactly like the Java version.
# • Replaces the old FastLoop call with the NumPy-based  self.calc(func) .
# -------------------------------------------------------------------------

import math

from _new25.dbg import dbg
from atom.wf.lcr.func.FuncLcrToCR2 import FuncLcrToCR2
from atom.wf.lcr.func.FuncLcrToCr import FuncLcrToCr
from atom.wf.lcr.func.FuncLcrToCr2DivR import FuncLcrToCr2DivR
from atom.wf.lcr.func.FuncLcrToCr2DivR2 import FuncLcrToCr2DivR2
from atom.wf.lcr.func.FuncLcrToCrDivR import FuncLcrToCrDivR
from atom.wf.lcr.func.FuncLcrToDivCr import FuncLcrToDivCr
from atom.wf.lcr.func.FuncLcrToDivR import FuncLcrToDivR
from atom.wf.lcr.func.FuncLcrToDivSqrtCr import FuncLcrToDivSqrtCr
from atom.wf.lcr.func.FuncLcrToR import FuncLcrToR
from atom.wf.lcr.func.FuncLcrToR2 import FuncLcrToR2
from atom.wf.lcr.func.FuncLcrToSqrtCr import FuncLcrToSqrtCr
from javax.utilx.log.Log import Log
from qm_math.func.FuncVec import FuncVec
from qm_math.vec.Vec import Vec


class TransLcrToR(FuncVec):
    log = Log.getLog("TransLcrToR")

    # ---------------- constructor ----------------------------------------
    # def __init__(self, x: Vec, r_min=0):
    def __init__(self, x: Vec, *, r_min):
        super().__init__(x)  # allocate y
        self._r_min = r_min
        dbg('r_min')
        # x = log(c + r)
        # exp(x) = c + r
        # r = exp(x) − c
        # c = exp(x) - r
        # c = exp(x_min) - r_min
        x_min = x.getFirst()
        dbg('x_min')
        # self._c: float = math.exp(x.getFirst())
        exp_x_min = math.exp(x_min)
        dbg('exp_x_min')
        c = exp_x_min - r_min
        dbg('c')
        self._c = c
        self._func: FuncLcrToR = FuncLcrToR(self._c)
        self.calc(self._func)  # y = r(x)

        # lazy-initialised caches
        self._CR2OverR2 = None
        self._CR2DivR = None
        self._r2 = None
        self._CR2 = None
        self._CR = None
        self._divR = None
        self._CRDivR = None
        self._divSqrtCR = None
        self._sqrtCR = None

    # @property # example
    # def ls(self) -> Ls: return self._ls
    @property
    def c(self): return self._c

    # ---------------- cached getters -------------------------------------
    def getR2(self) -> Vec:
        if self._r2 is None:
            self._r2 = FuncVec(self.getX(), FuncLcrToR2(self._c))
            self._r2 = self._r2.getY()  # force to vec
        return self._r2

    def getCR2DivR2(self) -> Vec:
        if self._CR2OverR2 is None:
            self._CR2OverR2 = FuncVec(self.getX(),
                                      FuncLcrToCr2DivR2(self._c))
            TransLcrToR.log.dbg("CR2OverR2=", self._CR2OverR2)
            self._CR2OverR2 = self._CR2OverR2.getY()  # force to vec
        return self._CR2OverR2

    def getCR2DivR(self) -> Vec:
        if self._CR2DivR is None:
            self._CR2DivR = FuncVec(self.getX(),
                                    FuncLcrToCr2DivR(self._c))
            self._CR2DivR = self._CR2DivR.getY()  # force to vec
        return self._CR2DivR

    def getDivR(self) -> Vec:
        if self._divR is None:
            self._divR = FuncVec(self.getX(), FuncLcrToDivR(self._c))
            self._divR = self._divR.getY()  # force to vec
        return self._divR

    def getCRDivR(self) -> Vec:
        if self._CRDivR is None:
            self._CRDivR = FuncVec(self.getX(), FuncLcrToCrDivR(self._c))
            self._CRDivR = self._CRDivR.getY()  # force to vec
        return self._CRDivR

    def getDivCR(self) -> Vec:
        if self._divCR is None:
            self._divCR = FuncVec(self.getX(), FuncLcrToDivCr())
            self._divCR = self._divCR.getY()  # force to vec
        return self._divCR

    def getDivSqrtCR(self) -> Vec:
        if self._divSqrtCR is None:
            self._divSqrtCR = FuncVec(self.getX(), FuncLcrToDivSqrtCr())
            TransLcrToR.log.dbg("divSqrtCR=", self._divSqrtCR)
            self._divSqrtCR = self._divSqrtCR.getY()  # force to vec
        return self._divSqrtCR

    def getSqrtCR(self) -> Vec:
        if self._sqrtCR is None:
            self._sqrtCR = FuncVec(self.getX(), FuncLcrToSqrtCr())
            TransLcrToR.log.dbg("sqrtCR=", self._sqrtCR)
            self._sqrtCR = self._sqrtCR.getY()  # force to vec
        return self._sqrtCR

    def getCR2(self) -> Vec:
        if self._CR2 is None:
            # self._CR2 = FuncVec(self.getX(), FuncLcrToCR2())
            fv = FuncVec(self.getX(), FuncLcrToCR2())
            self._CR2 = fv.getY()  # force vec
        return self._CR2

    def getCR(self) -> Vec:
        if self._CR is None:
            # self._CR = FuncVec(self.getX(), FuncLcrToCr())
            fv = FuncVec(self.getX(), FuncLcrToCr())
            self._CR = fv.getY()  # force vec
        return self._CR

    def getFunc(self) -> FuncLcrToR:
        return self._func

    def getPowR(self, k: int) -> Vec:
        if k == 1:
            return self
        elif k == 2:
            return self.getR2()
        raise ValueError(TransLcrToR.log.dbg("getPowR(int k); k not ready"))


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # build an x-grid:  x = ln(C + r), here choose C=0.8 and r ∈ [0,1]
    c_val = 0.8
    import numpy as np

    r_vals = np.linspace(0.0, 1.0, 6)
    x_vals = np.log(c_val + r_vals)
    x_vec = Vec(x_vals)

    tr = TransLcrToR(x_vec)

    print("r grid (y):", tr.toCSV())
    print("CR grid   :", tr.getCR().toCSV())
    print("1/r grid  :", tr.getDivR().toCSV())
    print("r² grid   :", tr.getPowR(2).toCSV())
