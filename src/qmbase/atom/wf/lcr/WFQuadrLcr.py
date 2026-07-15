# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : WFQuadrLcr.py Created : 2025-06-26 at 2:40 pm by Dmitry.A.Konovalov@gmail.com

# wf_quadr_lcr.py  –– Port of atom.wf.lcr.WFQuadrLcr
# --------------------------------------------------------------------
# Extends WFQuadrLr and wraps all Java helper-methods with identical
# names so existing source lines stay unchanged.
# --------------------------------------------------------------------

from __future__ import annotations
from typing import Optional

from _new25.dbg import dbg, set_dbg
from atom.wf.lcr.TransLcrToR import TransLcrToR
from atom.wf.lcr.func.FuncLcr import FuncLcr
from atom.wf.lr.WFQuadrLr import WFQuadrLr
from javax.utilx.log.Log import Log
from javax.utilx.log.kiss.KissLog import KissLog
from qm_math.func.FuncVec import FuncVec
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid

log = Log.getLog("WFQuadrLcr")
# log.setDbg(False)

import numpy as np
from functools import reduce
import operator

class WFQuadrLcr(WFQuadrLr):
    """
    Quadrature weights for an **equal-step logarithmic CR grid**
    (x = ln(c+r)).  Provides cached variants with extra (c+r)^n / r^m
    factors frequently used in atomic integrals.
    """

    # log = Log.getLog("WFQuadrLcr")

    # ----------------------------------------------------------------
    def __init__(self, log_cr_grid: StepGrid, *, r_min):
        super().__init__(log_cr_grid)
        dbg('r_min')
        self._r_min = r_min
        self._lcrToR = TransLcrToR(log_cr_grid, r_min=r_min)  # maps between x and r
        # print(self._lcrToR.getX().getArr())
        # print(self._lcrToR.getY().getArr())
        # cached variants
        self._wCR: Optional["WFQuadrLcr"] = None
        self._wCR2: Optional["WFQuadrLcr"] = None
        self._wCR2DivR: Optional["WFQuadrLcr"] = None
        self._wCR2DivR2: Optional["WFQuadrLcr"] = None
        log.dbg("WFQuadrLcr()=", self)

    def get_info(self):
        x1_arr = self.getX().arr
        r1_arr = self.getR().arr
        ret = f'nr1={x1_arr.shape[0]}, xmin={x1_arr[0]}, xmax={x1_arr[-1]}, rmin{r1_arr[0]}, rmax={r1_arr[-1]}'
        return ret
    # =========  integral API overrides  ==============================
    # All simply forward to the appropriate cached-weight variant
    def calcWithDivR2(self, wf: Vec, wf2: Vec) -> float:
        return self.getWithCR2DivR2().calc(wf, wf2)

    def calcWithDivR(self, wf: Vec, wf2: Vec, wf3: Vec) -> float:
        return self.getWithCR2DivR().calc(wf, wf2, wf3)

    # def calcInt(self, wf: Vec, wf2: Vec) -> float:
    #     return self.getWithCR2().calc(wf, wf2)
    def calcInt(self, *vecs: Vec) -> float:
        saved_dbg = set_dbg(False)
        res1 = self.getWithCR2().calc(*vecs)
        # return res1  # todo: into options?
        # check c
        c = self._lcrToR.c
        if c > 0:
            # x_min = log(r_min==0 + c)
            return res1
        assert c == 0
        # calc correction -infty to x_min
        # func_x = self.getCR2().arr.copy()  # REMEMBER getWithCR2
        # intg dr = intgr r dx # so wee need to load 1-CR, other CR will be approx
        func_x = self.getCR().arr.copy()  # todo!! remember .copy()!
        dbg(func_x)
        for i in range(len(vecs)):
            dbg(vecs[i].arr)
            func_x *= vecs[i].arr
            dbg(func_x)
        # Compute product of all vectors
        # func_x = reduce(operator.mul, vecs)
        x_arr = self.getX().arr
        dbg(x_arr)
        # quadr_dx = self.getY().arr
        # dbg(quadr_dx)
        corr = self.correction_neg_infty(func_x=func_x, x_arr=x_arr, method='const')
        dbg('res1')
        dbg('corr')
        res2 = res1 + corr
        dbg('res2')
        set_dbg(saved_dbg)
        return res2

    # def calcInt3(self, wf: Vec, wf2: Vec, wf3: Vec) -> float:
    #     return self.getWithCR2().calc(wf, wf2, wf3)

    # =========  cached weight variants  =============================
    def getWithCR(self) -> "WFQuadrLcr":
        if self._wCR is None:
            # # self._wCR.multSelf(Vec(self._lcrToR.getCR()))  # *= (c+r)
            self._wCR = WFQuadrLcr(self.getStepGrid(), r_min=self._r_min)
            log.dbg("w=", self._wCR)
            cr = self._lcrToR.getCR()
            log.dbg("cr=", cr)
            self._wCR.multSelf(cr)  # *= (c+r)

        log.dbg("w*CR=", self._wCR)
        return self._wCR

    def getWithCR2(self) -> "WFQuadrLcr":
        log = Log.getLog("WFQuadrLcr")
        if self._wCR2 is None:
            # # self._wCR2.multSelf(Vec(self._lcrToR.getCR2()))  # *= (c+r)²
            log.dbg("getStepGrid=", self.getStepGrid())
            self._wCR2 = WFQuadrLcr(self.getStepGrid(), r_min=self._r_min)  # major BUG 250726!
            log.dbg("w=", self._wCR2)
            cr2 = self._lcrToR.getCR2()
            log.dbg("cr2=", cr2)
            self._wCR2.multSelf(cr2)  # *= (c+r)²

        log.dbg("w*CR2=", self._wCR2)
        return self._wCR2

    def getWithCR2DivR(self) -> "WFQuadrLcr":
        if self._wCR2DivR is None:
            # was a bug WFQuadrLcr(self.getStepGrid()) was called without , r_min=self._r_min
            # self._wCR2DivR.multSelf(Vec(self._lcrToR.getCR2DivR()))  # *= (c+r)² / r
            self._wCR2DivR = WFQuadrLcr(self.getStepGrid(), r_min=self._r_min)
            cr2DivR = self._lcrToR.getCR2DivR()
            log.dbg("cr2DivR=", cr2DivR)
            self._wCR2DivR.multSelf(cr2DivR)  # *= (c+r)² / r

        log.dbg("_wCR2DivR=", self._wCR2DivR)
        return self._wCR2DivR

    def getCR2DivR(self) -> "Vec":  # just redirect
        return self._lcrToR.getCR2DivR()  #  # *= (c+r)² / r
    def getCR2DivR2(self) -> "Vec":  # just redirect
        return self._lcrToR.getCR2DivR2()  #  # *= (c+r)² / r
    def getCR2(self) -> "Vec":  # just redirect
        return self._lcrToR.getCR2()  #  # *= (c+r)² / r
    def getCR(self) -> "Vec":  # just redirect
        return self._lcrToR.getCR()  #  # *= (c+r)² / r

    def getWithCR2DivR2(self) -> "WFQuadrLcr":
        if self._wCR2DivR2 is None:
            # self._wCR2DivR2.multSelf(Vec(self._lcrToR.getCR2DivR2()))  # *= (c+r)² / r²
            self._wCR2DivR2 = WFQuadrLcr(self.getStepGrid(), r_min=self._r_min)  # todo fixed 250726 major BUG
            cr2DivR2 = self._lcrToR.getCR2DivR2()
            log.dbg("cr2DivR2=", cr2DivR2)
            self._wCR2DivR2.multSelf(cr2DivR2)  # *= (c+r)² / r²
            # extra_np = self._lcrToR.getCR2DivR2().arr
            # w = self.getY().arr.copy()  # need fresh copy
            # res = w * extra_np
            # dbg(res)
            # self._wCR2DivR2 = Vec(res)
        log.dbg("_wCR2DivR2=", self._wCR2DivR2)
        return self._wCR2DivR2

    # =========  miscellaneous helpers  ==============================
    def getDivSqrtCR(self) -> Vec:
        return self._lcrToR.getDivSqrtCR()
    def getSqrtCR(self) -> Vec:
        return self._lcrToR.getSqrtCR()

    def getLcrToR(self) -> TransLcrToR:
        return self._lcrToR

    # Java‐style alias retained
    def getR(self) -> Vec:
        # return self._lcrToR
        return self._lcrToR.getY()  #// dk250708 force to Vec!!

    def getPowR(self, k: int) -> Vec:
        return self._lcrToR.getPowR(k)

    def transRToLCR(self, f: FuncVec) -> None:
        """
        Convert a function *defined on r-grid* into its LCR form
        by multiplying with 1/√(c+r) and resetting the grid.
        """
        f.multSelf(self.getDivSqrtCR())
        f.setX(self.getX())

    def getLcrToRFunc(self) -> FuncLcr:
        return self._lcrToR.getFunc()

    def correction_neg_infty(self, *, func_x, x_arr, method='const'):
        log.dbg("x_arr =", x_arr)
        log.dbg("func_x =", func_x)
        x_min = x_arr[0]
        log.dbg("x_min =", x_min)

        r_arr = np.exp(x_arr[:5])
        log.dbg("r_arr =", r_arr)
        # x1, x2 = x_arr[:2]
        r1, r2 = r_arr[:2]
        r_min = r1
        f1, f2 = func_x[:2]
        # slope = (f2 - f1) / (x2 - x1)
        slope = (f2 - f1) / (r2 - r1)
        log.dbg("slope =", slope)
        # intercept = f1 - slope * x1
        intercept = f1 - slope * r1
        log.dbg("intercept =", intercept)
        # x0_cross = -intercept / slope  # y=0 = slope * x0_cross + intercept
        r0_cross = -intercept / slope  # y=0 = slope * x0_cross + intercept
        # log.dbg("x0_cross =", x0_cross)
        log.dbg("r0_cross NOT USED!! =", r0_cross)  # todo <--- not used!
        # Integrate from -∞ to x_min ~ limit x→-∞ (if slope>0, diverges)
        # Usually unsuitable unless slope<0 and f→0.
        # corr_lin = (x_min - x0_cross) * f1 / 2.
        corr_lin = (r_min - 0) * (f1 + intercept) / 2.
        log.dbg("corr_lin =", corr_lin)

        # todo:
        # # Fit f(x) ~ A exp(alpha x) to first two points
        # alpha = (np.log(f2) - np.log(f1)) / (x2 - x1)
        # log.dbg("alpha =", alpha)
        # A = f1 / np.exp(alpha * x1)
        # log.dbg("A =", A)
        # # Integral from -∞ to x_min
        # corr_exp = A / alpha * np.exp(alpha * x_min)
        #
        # log.dbg("corr_exp =", corr_exp)
        # log.dbg("corr_lin =", corr_lin)

        # assert slope > 0, f'check slope={slope}'
        # assert alpha > 0, f'check alpha={alpha}'
        # return corr_exp
        return corr_lin

from qm_math.func.FuncVecDbgView import FuncVecDbgView
from qm_math.vec.VecDbgView import VecDbgView
KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))
KissLog.register_formatter(FuncVec, lambda v: str(FuncVecDbgView(v)))

