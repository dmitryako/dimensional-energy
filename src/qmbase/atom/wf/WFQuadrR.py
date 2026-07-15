# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : WFQuadrR.py Created : 2025-06-27 at 12:53 pm by Dmitry.A.Konovalov@gmail.com

# wf_quadr_r.py  –– Port of atom.wf.WFQuadrR
# -------------------------------------------------------------------
# * Inherits WFQuadrD1 → QuadrPts5 → QuadrStep → Quadr → Vec
# * Acts as the “plain-r” quadrature weight vector plus a couple of
#   helper variants that already include the 1/r   or 1/r² factors.
# -------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

from atom.wf.WFQuadrD1 import WFQuadrD1
from qm_math.func.FuncVec import FuncVec
from qm_math.func.simple.FuncPowAbsInt import FuncPowAbsInt
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


# from wf_quadr_d1 import WFQuadrD1
# from step_grid   import StepGrid
# from vec         import Vec
# from func_vec    import FuncVec
# from func_pow_int import FuncPowInt


class WFQuadrR(WFQuadrD1):
    """
    Radial quadrature with P(5) Newton–Cotes weights, plus helpers
    for integrals that contain 1/r or 1/r² factors.

    calcInt(wf,wf2)         → ∫ wf·wf2 dr
    calcInt(wf,wf2,wf3)     → ∫ wf·wf2·wf3 dr
    calcWithDivR(wf,…)      → ∫ wf·… / r   dr
    calcWithDivR2(wf,…)     → ∫ wf·… / r²  dr
    """

    # ----------------------------------------------------------------
    def __init__(self, x: StepGrid, scale=1):
        super().__init__(x)
        assert isinstance(x, StepGrid)
        self._scale = scale  # scale is needed to integrate Hermite polynoms
        if scale != 1:
            w = self.getY().arr
            new_w = w * scale
            self.setY(Vec(new_w))
            # self.multSelf(scale)

        self._wDivR:  Optional["WFQuadrR"] = None
        self._wDivR2: Optional["WFQuadrR"] = None

    # --------------- basic integrals (just forwarders) --------------
    # todo: below old java code. WHY?
    # def calcInt(self, wf: Vec, wf2: Vec) -> float:              # two-body
    #     return self.calc(wf, wf2)
    # def calcInt3(self, wf: Vec, wf2: Vec, wf3: Vec) -> float:   # three-body
    #     return self.calc(wf, wf2, wf3)

    # --------------- integrals with explicit 1/r factors ------------
    def calcWithDivR2(self, wf: Vec, wf2: Vec) -> float:
        return self._getWithDivR2().calc(wf, wf2)

    def calcWithDivR(self, wf: Vec, wf2: Vec, wf3: Vec) -> float:
        return self._getWithDivR().calc(wf, wf2, wf3)

    # --------------- helper factories -------------------------------
    def _getWithDivR(self) -> "WFQuadrR":
        if self._wDivR is None:
            self._wDivR = WFQuadrR(self.getStepGrid())
            factor = FuncVec(self.getX(), FuncPowAbsInt(a=1.0, k=-1))   # 1/r
            self._wDivR.multSelf(factor)
        return self._wDivR

    def _getWithDivR2(self) -> "WFQuadrR":
        if self._wDivR2 is None:
            self._wDivR2 = WFQuadrR(self.getStepGrid())
            factor = FuncVec(self.getX(), FuncPowAbsInt(1.0, -2))   # 1/r²
            self._wDivR2.multSelf(factor)
        return self._wDivR2

    # --------------- alias kept from Java ---------------------------
    def getWithDivR(self) -> "WFQuadrR":
        return self._getWithDivR()

    def getWithDivR2(self) -> "WFQuadrR":
        return self._getWithDivR2()
