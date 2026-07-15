# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : WFQuadrLr.py Created : 2025-06-27 at 12:58 pm by Dmitry.A.Konovalov@gmail.com

# wf_quadr_lr.py  –– Port of atom.wf.lr.WFQuadrLr
# --------------------------------------------------------------------
# “Log-r” quadrature = ordinary WFQuadrR plus convenient variant that
# multiplies the weights by r² (needed for expectation values in x = ln r).
# --------------------------------------------------------------------

from __future__ import annotations
from typing import Optional

from atom.wf.WFQuadrR import WFQuadrR
from atom.wf.lr.TransLrToR import TransLrToR
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


# from step_grid    import StepGrid
# from vec          import Vec
# from wf_quadr_r   import WFQuadrR
# from trans_lr_to_r import TransLrToR


class WFQuadrLr(WFQuadrR):
    """
    Quadrature weights defined on an **equally-spaced log-r grid**.

    • `getLrToR()` returns a `TransLrToR` helper with r-dependent maps.
    • `getWithR2()` lazily builds / caches a new WFQuadrLr whose weights
       already include an extra factor r².
    """

    # ---------------------------------------------------------------
    def __init__(self, grid: StepGrid):
        super().__init__(grid)
        self._lrToR = TransLrToR(self.getX())
        self._wR2: Optional["WFQuadrLr"] = None
        self._wR: Optional["WFQuadrLr"] = None  # new, needed for 1d-Hy

    def calcInt_1d(self, *vecs: Vec) -> float:
        return self.getWithR().calc(*vecs)

    def calcInt_1d_from3d(self, *vecs: Vec) -> float:
        return self.getWithR2().calc(*vecs)

    def getLrToR(self) -> TransLrToR:
        return self._lrToR

    # ---------------------------------------------------------------
    def getWithR2(self) -> "WFQuadrLr":
        # Return a *separate* WFQuadrLr whose weight vector equals
        # this one multiplied by r².
        if self._wR2 is None:
            self._wR2 = WFQuadrLr(self.getStepGrid())
            self._wR2.multSelf(Vec(self._lrToR.getR2()))     # element-wise *= r²
        return self._wR2

    def getWithR(self) -> "WFQuadrLr":
        # Return a *separate* WFQuadrLr whose weight vector equals
        # this one multiplied by r. Needed for 1d-Hy
        if self._wR is None:
            self._wR = WFQuadrLr(self.getStepGrid())
            self._wR.multSelf(Vec(self._lrToR.getR()))     # element-wise *= r
        return self._wR
