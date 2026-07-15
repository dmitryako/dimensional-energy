# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : WFQuadrD1.py Created : 2025-06-27 at 12:51 pm by Dmitry.A.Konovalov@gmail.com

# wf_quadr_d1.py  –– Port of atom.wf.WFQuadrD1
# ------------------------------------------------------------------
# • Direct analogue of the Java abstract class.
# • Inherits from `QuadrPts5`.
# • Two abstract integrals are declared; concrete subclasses will
#   implement them just like in the original codebase.
# ------------------------------------------------------------------

from __future__ import annotations
from abc import ABC, abstractmethod

from qm_math.integral.QuadrPts5 import QuadrPts5
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


class WFQuadrD1(QuadrPts5, ABC):
    # todo? NOTE WFQuadrD1(QuadrPts5) Why not Pts7? try?

    # --------------------------------------------------------------
    def __init__(self, x: StepGrid) -> None:
        super().__init__(x)          # call QuadrPts5 constructor

    # --------------------------------------------------------------
    @abstractmethod
    def calcWithDivR2(self, wf: Vec, wf2: Vec) -> float:
        """∫  wf(r) * wf2(r) / r²  dr   (must be implemented)"""
        ...

    @abstractmethod
    def calcWithDivR(self, wf: Vec, wf2: Vec, wf3: Vec) -> float:
        """∫  wf(r) * wf2(r) * wf3(r) / r   dr   (must be implemented)"""
        ...

    # --------------------------------------------------------------
    def getR(self) -> Vec:
        """Return the underlying r-grid (alias for getX())."""
        return self.getX()
