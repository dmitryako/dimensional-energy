# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : QuadrStep.py Created : 2025-06-26 at 3:35 pm by Dmitry.A.Konovalov@gmail.com
from javax.utilx.log.Log import Log
from qm_math.integral.Quadr import Quadr
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


# quadr_step.py  –– Port of math.integral.QuadrStep
# (Java original: 02 Mar 2012, 15 : 49)
#
# • Extends the NumPy-based Quadr class you just ported.
# • Keeps all public method names identical to the Java source.
# • `calcFuncInt_DEV` is still a placeholder and raises
#   NotImplementedError (the Java code threw IllegalArgumentException).
# -------------------------------------------------------------------------

class QuadrStep(Quadr):
    log = Log.getLog("QuadrStep")

    # ---------------- constructor ----------------------------------------
    def __init__(self, grid: StepGrid, ptsNum: int):
        """
        Java:  public QuadrStep(StepGrid grid, int ptsNum)

        • grid   – StepGrid of x-values
        • ptsNum – number of points per integration interval
        """
        super().__init__(grid)  # store y = grid (inherits Vec data)
        assert isinstance(grid, StepGrid)
        self._stepGrid: StepGrid = grid
        self._ptsN: int = ptsNum
        self._nextN: int = ptsNum - 1  # running # of points per *step*
        assert self.isValid(size=self.size())

    # ---------------- API matching Java ----------------------------------
    def calcFuncInt_DEV(self, f: Vec) -> Vec:
        """Placeholder – not implemented in the original Java either."""
        raise NotImplementedError(
            QuadrStep.log.error("calcFuncInt_DEV not implemented")
        )

    def getStepGrid(self) -> StepGrid:
        return self._stepGrid

    def isValid(self, *, size: int) -> bool:
        """
        Validates that (size-1) is an exact multiple of nextN.
        Raises ValueError with the original diagnostic when not valid.
        """
        if (size - 1) % self._nextN != 0:
            n = (size - 1) // self._nextN
            error = (
                f"if ((size - 1) % {self._nextN} != 0); "
                f"{(size - 1) % self._nextN}!=0; "
                f"nearest sizes = {self._nextN * n + 1} "
                f"or {self._nextN * (n + 1) + 1}"
            )
            raise ValueError(QuadrStep.log.error(error))
        return True

    # simple getters with Java names
    def getPtsN(self) -> int:   return self._ptsN

    def getNextN(self) -> int:  return self._nextN


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    import numpy as np

    # build a StepGrid of 11 points: first = 0, last = 1, step = 0.1
    num_steps = 8
    step = 0.1
    first = 0.0
    last = first + step * num_steps
    x_grid = StepGrid(first=first, last=last, size=num_steps + 1)

    quad_step = QuadrStep(x_grid, ptsNum=5)

    print("ptsN  =", quad_step.getPtsN())  # 5
    print("nextN =", quad_step.getNextN())  # 4
    print(f"isValid({quad_step.size()}) ->", quad_step.isValid(size=quad_step.size()))

    try:
        quad_step.isValid(size=12)  # should raise
    except ValueError as e:
        print("Expected validation error:", e)
