# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : QuadrStep3.py Created : 2025-06-26 at 3:42 pm by Dmitry.A.Konovalov@gmail.com

# quadr_step3.py  –– Port of math.integral.QuadrStep3
# (Java original: 02 Mar 2012, 15 : 31)
#
# • Implements Simpson-rule (3-point) integration weights on a StepGrid.
# • Extends the previously ported QuadrStep (which handles grid + validation).
# • Public API identical: makeQuadrArr, makeQuadrArr_2From3, makeQuadrFuncInt.
# -------------------------------------------------------------------------

import numpy as np

from javax.utilx.log.Log import Log
from qm_math.integral.QuadrStep import QuadrStep
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


# from quadr_step import QuadrStep, Log          # reuse earlier stubs
# from vec        import Vec
# from step_grid   import StepGrid


class QuadrStep3(QuadrStep):
    log = Log.getLog("QuadrStep3")
    MIN_GRID_SIZE: int = 3  # must have at least 3 points

    # ------------------------------------------------------------------ #
    def __init__(self, grid: StepGrid):
        """
        Java: public QuadrStep3(StepGrid grid)
        Uses ptsNum = 3 (Simpson).
        """
        super().__init__(grid, ptsNum=3)  # nextN = 2 inside QuadrStep

        if self.size() < QuadrStep3.MIN_GRID_SIZE:
            raise ValueError(
                QuadrStep3.log.error(f"grid size must be ≥ {self.MIN_GRID_SIZE}")
            )
        if not self.isValid(size=self.size()):
            raise ValueError(QuadrStep3.log.error(f"invalid size={self.size()}"))

        # QuadrStep.isValid already checks (size-1) % nextN == 0
        self.loadWeights(grid.getGridStep())

    # ------------------------------------------------------------------ #
    # Static helpers replicate Java names / behaviour                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def makeQuadrArr(step: float) -> np.ndarray:
        """
        Returns array [h/3, 4h/3, h/3]  (Simpson 3-point weights).
        """
        tmp = step / 3.0
        return np.array([tmp, 4.0 * tmp, tmp], dtype=np.float64)

    @staticmethod
    def makeQuadrArr_2From3(step: float) -> np.ndarray:
        """
        Java helper for partial interval (2 weights from 3).
        """
        tmp = step / 12.0
        return np.array([5.0 * tmp, 8.0 * tmp, -1.0 * tmp], dtype=np.float64)

    # Factory method matching Java signature
    def makeQuadrFuncInt(self, step: float) -> Vec:
        return Vec(QuadrStep3.makeQuadrArr(step))

    # ------------------------------------------------------------------ #
    # Internal weight initialisation ------------------------------------ #
    def loadWeights(self, step: float) -> None:
        """
        Fills self._arr with Simpson weights for the *entire* grid:
            interior pattern  [2h/3, 4h/3, 2h/3, 4h/3, …]
        and half-weights at the boundaries.
        """
        tmp = step / 3.0
        a0, a1 = 2.0 * tmp, 4.0 * tmp  # alternating interior weights
        for i in range(self.size()):
            self.set(i, a0 if i % 2 == 0 else a1)

        # half-weights at the ends
        y_arr = self.getArr()  # todo!!! note inplace operation
        # self._arr[0] *= 0.5
        # self._arr[self.size() - 1] *= 0.5
        y_arr[0] *= 0.5
        y_arr[self.size() - 1] *= 0.5


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Build a uniform StepGrid: x ∈ [0,1] with 11 points (step = 0.1)
    first, num_steps, step = 0.0, 10, 0.1
    last = first + num_steps * step
    # from math.vec.grid.step_grid import StepGrid  # adapt import path if needed
    # grid = StepGrid(first, num_steps, step)
    grid = StepGrid(first=first, last=last, size=num_steps+1)

    qs3 = QuadrStep3(grid)

    print("Simpson weights on full grid:")
    print(qs3.toCSV())

    # quick check: makeQuadrArr
    print("Single interval weights:", QuadrStep3.makeQuadrArr(step))
