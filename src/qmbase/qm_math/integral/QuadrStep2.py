# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : QuadrStep2.py Created : 2025-06-26 at 3:41 pm by Dmitry.A.Konovalov@gmail.com

# quadr_step2.py  –– Port of math.integral.QuadrStep2
# (Java original: 02 Mar 2012, 15 : 31)
#
# • Implements the basic **trapezoidal rule** (2-point Newton–Cotes)
#   weights over a uniform StepGrid.
# • Extends QuadrStep with `ptsNum = 2` (so nextN = 1).
# • Keeps every public method name exactly as in the Java file.
# -------------------------------------------------------------------------

import numpy as np

from javax.utilx.log.Log import Log
from qm_math.integral.QuadrStep import QuadrStep
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


# from quadr_step import QuadrStep, Log
# from vec        import Vec
# from step_grid  import StepGrid


class QuadrStep2(QuadrStep):
    log = Log.getLog("QuadrStep2")
    MIN_GRID_SIZE: int = 2

    # ------------------------------------------------------------------ #
    def __init__(self, grid: StepGrid):
        """
        Java: public QuadrStep2(StepGrid grid)
        """
        super().__init__(grid, ptsNum=2)  # nextN = 1 (every interval)

        if self.size() < QuadrStep2.MIN_GRID_SIZE:
            raise ValueError(
                QuadrStep2.log.error(f"grid size must be ≥ {self.MIN_GRID_SIZE}")
            )
        if not self.isValid(size=self.size()):
            raise ValueError(QuadrStep2.log.error(f"invalid size={self.size()}"))

        # for trapezoidal rule, size is always valid (override below)
        self.loadWeights(grid.getGridStep())

    # ------------------------------------------------------------------ #
    # Static helper replicating Java signature                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def makeQuadrArr(step: float) -> np.ndarray:
        """
        Returns [h/2, h/2]  — trapezoidal single-interval weights.
        """
        tmp = 0.5 * step
        return np.array([tmp, tmp], dtype=np.float64)

    # Factory method (instance) ------------------------------------------
    def makeQuadrFuncInt(self, step: float) -> Vec:
        return Vec(QuadrStep2.makeQuadrArr(step))

    # ------------------------------------------------------------------ #
    # Internal weight filler                                             #
    # ------------------------------------------------------------------ #
    def loadWeights(self, step: float) -> None:
        """
        Full-grid trapezoidal weights:
            interior points:    h
            endpoints (0, last): h / 2
        """
        h = step
        self._arr[:] = h
        self._arr[0] *= 0.5
        self._arr[self.size() - 1] *= 0.5

    # ------------------------------------------------------------------ #
    # Validation (always true for trapezoidal rule)                      #
    # ------------------------------------------------------------------ #
    def isValid(self, size: int) -> bool:
        return True


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Uniform StepGrid: first = 0, last = 1, step = 0.1  ⇒  11 points
    first, num_steps, step = 0.0, 10, 0.1
    grid = StepGrid.from_first_numSteps_stepVal(first=first, numSteps=num_steps, stepVal=step)

    qs2 = QuadrStep2(grid)

    print("Trapezoidal weights on full grid:")
    print(qs2.toCSV())

    # Single-interval helper
    print("Single interval weights:", QuadrStep2.makeQuadrArr(step))
