# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : QuadrStep4.py Created : 2025-06-26 at 3:42 pm by Dmitry.A.Konovalov@gmail.com

# quadr_step4.py  –– Port of math.integral.QuadrStep4
# (Java original: 02 Mar 2012, 15 : 30)
#
# • Implements the 4-point Simpson “3/8” rule weights on a uniform StepGrid.
# • Extends QuadrStep (ptsNum = 4  ⇒  nextN = 3).
# • Public API identical to Java (makeQuadrArr, makeQuadrFuncInt, etc.).
# -------------------------------------------------------------------------

import numpy as np

from javax.utilx.log.Log import Log
from qm_math.integral.QuadrStep import QuadrStep
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


# from quadr_step import QuadrStep, Log
# from vec        import Vec
# from step_grid  import StepGrid


class QuadrStep4(QuadrStep):
    log = Log.getLog("QuadrStep4")
    MIN_GRID_SIZE: int = 4

    # ------------------------------------------------------------------ #
    def __init__(self, grid: StepGrid):
        """
        Java constructor:  QuadrStep4(StepGrid grid)
        """
        super().__init__(grid, ptsNum=4)  # nextN = 3 inside QuadrStep

        if self.size() < QuadrStep4.MIN_GRID_SIZE:
            raise ValueError(
                QuadrStep4.log.error(f"grid size must be ≥ {self.MIN_GRID_SIZE}")
            )
        if not self.isValid(size=self.size()):
            raise ValueError(QuadrStep4.log.error(f"invalid size={self.size()}"))

        self.loadWeights(grid.getGridStep())

    # ------------------------------------------------------------------ #
    # Static helpers (names exactly as Java)                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def makeQuadrArr(step: float) -> np.ndarray:
        """
        Returns Simpson 3/8 weights for one interval:

            h = step
            weights = [3h/8, 9h/8, 9h/8, 3h/8]
        """
        tmp = 3.0 * step / 8.0
        return np.array([tmp, 3 * tmp, 3 * tmp, tmp], dtype=np.float64)

    def makeQuadrFuncInt(self, step: float) -> Vec:
        """Factory returning a Vec of the single-interval weights."""
        return Vec(QuadrStep4.makeQuadrArr(step))

    # ------------------------------------------------------------------ #
    # Internal weight initialisation                                    #
    # ------------------------------------------------------------------ #
    def loadWeights(self, step: float) -> None:
        """
        Fills self._arr with 3/8-rule weights across the whole grid:

            interior pattern: [2w, 3w, 3w]  repeating
            boundary points  : first and last weights halved
        where  w = 3*step/8.
        """
        w = 3.0 * step / 8.0
        pattern = (2.0 * w, 3.0 * w, 3.0 * w)  # length 3

        for i in range(self.size()):
            self.set(i, pattern[i % 3])

        # half-weights on the boundaries
        y_arr = self.getY().getArr()
        # self._arr[0] *= 0.5
        # self._arr[self.size() - 1] *= 0.5
        y_arr[0] *= 0.5
        y_arr[self.size() - 1] *= 0.5


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Uniform StepGrid: first=0, last=1, step=0.1  ⇒  11 points
    nextN = 4
    first, num_steps, step = 0.0, nextN*3, 0.1
    last = first + num_steps * step
    grid = StepGrid(first=first, last=last, size=num_steps+1)

    qs4 = QuadrStep4(grid)

    print("3/8-rule weights on full grid:")
    print(qs4.toCSV())

    # Single-interval weights helper
    print("Single interval weights:", QuadrStep4.makeQuadrArr(step))
