# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : QuadrPts5.py Created : 2025-06-26 at 4:30 pm by Dmitry.A.Konovalov@gmail.com

# quadr_pts5.py  –– Port of math.integral.QuadrPts5
# (Java original: 10 Jul 2008, 16 : 42 : 54)
#
# • Implements the 5-point closed Newton–Cotes rule (often nick-named
#   “Bode’s rule”) on a uniform StepGrid.
# • Extends QuadrStep with ptsNum = 5, so   nextN = 4.
# • Provides an *optional* helper   calcFuncInt_DEV(vec)   that builds a
#   **cumulative integral table** by mixing lower-order rules until the
#   first 5-point stencil is available (same algorithm as the Java code).
# -------------------------------------------------------------------------

from __future__ import annotations
import numpy as np

from javax.utilx.log.Log import Log
from qm_math.func.FuncVec import FuncVec
from qm_math.func.intrg.IntgPts7 import IntgPts7
from qm_math.integral.QuadrStep import QuadrStep
from qm_math.integral.QuadrStep2 import QuadrStep2
from qm_math.integral.QuadrStep3 import QuadrStep3
from qm_math.integral.QuadrStep4 import QuadrStep4
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


class QuadrPts5(QuadrStep):
    PTS_N: int = 5
    log = Log.getLog("QuadrPts5")

    # ------------------------------------------------------------------ #
    def __init__(self, grid: StepGrid):
        """
        Java: public QuadrPts5(StepGrid grid)
        """
        super().__init__(grid, ptsNum=QuadrPts5.PTS_N)  # nextN = 4

        if not self.isValid(size=self.size()):
            raise ValueError(QuadrPts5.log.error(f"invalid size={self.size()}"))

        self._loadWeights(grid.getGridStep())

    # ------------------------------------------------------------------ #
    # Internal weight filler (pattern length 4)                           #
    # ------------------------------------------------------------------ #
    def _loadWeights(self, step: float) -> None:
        """
        Fill self._arr with Bode-rule weights across the full grid:

            interior repeating pattern  [14w, 32w, 12w, 32w]
            with  w = 2h / 45   (h = step)
            endpoints halved.
        """
        w = 2.0 * step / 45.0
        pattern = (14.0 * w, 32.0 * w, 12.0 * w, 32.0 * w)

        for i in range(self.size()):
            self.set(i, pattern[i % 4])

        # half-weight the boundaries
        y_arr = self.getArr()  # todo!!! note inplace operation
        # self._arr[0] *= 0.5
        # self._arr[self.size() - 1] *= 0.5
        y_arr[0] *= 0.5
        y_arr[self.size() - 1] *= 0.5

    # ------------------------------------------------------------------ #
    # Public helpers (names exactly as in Java)                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def makeQuadrArr(step: float) -> np.ndarray:
        """Single-interval 5-point weights  [14w, 32w, 12w, 32w] ."""
        w = 2.0 * step / 45.0
        return np.array([14.0 * w, 32.0 * w, 12.0 * w, 32.0 * w], dtype=np.float64)

    def makeQuadrFuncInt(self, step: float) -> Vec:
        return Vec(QuadrPts5.makeQuadrArr(step))

    # ------------------------------------------------------------------ #
    # Cumulative integral builder (DEV helper, mirrors Java logic)       #
    # ------------------------------------------------------------------ #
    def calcFuncInt_DEV(self, funcV: Vec) -> FuncVec:
        """
        Constructs an integral table  I[i] = ∫_{x0}^{x_i} f(x) dx  using
        mixed trapezoid / Simpson / 3/8 / 5-point Newton–Cotes rules,
        exactly as in the original Java switch-ladder.
        """
        f = funcV.getArr()
        resF = FuncVec(self.getX())  # integral vector
        res = resF.getArr()
        step = self.getStepGrid().getGridStep()

        # pre-compute single-interval weight arrays
        a2 = QuadrStep2.makeQuadrArr(step)
        a32 = QuadrStep3.makeQuadrArr_2From3(step)
        a3 = QuadrStep3.makeQuadrArr(step)
        a4 = QuadrStep4.makeQuadrArr(step)
        a5 = IntgPts7.makePts5(step)

        curr_tot = 0.0
        curr = 0.0
        pts_per_step = self.getNextN()  # 4

        res[0] = 0.0  # integral from A to A
        for i in range(1, self.size()):  # NOTE: starts at 1
            switch_type = (i - 1) % pts_per_step

            if switch_type == 0:
                # first sub-interval uses the special 3-point (2-from-3) rule
                curr = a32[0] * f[i - 1] + a32[1] * f[i] + a32[2] * f[i + 1]

            elif switch_type == 1:
                curr = a3[0] * f[i - 2] + a3[1] * f[i - 1] + a3[2] * f[i]

            elif switch_type == 2:
                curr = (a4[0] * f[i - 3] + a4[1] * f[i - 2] +
                        a4[2] * f[i - 1] + a4[3] * f[i])

            elif switch_type == 3:
                curr = (a5[0] * f[i - 4] + a5[1] * f[i - 3] +
                        a5[2] * f[i - 2] + a5[3] * f[i - 1] +
                        a5[4] * f[i])
                curr_tot += curr
                curr = 0.0

            res[i] = curr_tot + curr

        return resF


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    import numpy as np

    # Uniform StepGrid over [0, 1] with 13 points  (step = 1/12, size = 13)
    first, num_steps, step = 0.0, 12, 1.0 / 12.0
    grid = StepGrid.from_first_numSteps_stepVal(first=first, numSteps=num_steps, stepVal=step)

    # Test integrand  f(x) = x²
    x_vals = grid.getArr().copy()
    y_vals = x_vals ** 2
    f_vec = FuncVec(grid, y_vals)

    # Quadrature weights vector
    qp5 = QuadrPts5(grid)
    print("Bode weights:", qp5.toCSV())

    # Build cumulative integral
    int_tbl = qp5.calcFuncInt_DEV(f_vec)
    print("\nIntegral table:")
    print(int_tbl.toCSV())
