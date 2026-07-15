# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : IntgPts7.py Created : 2025-06-26 at 4:43 pm by Dmitry.A.Konovalov@gmail.com

# intg_pts7.py  –– Port of math.func.intrg.IntgPts7
# (Java original: 03 May 2012, 12 : 58)
#
# • Extends the NumPy-backed FuncVec.  Given a function tabulated on a
#   StepGrid, it builds an **integral table** using successive open
#   Newton-Cotes rules of length 3, 4, 5, 6, 7 (same formulae the Java
#   code used).
# • The algorithm is copied line-for-line; index maths is unchanged.
# • `calc_h` works **in-place** on self._arr (the integral y-values).
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Sequence
import numpy as np
import math

from javax.utilx.log.Log import Log
from qm_math.func.FuncVec import FuncVec
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


# from func_vec   import FuncVec
# from vec        import Vec
# from step_grid  import StepGrid


# --- minimal log stub -----------------------------------------------------
# class Log:
#     @staticmethod
#     def getLog(cls=None): return Log(cls)
#     def __init__(self, cls=None): self._name = getattr(cls, "__name__", "Log")
#     def error(self, msg): msg2 = f"[ERR] {self._name}: {msg}"; print(msg2); return msg2
#     def setDbg(self): pass
# --------------------------------------------------------------------------


class IntgPts7(FuncVec):
    """
    Integral table produced with mixed Newton-Cotes rules.
    """

    log = Log.getLog("IntgPts7")
    N_STEPS: int = 6  # distance between successive 7-pt stencils

    # ------------------------------------------------------------------ #
    def __init__(self, f: FuncVec, step: int = 1):
        """
        Java constructors:
            IntgPts7(f)         – step = 1
            IntgPts7(f, step)
        """
        super().__init__(f.getX())  # allocate y array, x is shared
        self._STEP: int = step
        self._calc(f)  # fill self._arr with integral values

    # ------------------------------------------------------------------ #
    # Internal driver --------------------------------------------------- #
    def _calc(self, f: FuncVec) -> None:
        # --- sanity checks ---------------------------------------------
        if not isinstance(f.getX(), StepGrid):
            raise ValueError(IntgPts7.log.error("IntgPts7 needs a StepGrid"))
        if f.size() < 7:
            raise ValueError(IntgPts7.log.error("IntgPts7 needs at least 7 grid points"))

        self._calc_h(f)

    # ------------------------------------------------------------------ #
    # Translation of the Java calc_h algorithm                           #
    # ------------------------------------------------------------------ #
    def _calc_h(self, fv: FuncVec) -> None:
        IntgPts7.log.setDbg(False)

        # res = self._arr  # integral output
        res = self.getArr()  # integral output
        f = fv.getArr()  # function values
        grid = fv.getX()
        h = grid.getGridStep()

        size = fv.size()
        step = self._STEP
        i = 0 if step > 0 else size - 1

        res[i] = 0.0  # starting integral at r = 0

        # --- first sub-interval (quadratic fit, 3 pts) -----------------
        i += step
        f1, f2, f3 = f[i - step], f[i], f[i + step]
        a = (f3 - 2.0 * f2 + f1) / 2.0
        b = (4.0 * f2 - f3 - 3.0 * f1) / 2.0
        intgl = (a / 3.0 + b / 2.0 + f1) * h
        res[i] = res[i - step] + intgl

        # --- next five partial Newton-Cotes rules ----------------------
        i += step
        res[i] = (h / 3.0) * self._calcPts3(i, f)

        i += step
        res[i] = (3.0 * h / 8.0) * self._calcPts4(i, f)

        i += step
        res[i] = (2.0 * h / 45.0) * self._calcPts5(i, f)

        i += step
        res[i] = (5.0 * h / 288.0) * self._calcPts6(i, f)

        # --- regular 7-point closed Newton-Cotes thereafter -----------
        h7 = h / 140.0
        i += step
        while 0 <= i < size:
            res[i] = res[i - step * IntgPts7.N_STEPS] + h7 * self._calcPts7(i, f)
            i += step

    # ------------------------------------------------------------------ #
    # Newton-Cotes coefficient helpers                                   #
    # ------------------------------------------------------------------ #
    def _calcPts7(self, idx: int, f: Sequence[float]) -> float:
        step = self._STEP
        coeffs = [41, 216, 27, 272, 27, 216, 41]
        res = 0.0
        for c in coeffs:
            res += c * f[idx]
            idx -= step
        return res

    def _calcPts3(self, idx: int, f: Sequence[float]) -> float:
        step = self._STEP
        coeffs = [1, 4, 1]
        res = 0.0
        for c in coeffs:
            res += c * f[idx]
            idx -= step
        return res

    def _calcPts4(self, idx: int, f: Sequence[float]) -> float:
        step = self._STEP
        coeffs = [1, 3, 3, 1]
        res = 0.0
        for c in coeffs:
            res += c * f[idx]
            idx -= step
        return res

    def _calcPts5(self, idx: int, f: Sequence[float]) -> float:
        step = self._STEP
        coeffs = [7, 32, 12, 32, 7]
        res = 0.0
        for c in coeffs:
            res += c * f[idx]
            idx -= step
        return res

    def _calcPts6(self, idx: int, f: Sequence[float]) -> float:
        step = self._STEP
        coeffs = [19, 75, 50, 50, 75, 19]
        res = 0.0
        for c in coeffs:
            res += c * f[idx]
            idx -= step
        return res

    # ---- static helper copied from Java (used elsewhere) --------------
    @staticmethod
    def makePts5(step: float) -> np.ndarray:
        tmp = 2.0 * step / 45.0
        return np.array([7 * tmp, 32 * tmp, 12 * tmp, 32 * tmp, 7 * tmp],
                        dtype=np.float64)


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Simple test: integrate f(r)=r on StepGrid r=[0,1] with 13 points
    import numpy as np

    # from step_grid import StepGrid

    first, num_steps, step = 0.0, 12, 0.08333333333333333  # 1/12
    grid = StepGrid.from_first_numSteps_stepVal(first=first, numSteps=num_steps, stepVal=step)

    f_vals = Vec(grid.getArr().copy())  # y = r
    f_vec = FuncVec(grid, f_vals)  # wrap into FuncVec

    int_tbl = IntgPts7(f_vec)  # build integral table
    print("Integral table y =", int_tbl.toCSV())
    print("Last value ≈ 0.5:", int_tbl.getLast())
