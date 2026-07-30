# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : DerivPts9.py Created : 2025-06-27 at 11:07 am by Dmitry.A.Konovalov@gmail.com

# deriv_pts9.py  –– Port of math.func.deriv.DerivPts9
# (Java original: 09 Jul 2008, 16 : 26 : 46)
#
# • First-derivative via 9-point finite-difference stencils
#   (coefficients from Bickley, Math. Gaz. 25 (1941), p. 708).
# • Logic, variable names, and comment layout follow the Java source
#   as closely as Python allows, so you can diff line-by-line.
# -------------------------------------------------------------------------

from __future__ import annotations
import numpy as np

from javax.utilx.log.Log import Log
from qm_math.func.FuncVec import FuncVec
from qm_math.func.FuncVecDbgView import FuncVecDbgView
from qm_math.func.simple.FuncPowAbsInt import FuncPowAbsInt
from qm_math.mtrx.MtrxDbgView import MtrxDbgView
from qm_math.mtrx.api.Mtrx import Mtrx
from qm_math.vec.VecDbgView import VecDbgView
from qm_math.vec.grid.StepGrid import StepGrid
from qm_math.vec.metric.DistMaxAbsErr import DistMaxAbsErr


class DerivPts9(FuncVec):
    log = Log.getLog("DerivPts9")
    MIN_GRID_SIZE = 9

    # -------- coefficient matrix 9×5 (same ordering as Java) ----------
    _coeff = np.array([
        [-109584., 322560., -564480., 752640., -705600.],
        [-5040., -64224., 141120., -141120., 117600.],
        [720., -11520., -38304., 80640., -50400.],
        [-240., 2880., -20160., -18144., 50400.],
        [144., -1536., 8064., -32256., 0.],
        [-144., 1440., -6720., 20160., -50400.],
        [240., -2304., 10080., -26880., 50400.],
        [-720., 6720., -28224., 70560., -117600.],
        [5040., -46080., 188160., -451584., 705600.]
    ], dtype=float)
    _N_ROW = 9
    _SIZE_1 = _N_ROW - 1
    _mid = _N_ROW // 2

    # ------------------------------------------------------------------
    def __init__(self, f: FuncVec):
        super().__init__(f.getX())  # same x-grid
        self._calc(f)

    # ------------------------------------------------------------------
    def _calc(self, f: FuncVec):
        if not isinstance(f.getX(), StepGrid):  # todo
            raise ValueError(DerivPts9.log.error("DerivPts9 can only work with StepGrid"))
        if f.size() < DerivPts9.MIN_GRID_SIZE:
            raise ValueError(DerivPts9.log.error("DerivPts9 needs at least 9 grid points"))

        grid = f.getX()
        h2 = 0.5 / (20160.0 * grid.getGridStep())  # 1/(2·8!)·1/h
        self._calc_h(h2, f)

    # ------------------------------------------------------------------
    def _calc_h(self, h2: float, f: FuncVec):
        max_size = f.size()
        k = 0
        self.set(k, h2 * self._calc_pts(k, 0, f));
        k += 1
        self.set(k, h2 * self._calc_pts(k, 0, f));
        k += 1
        self.set(k, h2 * self._calc_pts(k, 0, f));
        k += 1
        self.set(k, h2 * self._calc_pts(k, 0, f));
        k += 1

        for i in range(0, max_size - 2 * k):
            self.set(i + k, h2 * self._calc_pts(k, i, f))

        i = max_size - 1 - DerivPts9._SIZE_1
        k += 1
        self.set(i + k, h2 * self._calc_pts(k, i, f));
        k += 1
        self.set(i + k, h2 * self._calc_pts(k, i, f));
        k += 1
        self.set(i + k, h2 * self._calc_pts(k, i, f));
        k += 1
        self.set(i + k, h2 * self._calc_pts(k, i, f))

    # ==============================
    #   First derivative, error O(h⁹)
    # ==============================
    def _calc_pts(self, idx: int, i: int, f: FuncVec) -> float:
        res = 0.0
        k = 0
        coeff = DerivPts9._coeff

        # left half (add)
        res += coeff[idx, k] * f.get(i);
        k += 1;
        i += 1  # 0
        res += coeff[idx, k] * f.get(i);
        k += 1;
        i += 1  # 1
        res += coeff[idx, k] * f.get(i);
        k += 1;
        i += 1  # 2
        res += coeff[idx, k] * f.get(i);
        k += 1;
        i += 1  # 3
        res += coeff[idx, k] * f.get(i);
        i += 1  # 4

        # right half (subtract, mirrored row)
        idx_mirror = DerivPts9._mid + DerivPts9._mid - idx
        k -= 1
        res -= coeff[idx_mirror, k] * f.get(i);
        k -= 1;
        i += 1  # 5
        res -= coeff[idx_mirror, k] * f.get(i);
        k -= 1;
        i += 1  # 6
        res -= coeff[idx_mirror, k] * f.get(i);
        k -= 1;
        i += 1  # 7
        res -= coeff[idx_mirror, k] * f.get(i)  # 8
        return res

    # ------------------------------------------------------------------
    #  Public helper: full first-derivative operator  (N × N matrix)
    # ------------------------------------------------------------------
    # @staticmethod
    # def diff_matrix(x: "StepGrid") -> np.ndarray:
    #     """
    #     Return the O(h⁹) first-derivative matrix **D** that acts on any
    #     vector of values sampled on the *same* StepGrid ``x``.
    #
    #     After calling
    #     # >>> D = DerivPts9.diff_matrix(grid)
    #     you can obtain the derivative of *any* function `f` defined on
    #     that grid simply with
    #     # >>> f1 = D @ f          # matrix–vector product (numpy '@')
    #
    #     The matrix is built by feeding a set of unit vectors through the
    #     existing DerivPts9 stencil, so it is 100 % consistent with
    #     ``DerivPts9(FuncVec)`` you already use.
    #     """
    #     N = x.size()
    #     D = np.zeros((N, N), dtype=float)
    #
    #     # --- a tiny helper to create a "delta-vector" on the given grid
    #     def _delta(idx: int) -> FuncVec:
    #         v = FuncVec(x)  # zero everywhere …
    #         v.set(idx, 1.0)  # … except at idx
    #         return v
    #
    #     # build the matrix column-by-column
    #     for j in range(N):
    #         col = DerivPts9(_delta(j))  # derivative of δ_j
    #         D[:, j] = col.getArr()  # copy into j-th column
    #
    #     # ----------------------------------------------------------------------
    #     #   ⬇️ CONSTANT-FUNCTION CORRECTION ⬇️
    #     # Each row of D *should* sum to zero, because the derivative of a
    #     # constant must be zero.  Numerical round-off leaves a drift of order ε.
    #     # Subtract that tiny drift so that D·[1,1,…,1] = 0 exactly.
    #     # drift = D.sum(axis=1, keepdims=True)  # shape (N,1)
    #     # print("mD :", str(MtrxDbgView(Mtrx(data=drift))))
    #     # print("mD :", str(Mtrx(data=drift)))
    #     # D -= drift / N
    #     # ----------------------------------------------------------------------
    #
    #     # D = DerivPts9.calibrate_first_derivative(D)
    #     res = Mtrx(data=D)
    #     return res




# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    grid = StepGrid(0.0, 2.0, 101)  # uniform grid
    f = np.sin(grid.getArr())  # any sampled function
    fv = FuncVec(grid, f)
    print("fv :", FuncVecDbgView(fv))
    # D = DerivPts9.diff_matrix(grid)  # build once, reuse many times
    from qm_math.func.deriv.DerivFactory import DerivFactory
    D = DerivFactory.calcDerivMtrxPts9(grid)  # build once, reuse many times
    mD = Mtrx(data=D)
    print("mD :", str(MtrxDbgView(mD)))
    # f1 = D @ f  # high–order derivative
    fv1 = mD.multVec(fv)
    print("fv1 :", fv1.toCSV())
    # print("fv1 :", FuncVecDbgView(fv1))
    print("fv1 :", VecDbgView(fv1))
    df = DerivPts9(fv)  # ≈ 4x³
    print("df :", df.toCSV())
    print("df :", FuncVecDbgView(df))
    max_err = DistMaxAbsErr.distSLOW(df.getY(), fv1)
    print('max_err = ', max_err)
    assert max_err < 1e-10

    # Example: f(x) = x⁴ on [0,1] with 13 points
    grid = StepGrid(first=0.0, last=3.0, size=101)
    fv = FuncVec(grid, FuncPowAbsInt(1.0, 4))  # x⁴
    df = DerivPts9(fv)  # ≈ 4x³
    # D = DerivPts9.diff_matrix(grid)  # build once, reuse many times
    D = DerivFactory.calcDerivMtrxPts9(grid)  # build once, reuse many times
    mD = Mtrx(data=D)
    print("mD :", str(MtrxDbgView(mD)))
    fv1 = mD.multVec(fv)
    print("fv1 :", fv1.toCSV())
    max_err = DistMaxAbsErr.distSLOW(df.getY(), fv1)
    print('max_err = ', max_err)
    assert max_err < 1e-10

    print("x :", grid.toCSV())
    print("f :", fv.toCSV())
    print("df:", df.toCSV())
