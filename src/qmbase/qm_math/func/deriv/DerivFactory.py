# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : DerivFactory.py Created : 2025-07-03 at 9:33 am by Dmitry.A.Konovalov@gmail.com
# qm_math/func/deriv/DerivFactory.py
# Copyright dmitry.konovalov@jcu.edu.au Date: 9/07/2008, Time: 16:23:46
from __future__ import annotations
import numpy as np
from numpy.linalg import solve


# ------------------------------------------------------------
# helper: coefficients for 1st derivative on arbitrary offsets
# ------------------------------------------------------------
def _coeff_first_derivative_v2(offsets):
    # Return an array c such that
    #     f'(x0) ≈ Σ c_k f(x0 + offset_k*h)
    # accurate through order len(offsets)-1.
    m = len(offsets)
    A = np.array([[off ** p for off in offsets] for p in range(m)], dtype=float)
    b = np.zeros(m);
    b[1] = 1.0  # match the first derivative
    return solve(A, b)

# ------------------------------------------------------------
# helper: coefficients for d/dx on arbitrary offsets
# ------------------------------------------------------------
def _coeff_first_derivative_v3(offsets):
    """
    Return an array c such that  f'(x0) ≈ Σ c_k f(x0 + offsets_k*h)
    matching derivatives up to order len(offsets)-1 (i.e. O(h^{m-1})).
    """
    m = len(offsets)
    A = np.array([[o ** p for o in offsets] for p in range(m)], dtype=float)
    b = np.zeros(m);
    b[1] = 1.0
    return solve(A, b)  # length-m array




class DerivFactory:
    @staticmethod
    def makeDeriv(f):        # def makeDeriv(f: FuncVec) -> FuncVec:
        """
        public static FuncVec makeDeriv(FuncVec f)
        Choose derivative stencil by f.size():
          - 9-point if size >= 9
          - 5-point if size >= 5
          - 3-point if size >= 3
        Returns None if size < 3.
        """
        res = None
        if f.size() >= 9:
            # todo!!! DerivFactory is used in FuncVec. that is why local import of DerivPts9
            from qm_math.func.deriv.DerivPts9 import DerivPts9
            # res = DerivPts9(f)
            res = DerivPts9(f).getY()  # try forcing to Vec
        elif f.size() >= 5:
            from qm_math.func.deriv.DerivPts5 import DerivPts5
            res = DerivPts5(f).getY()
        elif f.size() >= 3:
            from qm_math.func.deriv.DerivPts3 import DerivPts3
            res = DerivPts3(f).getY()
        return res

    @staticmethod
    def calcDerivMtrxPts3(x: "StepGrid"):
        from qm_math.func.deriv.DerivPts3 import DerivPts3
        return DerivFactory.calc_deriv_matrix_from_any(x=x, class_name=DerivPts3)

    @staticmethod
    def calcDerivMtrxPts5(x: "StepGrid"):
        from qm_math.func.deriv.DerivPts5 import DerivPts5
        return DerivFactory.calc_deriv_matrix_from_any(x=x, class_name=DerivPts5)

    @staticmethod
    def calcDerivMtrxPts9(x: "StepGrid"):
        # NOTE!! This uses my old deriv. and calculates matrix
        from qm_math.func.deriv.DerivPts9 import DerivPts9
        return DerivFactory.calc_deriv_matrix_from_any(x=x, class_name=DerivPts9)


    @staticmethod
    def calc_deriv_matrix_from_any(x: "StepGrid", class_name):
        """
        Return the O(h⁹) first-derivative matrix **D** that acts on any
        vector of values sampled on the *same* StepGrid ``x``.

        After calling
        # >>> D = DerivPts9.diff_matrix(grid)
        you can obtain the derivative of *any* function `f` defined on
        that grid simply with
        # >>> f1 = D @ f          # matrix–vector product (numpy '@')

        The matrix is built by feeding a set of unit vectors through the
        existing DerivPts9 stencil, so it is 100 % consistent with
        ``DerivPts9(FuncVec)`` you already use.
        """
        N = x.size()
        D = np.zeros((N, N), dtype=float)

        # --- a tiny helper to create a "delta-vector" on the given grid
        from qm_math.func.FuncVec import FuncVec
        def _delta(idx: int) -> FuncVec:
            v = FuncVec(x)  # zero everywhere …
            v.set(idx, 1.0)  # … except at idx
            return v

        # build the matrix column-by-column
        for j in range(N):
            # col = DerivPts9(_delta(j))  # derivative of δ_j
            col = class_name(_delta(j))  # derivative of δ_j
            D[:, j] = col.getArr()  # copy into j-th column

        # ----------------------------------------------------------------------
        #   ⬇️ CONSTANT-FUNCTION CORRECTION ⬇️
        # Each row of D *should* sum to zero, because the derivative of a
        # constant must be zero.  Numerical round-off leaves a drift of order ε.
        # Subtract that tiny drift so that D·[1,1,…,1] = 0 exactly.
        # drift = D.sum(axis=1, keepdims=True)  # shape (N,1)
        # print("mD :", str(MtrxDbgView(Mtrx(data=drift))))
        # print("mD :", str(Mtrx(data=drift)))
        # D -= drift / N
        # ----------------------------------------------------------------------

        # D = DerivPts9.calibrate_first_derivative(D)
        from qm_math.mtrx.api.Mtrx import Mtrx
        res = Mtrx(data=D)
        return res

    @staticmethod
    def calibrate_first_derivative(D: np.ndarray) -> None:
        # todo: NOT TESTED
        # In-place calibration:  make sure the derivative of a constant is exactly 0.
        # Given an N×N first-derivative matrix D (rows are the stencils),
        # we compute d = D @ 1  (i.e. the numeric derivative of the constant 1).
        # Ideally d == 0, but finite precision leaves ~1e-14.             ───────────
        # We subtract that small drift from every column,        D ← D − d·1ᵀ / N,
        # so afterwards D @ 1 == 0 (to machine epsilon).
        N = D.shape[0]  # grid size
        if D.shape[1] != N:
            raise ValueError("D must be square")

        drift = D.sum(axis=1, keepdims=True)  # same as D @ np.ones(N)
        D -= drift / N  # rank-1 correction in-place
        return D

    @staticmethod
    def make_deriv_mtrx_np(x):
        # return make_first_derivative_matrix_full_3pt(x)
        return DerivFactory.make_deriv_mtrx_5pt(x)

    @staticmethod
    def makeDerivMtrxV2(x):
        mtrx = DerivFactory.make_deriv_mtrx_np(x)
        from qm_math.mtrx.api.Mtrx import Mtrx
        res = Mtrx(data=mtrx)
        return res

    import numpy as np

    # --- helper: build coefficients for any forward/backward 1-st derivative ---
    @staticmethod
    def _fd_coefficients_first_derivative(offsets):
        """
        Return a dict {offset: coeff} s.t.
            f'(x0) ≈ Σ coeff_k * f(x0 + offset_k*h)  (O(h^{m-1}))
        where 'offsets' is a list/array of integer grid offsets.
        """
        m = len(offsets)
        A = np.zeros((m, m))
        b = np.zeros(m)
        b[1] = 1.0  # match first derivative
        for p in range(m):  # match 0-th,2-nd,…,m-1-th moments
            A[p] = np.array(offsets) ** p
        return dict(zip(offsets, np.linalg.solve(A, b)))

    # ---   MAIN 7-POINT 1st-derivative constructor   ---------------------------
    @staticmethod
    def make_deriv_mtrx_7pt(x):
        # 7-point sixth-order first-derivative matrix (shape (N,N)),
        # including forward/backward rows for i = 0,1,2 and N-3,N-2,N-1.
        # x : 1-D numpy array (uniform grid)
        # Returns
        # D1 : (N,N) ndarray   finite-difference matrix for d/dx
        N = len(x)
        h = x[1] - x[0]
        D1 = np.zeros((N, N))

        # ---- 1. interior centred stencil  (i = 3 … N-4) ----------------------
        c = 1.0 / (60.0 * h)
        for i in range(3, N - 3):
            D1[i, i - 3] = 1 * c
            D1[i, i - 2] = -9 * c
            D1[i, i - 1] = 45 * c
            D1[i, i + 1] = -45 * c
            D1[i, i + 2] = 9 * c
            D1[i, i + 3] = -1 * c
            # D1[i,i] stays 0

        # ---- 2. forward rows  i = 0,1,2  ------------------------------------
        fwd0 = np.array([-147, 360, -450, 400, -225, 72, -10]) / (60 * h)
        fwd1 = np.array([-10, -77, 150, -100, 50, -15, 2]) / (60 * h)
        fwd2 = np.array([2, -24, -35, 80, -30, 8, -1]) / (60 * h)
        D1[0, 0:7] = fwd0
        D1[1, 0:7] = fwd1
        D1[2, 0:7] = fwd2

        # ---- 3. backward rows  i = N-3,N-2,N-1  (mirror of forward) ----------
        D1[-1] = -D1[0, ::-1]  # i = N-1
        D1[-2] = -D1[1, ::-1]  # i = N-2
        D1[-3] = -D1[2, ::-1]  # i = N-3
        return D1

    @staticmethod
    def make_deriv_mtrx_5pt(x):
        """
        Creates a 5-point first derivative matrix D1 (O(h^4) accuracy),
        including accurate forward and backward stencils at the boundaries.
        Parameters:
            x (np.ndarray): Grid points, uniformly spaced
        Returns:
            D1 (np.ndarray): First derivative matrix of shape (len(x), len(x))
        """
        N = len(x)
        h = x[1] - x[0]
        D1 = np.zeros((N, N))

        # --- Forward difference at x[0] and x[1] (O(h⁴)) ---
        # Derived using Taylor expansion
        D1[0, 0:5] = [-25, 48, -36, 16, -3]
        D1[1, 0:5] = [-3, -10, 18, -6, 1]
        D1[0:2, 0:5] /= (12 * h)

        # --- Central difference for interior (O(h⁴)) ---
        for i in range(2, N - 2):
            D1[i, i - 2] = 1
            D1[i, i - 1] = -8
            D1[i, i + 1] = 8
            D1[i, i + 2] = -1
        D1[2:N - 2] /= (12 * h)

        # --- Backward difference at x[-2] and x[-1] (O(h⁴)) ---
        # Again, derived from Taylor expansion
        D1[N - 2, N - 5:N] = [3, -16, 36, -48, 25]
        D1[N - 1, N - 5:N] = [-1, 6, -18, 10, 3]
        D1[N - 2:, N - 5:N] /= (12 * h)
        return D1

    @staticmethod
    def make_deriv_mtrx_3pt(x):
        """
        Creates a first derivative matrix D1 (with boundary-aware finite difference stencils)
        for equally spaced grid x (length nx + 4).
        Returns:
            np.ndarray: First derivative matrix D1 of shape (len(x), len(x)).
        """
        N = len(x)
        h = x[1] - x[0]  # uniform spacing

        D1 = np.zeros((N, N))

        # Forward difference at first two points
        D1[0, 0] = -3 / (2 * h)
        D1[0, 1] = 4 / (2 * h)
        D1[0, 2] = -1 / (2 * h)

        D1[1, 1] = -3 / (2 * h)
        D1[1, 2] = 4 / (2 * h)
        D1[1, 3] = -1 / (2 * h)

        # Central difference in interior
        for i in range(2, N - 2):
            D1[i, i - 1] = -1 / (2 * h)
            D1[i, i + 1] = 1 / (2 * h)

        # Backward difference at last two points
        D1[N - 2, N - 4] = 1 / (2 * h)
        D1[N - 2, N - 3] = -4 / (2 * h)
        D1[N - 2, N - 2] = 3 / (2 * h)

        D1[N - 1, N - 3] = 1 / (2 * h)
        D1[N - 1, N - 2] = -4 / (2 * h)
        D1[N - 1, N - 1] = 3 / (2 * h)
        # # Example:
        # x = np.linspace(0, 1, 10)  # grid with 10 points
        # D1 = make_first_derivative_matrix_full(x)
        # print(np.round(D1, 2))  # rounded for easier reading
        return D1

    # ------------------------------------------------------------
    # 9-point first-derivative matrix (centre+forward+backward)
    # ------------------------------------------------------------
    @staticmethod
    def make_deriv_mtrx_9pt(x):
        """
        Constructs a 9-point (8-th-order centred, 6-th-order edges) first-derivative
        matrix for a uniform grid x.

        Parameters
        ----------
        x : 1-D ndarray, uniform grid.

        Returns
        -------
        D1 : (N,N) ndarray   finite-difference d/dx matrix.
        """
        N = len(x)
        h = x[1] - x[0]
        D1 = np.zeros((N, N), dtype=float)

        # --- 1. interior centred rows -----------------------------------------
        offsets_c = np.arange(-4, 5)  # -4 … +4
        coeff_c = _coeff_first_derivative_v2(offsets_c) / h
        for i in range(4, N - 4):
            D1[i, i - 4:i + 5] = coeff_c

        # --- 2. forward rows  i = 0,1,2,3 --------------------------------------
        for row in range(4):
            offs = np.arange(-row, 9 - row)  # e.g. 0..8  -1..7  …
            coeff = _coeff_first_derivative_v2(offs) / h
            D1[row, offs + row] = coeff  # shift to matrix columns

        # --- 3. backward rows  i = N-4..N-1  (mirror of forward) --------------
        for k in range(4):  # k = 0..3
            row = N - 1 - k
            offs = -np.arange(-k, 9 - k)  # mirror of forward offsets
            coeff = -_coeff_first_derivative_v2(-offs) / h  # antisymmetry
            D1[row, row + offs] = coeff

        return D1

    # ------------------------------------------------------------
    # 11-point first-derivative matrix
    # ------------------------------------------------------------
    @staticmethod
    def make_deriv_mtrx_11pt(x):
        """
        11-point (10-th-order centred, 6-th-order edges) first-derivative matrix
        for a uniform grid.

        Parameters
        ----------
        x : 1-D ndarray (uniform grid)

        Returns
        -------
        D1 : (N,N) ndarray   finite-difference d/dx operator
        """
        N = len(x)
        h = x[1] - x[0]
        D1 = np.zeros((N, N), dtype=float)

        # ---- 1. centred interior rows (i = 5 … N-6) --------------------------
        offs_c = np.arange(-5, 6)  # −5 … +5
        coeff_c = _coeff_first_derivative_v3(offs_c) / h
        for i in range(5, N - 5):
            D1[i, i - 5:i + 6] = coeff_c

        # ---- 2. forward rows  i = 0…4 ---------------------------------------
        for row in range(5):
            offs = np.arange(-row, 11 - row)  # e.g. 0..10, −1..9, …
            coeff = _coeff_first_derivative_v3(offs) / h
            D1[row, offs + row] = coeff  # shift to correct columns

        # ---- 3. backward rows  i = N-5 … N-1  (mirror of forward) -----------
        for k in range(5):
            row = N - 1 - k
            offs = -np.arange(-k, 11 - k)  # mirror offsets
            coeff = -_coeff_first_derivative_v3(-offs) / h  # antisymmetry
            D1[row, row + offs] = coeff

        return D1










