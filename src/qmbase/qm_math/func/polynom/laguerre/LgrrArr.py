# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LgrrArr.py Created : 2025-06-29 at 3:30 pm by Dmitry.A.Konovalov@gmail.com
# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LgrrArr.py    (port of math.func.polynom.laguerre.LgrrArr)
# -----------------------------------------------------------------------------
"""
Array/container of generalised (associated) Laguerre polynomials **Lₙ
(α, λ·x)** stored as `FuncVec`s, exactly mirroring the original Java
`LgrrArr` public API.

Dependencies (`FuncArr`, `FuncVec`, `Vec`, `FuncConst`, `FuncLagrr_1`,
`FactLn`) are assumed to exist elsewhere in the code-base — this file
*just* ports the logic.
"""

from __future__ import annotations

import numpy as np

from _new25.dbg import dbg
from javax.utilx.log.Log import Log
# --- external types (import-paths match the original package layout) ----
from qm_math.func.arr.FuncArr import FuncArr
from qm_math.func.FuncVec import FuncVec
from qm_math.func.simple.FuncConst import FuncConst
from qm_math.func.polynom.laguerre.FuncLagrr_1 import FuncLagrr_1
from qm_math.func.FactLn import FactLn
from qm_math.vec.Vec import Vec

# ------------------------------------------------------------------------
log = Log.getLog('LgrrArr')


class LgrrArr(FuncArr):
    """
    Container of Laguerre polynomials  *L₀, L₁, …, Lₙ₋₁*  on a common grid.

    Parameters
    ----------
    x : Vec
        Grid on which every polynomial will be sampled.
    size : int
        How many consecutive polynomials to generate (n ≥ 0).
    alpha : float
        Generalised Laguerre parameter α.
    lambda_ : float
        Scaling factor λ — note that Java used the bare name ``lambda`` which
        is a keyword in Python, so the constructor takes ``lambda_`` while the
        public getter remains `getLambda()` for full API compatibility.
    """

    # ---------------------------- construction -------------------------
    def __init__(self, x: Vec, size: int, alpha: float, lambda_: float) -> None:
        super().__init__(x, size)                            # Java super(x,size)
        self.alpha: float = alpha
        self._lambda: float = lambda_                        # keep internal name
        LgrrArr._fLn = FactLn.getInstance()                  # static single-ton
        self._load()                                         # build all Lₙ

    # ---------------------------- public API ---------------------------
    def getLambda(self) -> float:
        """Return the λ parameter (method name kept unchanged)."""
        return self._lambda
    def getAlpha(self) -> float:
        return self.alpha

    # ---------------------------- helpers ------------------------------
    def _load(self) -> None:
        """Generate and cache the polynomial samples (line-for-line port)."""
        if self.size() == 0:
            return

        # Java: set(0, new FuncVec(getX(), new FuncConst(1.)));
        self.set(0, FuncVec(self.getX(), FuncConst(1.0)))
        n = 0
        log.dbg(f" n={n} =", self.get(n))
        if self.size() == 1:
            return

        # L₁(α, λx) = α+1 − λx
        self.set(1, FuncVec(self.getX(), FuncLagrr_1(self.alpha, self._lambda)))
        if self.size() == 2:
            return

        xa = self.getX().getArr()            # raw NumPy view of grid values
        # For every grid point evolve the three-term recurrence:
        #   n Lₙ  = ((2n+α−1) − λx) Lₙ₋₁  −  (n−1+α) Lₙ₋₂
        # (ref: Russian Abramowitz & Stegun, p 588)
        for i, x_val in enumerate(xa):
            x_val = np.abs(x_val)    # 250810 todo: <--- only defined on x>=0 !!!!!!!!!!!!!!!
            x_scaled = self._lambda * x_val   # x = λ x   (same as Java)
            n = 0
            L_n_2 = self.get(n).get(i)        # L₀
            n += 1
            L_n_1 = self.get(n).get(i)        # L₁
            n += 1
            while n < self.size():  # ?
                L_n = (
                    ((2.0 * n + self.alpha - 1.0) - x_scaled) * L_n_1
                    - (n - 1.0 + self.alpha) * L_n_2
                ) / n
                self.get(n).set(i, L_n)       # write into the n-th FuncVec
                # shift window for next n
                L_n_2, L_n_1 = L_n_1, L_n
                n += 1
