# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LgrrOrth.py Created : 2025-06-29 at 3:30 pm by Dmitry.A.Konovalov@gmail.com
# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LgrrOrth.py      (port of math.func.polynom.laguerre.LgrrOrth)
# ------------------------------------------------------------------------------
"""
**LgrrOrth** – an orthonormal set of Laguerre‐type radial basis functions.

Exactly mirrors the public API of the original Java class:

* constructor `LgrrOrth(r, size, alpha, lambda_)`
* `getFromNonOrth()` – scale factors that turn the *non-orthonormal*
  polynomials produced by :pyclass:`~LgrrArr` into an orthonormal basis.

All helper classes (`Vec`, `Func`, `FuncGamma`, `LgrrArr`, …) are expected to
exist elsewhere in the code-base; this file contains only the translated logic.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

# --- external types (same import paths as Java --------------------------------
from qm_math.vec.Vec import Vec
from qm_math.func.Func import Func
from qm_math.func.FuncGamma import FuncGamma
from qm_math.func.polynom.laguerre.LgrrArr import LgrrArr


# -----------------------------------------------------------------------------


class _ThisNormFunc_LgrrOrth(Func):
    """
    Point-wise weight  exp(-λr/2) · (λr)^{α/2}

    Defined as a *private* helper to replicate Java’s inner class
    `ThisNormFunc`.
    """

    def __init__(self, alpha: float, lambda_: float) -> None:
        self._alpha = alpha
        self._lambda = lambda_

    def calc(self, r):
        r = np.abs(r)  # only defined on r>=0
        u = self._lambda * r
        if u == 0.0:
            # print(f"u={u}, r={r}")
            return 0.0
        res = math.exp(-0.5 * u + 0.5 * self._alpha * math.log(u))
        # print(f"u={u}, r={r}, f(u)={res}")
        return res


# -----------------------------------------------------------------------------


class LgrrOrth(LgrrArr):
    """
    Orthonormal radial Laguerre basis (Abramowitz & Stegun, p 579–580).

    **Important:** integration weight is *dr* (not r² dr).  After
    construction the set satisfies::

        ∫₀^∞  Lₙ(r) · Lₙ′(r) dr  =  δₙₙ′
    """

    # ------------------------------------------------------------------
    def __init__(self, r: Vec, size: int, alpha: float, lambda_: float) -> None:
        super().__init__(r, size, alpha, lambda_)  # build non-orth set

        # --- multiply every polynomial by the common weight function ----
        weight = _ThisNormFunc_LgrrOrth(alpha, lambda_)
        # FuncArr has `.mult(Func)` in the Java code; assume same here
        # If your Python `FuncArr` uses another method name adjust accordingly.
        self.mult(weight)

        # --- convert to orthonormal form --------------------------------
        self._from_non_orth: List[float] = []
        if alpha <= 1.0e-16:
            self._norm_zero_alpha()
        else:
            self._norm_general(alpha, lambda_)

    # ------------------------------------------------------------------
    # Java: public double[] getFromNonOrth()
    def getFromNonOrth(self) -> List[float]:
        """Return the list of scaling factors applied to each *n*."""
        return self._from_non_orth

    # ------------------------------------------------------------------
    # ------------------------- helpers ---------------------------------
    def _norm_general(self, alpha: float, lambda_: float) -> None:
        """General α > 0 branch — Eq.(…) from original sources."""
        gamma = FuncGamma()  # Java: new FuncGamma()
        for n in range(self.size()):
            norm_n = 1.0 / math.sqrt(
                gamma.calc(alpha + n + 1) / gamma.calc(n + 1) / lambda_
            )
            self._from_non_orth.append(norm_n)
            self.get(n).mult(norm_n)  # scale FuncVec in-place

    # ------------------------------------------------------------------
    def _norm_zero_alpha(self) -> None:
        """Special case α ≈ 0 (original `normZeroAlpha()` logic)."""
        norm_n = math.sqrt(self.getLambda())  # λ½
        self._from_non_orth = [norm_n] * self.size()
        for n in range(self.size()):
            self.get(n).mult(norm_n)
