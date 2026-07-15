# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLagrr_1.py Created : 2025-06-29 at 3:29 pm by Dmitry.A.Konovalov@gmail.com
# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLagrr_1.py           (port of math.func.polynom.laguerre.FuncLagrr_1)
# -----------------------------------------------------------------------------
"""
Straight-port of the tiny Java class

    public class FuncLagrr_1 implements Func {
        double alpha, lambda;
        double calc(double x) { return alpha + 1 - lambda * x; }
    }

* **All public method names are kept unchanged** so existing call-sites continue
  to work.
* Uses exactly the same constructor signature `(alpha, lambda)`.
"""

from __future__ import annotations

from qm_math.func.Func import Func   # ← assumed to exist elsewhere


class FuncLagrr_1(Func):
    """
    Scalar function  *f(x) = α + 1 − λ·x*   (Laguerre polynomial L₁).
    """

    # ----------------------------- construction --------------------------
    def __init__(self, alpha: float, lambda_: float) -> None:
        self._alpha = float(alpha)
        self._lambda = float(lambda_)

    # ----------------------------- Func API -----------------------------
    def calc(self, x: float) -> float:               # exactly the same name
        # todo: 250810: only defined on x>=0 !!!!!!!
        # return self._alpha + 1.0 - self._lambda * x
        return self._alpha + 1.0 - self._lambda * abs(x)
