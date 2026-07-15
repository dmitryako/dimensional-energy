# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Mathx.py Created : 2025-06-26 at 1:31 pm by Dmitry.A.Konovalov@gmail.com

# mathx.py  ––  Port of math.Mathx
# (Java original: 11 Jul 2008, 09 : 44 : 18)
#
# • All static fields / methods are reproduced as @staticmethods inside the
#   Mathx class; call them exactly like in Java, e.g.   Mathx.factLn(10)
# • Java’s BigDecimal routines are marked “_todo”; a quick Decimal-based
#   implementation is included but feel free to refine or remove.
# -------------------------------------------------------------------------

from __future__ import annotations
import math
from decimal import Decimal, getcontext

from qm_math.func.FactLn import FactLn


# from fact_ln import FactLn
math_zero_eps = 1e-32

class Mathx:
    # ---------------------------------------------------------------------
    _fact: FactLn = FactLn.getInstance()

    # -------- basic helpers ----------------------------------------------
    @staticmethod
    def mod(a: int, b: int) -> int:
        """Fortran-style remainder (same as Python % for ints)."""
        return a % b

    # -------- factorial utilities ----------------------------------------
    @staticmethod
    def factLn(n: int) -> float:
        return Mathx._fact.calc(n)

    @staticmethod
    def factorial(n: int) -> float:
        return math.exp(Mathx.factLn(n))

    # n!/(n-k)!
    @staticmethod
    def factLn2(n: float, k: int) -> float:
        if n < 2 or n < k:
            return 0.0
        return sum(math.log(i) for i in range(int(n - k + 1), int(n) + 1))

    # n!/[(n-k)! n^k]
    @staticmethod
    def factLn3(n: float, k: int) -> float:
        if n < 2 or n < k:
            return 0.0
        return sum(math.log(i / n) for i in range(int(n - k + 1), int(n) + 1))

    @staticmethod
    def fact2(n: float, k: int) -> float:
        return math.exp(Mathx.factLn2(n, k))

    # -------- power shortcut ---------------------------------------------
    @staticmethod
    def pow(x: float, k: int) -> float:
        EPS = 1e-20
        if k == 0:
            return 1.0
        abs_k = abs(k)
        if abs_k <= 10:
            res = x
            for _ in range(abs_k - 1):
                res *= x
            if Mathx._isZero(res):
                # return math_zero_eps  # todo?
                # return 0.0
                res = EPS
                # return EPS
            return res if k > 0 else 1.0 / res
        # fallback to math.pow for larger exponents
        return math.pow(x, k)

    @staticmethod
    def _isZero(v: float) -> bool:
        return v == 0.0

    # -------- max / min convenience --------------------------------------
    @staticmethod
    def max(a: int, b: int, c: int, d: int | None = None) -> int:
        return max(a, b, c) if d is None else max(a, b, c, d)

    @staticmethod
    def min(a: int, b: int, c: int, d: int | None = None) -> int:
        return min(a, b, c) if d is None else min(a, b, c, d)

    @staticmethod
    def max_byte(a: int, b: int) -> int:  # bytes in Java are ints in Python
        return a if a > b else b

    @staticmethod
    def min_byte(a: int, b: int) -> int:
        return a if a < b else b

    # -------- delta / Kronecker-delta ------------------------------------
    @staticmethod
    def delta(a: object, b: object) -> int:
        return 1 if a == b else 0

    @staticmethod
    def dlt(i: int, j: int) -> int:
        return 1 if i == j else 0

    # -------- binomial coefficients --------------------------------------
    @staticmethod
    def binomialCoeffLn(n: int, k: int) -> float:
        return Mathx.factLn(n) - Mathx.factLn(n - k) - Mathx.factLn(k)

    @staticmethod
    def binomialCoeff(n: int, k: int) -> float:
        return math.exp(Mathx.binomialCoeffLn(n, k))

    # -------- range limit / step -----------------------------------------
    @staticmethod
    def limit(v: int, minVal: int, maxVal: int) -> int:
        return max(minVal, min(v, maxVal))

    @staticmethod
    def step(i: int) -> int:
        return 0 if i <= 0 else 1

    # -------- slow series expansions -------------------------------------
    # todo: NOTE! if you want the Java behaviour exactly, make Mathx.expSLOW return the same_bits value as math.exp:
    # def expSLOW(x: float) -> float:
    #     res = 0.0
    #     v   = 1.0
    #     t   = 1.0
    #     i   = 1
    #     while True:
    #         res_old = res
    #         res = v
    #         t *= x / i
    #         v += t
    #         if v == res:          # no further change
    #             return res        # <<—  return res, not v
    #         i += 1
    @staticmethod
    def expSLOW(x: float) -> float:
        res = 0.0
        v = 1.0
        t = 1.0
        i = 1
        while v != res:
            res = v
            t *= x / i
            v += t
            i += 1
        return res

    @staticmethod
    def expOneXSLOW(x: float) -> float:
        res = 0.0
        v = 1.0
        t = 1.0
        i = 1
        while v != res:
            res = v
            t *= x / (i + 1)
            v += t
            i += 1
        return res

    # ---- Decimal (BigDecimal) versions – basic illustrative port ---------
    @staticmethod
    def expSLOW_todo(x: Decimal) -> Decimal:
        getcontext().prec += 10  # increase precision temporarily
        res = Decimal(0)
        v = Decimal(1)
        t = Decimal(1)
        i = 1
        while v != res:
            res = v
            t = t * (x / Decimal(i))
            v = v + t
            i += 1
        getcontext().prec -= 10
        return res

    @staticmethod
    def expOneXSLOW_todo(x: Decimal) -> Decimal:
        getcontext().prec += 10
        res = Decimal(0)
        v = Decimal(1)
        t = Decimal(1)
        i = 1
        while v != res:
            res = v
            t = t * (x / Decimal(i + 1))
            v = v + t
            i += 1
        getcontext().prec -= 10
        return res

if __name__ == "__main__":
    print("5! =", Mathx.factorial(5))
    print("pow shortcut 2^8 =", Mathx.pow(2, 8))
    print("binom(10,3) =", Mathx.binomialCoeff(10, 3))
    print("limit(15,0,10) =", Mathx.limit(15, 0, 10))

