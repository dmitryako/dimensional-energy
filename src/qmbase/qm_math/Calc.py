# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Calc.py Created : 2025-06-26 at 1:29 pm by Dmitry.A.Konovalov@gmail.com

# calc.py  ––  Port of math.Calc
# (Java original: 10 Jul 2008, 10 : 38 : 19)
#
# • All constant names and method names match the Java source exactly.
# • Constants are class attributes.  The “private” IGNORE is prefixed “_”.
# • Uses Python’s built-in math module; no external dependencies.
# -------------------------------------------------------------------------

import math


class Calc:
    # ---------- public EPS constants -------------------------------------
    EPS_2 = 1e-2
    EPS_3 = 1e-3
    EPS_4 = 1e-4
    EPS_5 = 1e-5
    EPS_6 = 1e-6
    EPS_7 = 1e-7
    EPS_8 = 1e-8
    EPS_9 = 1e-9
    EPS_10 = 1e-10
    EPS_11 = 1e-11
    EPS_12 = 1e-12
    EPS_13 = 1e-13
    EPS_14 = 1e-14
    EPS_15 = 1e-15
    EPS_16 = 1e-16
    EPS_17 = 1e-17
    EPS_18 = 1e-18
    EPS_32 = 1e-32
    EPS_100 = 1e-100

    # ---------- private constant -----------------------------------------
    _IGNORE = 1e-32

    # ---------- static utility methods -----------------------------------
    @staticmethod
    def isZero(v: float) -> bool:
        """Return True if |v| is smaller than the IGNORE threshold."""
        return abs(v) < Calc._IGNORE

    @staticmethod
    def prty(x: int) -> int:
        """Parity: returns (-1)^x (1 for even, -1 for odd)."""
        return 1 if x % 2 == 0 else -1

    @staticmethod
    def parity(x: int) -> int:
        """Parity: returns (-1)^x (1 for even, -1 for odd)."""
        return 1 if x % 2 == 0 else -1

    @staticmethod
    def cosFromTan(tanX: float) -> float:
        """Compute cos(x) given tan(x) using identity  cos = 1 / √(1 + tan²)."""
        return math.sqrt(1.0 / (1.0 + tanX * tanX))

    @staticmethod
    def sinFromCos(cosX: float) -> float:
        """Compute sin(x) given cos(x) using identity  sin = √(1 - cos²)."""
        return math.sqrt(max(0.0, 1.0 - cosX * cosX))


if __name__ == "__main__":
    print(Calc.isZero(1e-40))  # True
    print(Calc.prty(3))  # -1
    tan45 = 1.0
    print(Calc.cosFromTan(tan45))  # ≈ 0.707106...
    cos30 = math.sqrt(3) / 2
    print(Calc.sinFromCos(cos30))  # ≈ 0.5
