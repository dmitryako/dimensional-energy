# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Range.py Created : 2025-06-26 at 12:41 pm by Dmitry.A.Konovalov@gmail.com

# range.py  –– Port of  math.vec.grid.Range   (9 Jul 2008, 14:55:41)
#
# Very small utility class; method names kept exactly the same as Java.
# Internal fields use “_” prefix to mark them as private in Python style.
# -------------------------------------------------------------------------

class Range:
    def __init__(self, first: float, last: float) -> None:
        self._left: float = first
        self._right: float = last
        self._range: float = last - first
        # safeguard: avoid division-by-zero if first == last
        self._oneOver: float = 1.0 / self._range if self._range != 0.0 else float("inf")

    # ---- getters (same names as Java) ------------------------------------
    def getLeft(self) -> float:
        return self._left

    def getRight(self) -> float:
        return self._right

    def getRange(self) -> float:
        return self._range

    def getOneOver(self) -> float:
        return self._oneOver
