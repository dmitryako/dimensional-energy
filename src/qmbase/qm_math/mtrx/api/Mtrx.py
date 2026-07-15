# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Mtrx.py Created : 2025-06-30 at 9:46 am by Dmitry.A.Konovalov@gmail.com
# Next is tricky. I need to switch to np.array as the base for matrix=2d-arrays. Assume MtrxToStr is ready. Also please comment. If I need pytorch GPU, I would create a specialised method or class later specifically for any heavy-calculations. It should not be mixed into this Mtrx class. Yes?
#
# There maybe methods in MtrxEjml that are missing. we will convert them when needed. So now Mtrx will not inherit from any class.  make local implementation variable _mtrx. similar as we did _arr for vector type classes.

# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Mtrx.py   Created : 2025-06-29
"""
Lightweight dense-matrix wrapper used throughout *qm_math*.

*   Back-end  : **NumPy ndarray** (stored in ``self._mtrx``).
*   Interface : keeps every public method name that the Java class exposed
    so existing call-sites compile unchanged.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

from qm_math.func.FuncVec import FuncVec
# from qm_math.mtrx.MtrxToStr import MtrxToStr          # assumed to exist
from qm_math.vec.Vec import Vec                       # thin NumPy-based vector
# ---------------------------------------------------------------------------


class Mtrx:
    # def __init__(self, data: "Mtrx | Sequence[Sequence[float]] | tuple[int, int]"):
    def __init__(self, *, data=None, rows=None, cols=None, shallow=True):
        if isinstance(data, Mtrx):                          # copy-ctor
            self._mtrx = data._mtrx
            assert shallow, 'IN JAVA was always shallow!'
            if not shallow:  # todo?
                self._mtrx = data._mtrx.copy()
        elif isinstance(data, np.ndarray):                          # copy-ctor
            # self._mtrx = np.asarray(data, dtype=float)
            # self._mtrx = np.array(data, copy=False, dtype=float)
            self._mtrx = data
            assert shallow, 'IN JAVA was always shallow!'
            if not shallow:  # todo?
                self._mtrx = np.asarray(data, dtype=float).copy()
        elif rows is not None and cols is not None:
            assert data is None
            self._mtrx = np.zeros((rows, cols), dtype=float)
            # self._mtrx = np.zeros((rows, cols), dtype=np.float64)
        else:
            assert False, 'TODO!'
        # elif isinstance(data, (tuple, list)) and len(data) == 2 \
        #         and all(isinstance(x, int) for x in data):  # shape
        #     rows, cols = data
        #     self._mtrx = np.zeros((rows, cols), dtype=float)
        # else:                                               # assume 2-D array-like
        #     self._mtrx = np.asarray(data, dtype=float).copy()

    @property
    def mtrx(self): return self._mtrx  # python way
    def getNumRows(self) -> int:
        return self._mtrx.shape[0]

    def getNumCols(self) -> int:
        return self._mtrx.shape[1]

    # ---------- element access ---------------------------------------------
    def get(self, r: int, c: int) -> float:
        return float(self._mtrx[r, c])

    def set(self, r: int, c: int, val: float) -> None:
        self._mtrx[r, c] = val

    # ---------- string helpers (delegate to MtrxToStr) ----------------------
    # def __str__(self) -> str:                 # Java's toString()
    #     return MtrxToStr.toCsv(self)

    # def toTab(self, digs: int | None = None) -> str:
    #     """Tab-separated view (see *MtrxToStr*)."""
    #     return MtrxToStr.toTab(self, digs) if digs is not None else MtrxToStr.toTab(self)

    # ---------- linear algebra ---------------------------------------------
    def copy(self) -> "Mtrx":
        return Mtrx(self)

    def inverse(self) -> "Mtrx":
        return Mtrx(np.linalg.inv(self._mtrx))

    # in-place operations ----------------------------------------------------
    def addEquals(self, B: "Mtrx") -> "Mtrx":
        self._mtrx += B._mtrx
        return self

    def subEquals(self, B: "Mtrx") -> "Mtrx":
        self._mtrx -= B._mtrx
        return self

    def multEquals(self, scalar: float) -> "Mtrx":
        self._mtrx *= scalar
        return self

    # new-object operations --------------------------------------------------
    def mult(self, B: "Mtrx") -> "Mtrx":
        return Mtrx(self._mtrx @ B._mtrx)

    def transpose(self) -> "Mtrx":
        return Mtrx(self._mtrx.T)

    # matrix-vector product (kept SLOW fallback from Java for full API) ------
    def multVec(self, vec) -> Vec:               # Java name was mult(Vec)
        """`Vec_out = self · vec`  (dense × dense)."""
        if isinstance(vec, FuncVec):
            vec = vec.getY()
        return Vec(self._mtrx @ vec.getArr())

    # ---------- column / row helpers ---------------------------------------
    def getColCopy(self, c: int) -> np.ndarray:
        return self._mtrx[:, c].copy()

    def getRowCopy(self, r: int) -> np.ndarray:
        return self._mtrx[r, :].copy()

    def addSelf(self, r: int, c: int, d: float) -> None:
        self._mtrx[r, c] += d

    def multSelf(self, r: int, c: int, d: float) -> None:
        self._mtrx[r, c] *= d
