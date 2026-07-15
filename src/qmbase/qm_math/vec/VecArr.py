# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : VecArr.py Created : 2025-06-27 at 12:39 pm by Dmitry.A.Konovalov@gmail.com

# vec_arr.py  –– Port of math.vec.VecArr
# -------------------------------------------------------------
# • Method names, signatures, and semantics are preserved, so the
#   translated library can still call add(), get(), size(), toCSV(), …
# • Internal storage uses a plain Python list of Vec objects.
# -------------------------------------------------------------

from __future__ import annotations
from typing import List

import numpy as np

from javax.utilx.log.Log import Log
from qm_math.vec.Vec import Vec


# from vec import Vec


# ── minimal logger (same pattern as earlier ports) ─────────────
# class Log:
#     @staticmethod
#     def getLog(cls=None): return Log()
#     def dbg(self, *a):  return self
#     def info(self, *a): return self
# --------------------------------------------------------------


class VecArr:
    log = Log.getLog("VecArr")

    # ----------------------------------------------------------
    def __init__(self, src: "VecArr | None" = None) -> None:
        """
        • VecArr()           – empty container
        • VecArr(other)      – shallow-copy pointer to other's storage
        """
        if src is None:
            self._arr: List[Vec] = []
        else:
            self._arr = src._arr                                         # share reference

    # ----------------------------------------------------------
    # direct Java-style accessors (names unchanged)
    def setArr(self, src: "VecArr") -> None:
        self._arr = src._arr

    def get_np_in_rows(self):
        nr = self.get(0).size()
        n = self.size()
        ret = np.zeros(shape=[n, nr], dtype=float)
        for i in range(n):
            ret[i, :] = self.get(i).arr
        return ret

    def size(self) -> int:
        return len(self._arr)

    def get(self, i: int) -> Vec:
        return self._arr[i]

    def getLast(self) -> Vec:
        return self._arr[-1]

    def getFirst(self) -> Vec:
        return self._arr[0]

    def add(self, v: Vec) -> None:
        self._arr.append(v)

    def set(self, i: int, v: Vec) -> None:
        self._arr[i] = v

    # ----------------------------------------------------------
    def mult(self, v) -> None:
        """ element-wise in-place multiply: each Vec *= v """
        for vec in self._arr:
            vec.multSelf(v)

    # ----------------------------------------------------------
    # pretty-printing helpers
    def toString(self) -> str:                                           # Java toString()
        lines = [f"VecArr[{i}] = {vec}" for i, vec in enumerate(self._arr)]
        return "\n".join(lines)

    def toCSV(self) -> str:
        return "\n".join(vec.toCSV() for vec in self._arr)

    # Python niceties (optional)
    def __len__(self)  -> int:          return self.size()
    def __getitem__(self, i: int) -> Vec: return self.get(i)
    def __str__(self)   -> str:         return self.toString()
