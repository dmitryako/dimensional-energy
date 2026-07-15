# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Vec.py Created : 2025-06-26 at 12:35 pm by Dmitry.A.Konovalov@gmail.com

# vec.py  –– NumPy-based rewrite of the old math.vec.Vec
#
# RULES YOU ASKED FOR
# -------------------
# •  Keep every Java method that existed.                         ✔
# •  If Java overloaded a name, give each Python version a
#    clear suffix like “…_func”, “…_vec_func”.                    ✔
# •  Internal data use “_arr”, “_size”; operations done with
#    plain NumPy (FastLoop discarded).                            ✔
# •  No “clever” Python features beyond list-comprehensions.      ✔
#
# Only external dependency:  pip install numpy
# Runs on Python 3.9+ (no “|” union types).
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Sequence, Union, List, Any
import numpy as np

from javax.utilx.log.Log import Log
from javax.utilx.log.kiss.KissLog import KissLog
from qm_math.func.Func import Func

# -------------------------------------------------------------------------


NumberArray = Union[Sequence[Union[int, float]], np.ndarray]


class Vec:
    # -------- static log --------------------------------------------------
    log = Log.getLog("Vec")

    # -------- constructors ------------------------------------------------
    def __init__(self, src: Union[int, NumberArray, "Vec"]) -> None:
        # todo!!!! It's always!!! shallow copy !!!!
        if isinstance(src, int):
            #     public Vec(int size) {
            #         this.arr = new double[size];
            #         this.size = size;
            self._arr = np.zeros(src, dtype=np.float64)
            self._size = src
        elif isinstance(src, Vec):
            # public Vec(Vec from) {
            #         this(from.arr);
            assert src._arr.dtype == np.float64
            # self._arr = src._arr.copy()
            self._arr = src._arr
            self._size = src._size
        elif isinstance(src, np.ndarray):  # array-like
            #     public Vec(double[] from) {
            #         this.arr = from;
            #         this.size = from.length;
            # self._arr = np.asarray(src, dtype=np.float64)
            self._arr = src
            self._size = self._arr.size
        elif isinstance(src, list):  # todo? NumberArray? array-like
            self._arr = np.asarray(src, dtype=np.float64)
            self._size = self._arr.size
        else:
            raise ValueError(
                Vec.log.error(f"Invalid src={src}")
            )

    # -------- size handling ----------------------------------------------
    def size(self) -> int:
        return self._size

    def setSize(self, newSize: int) -> None:
        if newSize < 0 or newSize > self._arr.size:
            raise ValueError(
                Vec.log.error(f"Invalid newSize={newSize}, oldSize={self._size}")
            )
        self._size = newSize

    # ---------- human-readable dump ----------------------------------
    def __str__(self) -> str:                     # prints only *active* part
        data = ", ".join(f"{self._arr[i]:.6g}" for i in range(self._size))
        return f"Vec[{self._size}]={{ {data} }}"

    # keep old Java helper name if you still call it elsewhere
    def toCSV(self) -> str:
        return ", ".join(f"{self._arr[i]:.6g}" for i in range(self._size))

    # -------- basic element access ---------------------------------------
    def getLast(self) -> float:
        return float(self._arr[self._size - 1])

    def getFirst(self) -> float:
        return float(self._arr[0])

    # def __call__(self, i: int) -> float: return float(self._arr[i]) # python way
    def __getitem__(self, i: int) -> float: return float(self._arr[i]) # python way
    def get(self, i: int) -> float: return float(self._arr[i])

    def set(self, i: int, v: float) -> None:
        self._arr[i] = v

    # -------- raw array getters / setters --------------------------------
    @property
    def arr(self): return self._arr  # python way
    def getArr(self) -> np.ndarray: return self._arr

    def setArr(self, arr: NumberArray) -> None:
        self._arr = np.asarray(arr, dtype=np.float64)
        self._size = self._arr.size

    # -------- copy & append ----------------------------------------------
    def copy(self) -> "Vec":
        return Vec(self._arr[: self._size].copy())

    def append(self, v2: "Vec") -> "Vec":
        joined = np.concatenate(
            (self._arr[: self._size], v2._arr[: v2._size])
        )
        return Vec(joined)

    # -------- *former* FastLoop.calc overloads ---------------------------
    def calc_vec_func(self, x: "Vec", f: Func) -> None:
        """Equivalent to Java  calc(Vec x, Func f)  (in-place)."""
        self._arr[: self._size] = np.array(
            [f.calc(v) for v in x.getArr()[: x.size()]], dtype=np.float64
        )

    def calc_func(self, f: Func) -> None:
        """Equivalent to Java  calc(Func f)  (in-place on self)."""
        self._arr[: self._size] = np.array(
            [f.calc(v) for v in self._arr[: self._size]], dtype=np.float64
        )

    # -------- simple arithmetic ------------------------------------------
    def add(self, s: float) -> None:  # in-place
        self._arr[: self._size] += s

    def mult(self, s: float) -> None:  # in-place
        self._arr[: self._size] *= s

    def multSelf(self, s, s2=None) -> None:
        if s2 is None and isinstance(s, float):
            self.mult(s)
            return
        if s2 is None:
            # self._arr[: self._size] *= s._arr[: s._size]
            self._arr[: self._size] *= s.getArr()[: s.size()]
        else:
            # self._arr[: self._size] *= s._arr[: s._size] * s2._arr[: s2._size]
            self._arr[: self._size] *= s.getArr()[: s.size()] * s2.getArr()[: s2.size()]

    def addMultSafe(self, c: float, from_: "Vec") -> None:
        from_arr = from_.getArr()
        if self.size() >= from_.size():
            # self._arr[: from_.size()] += c * from_._arr[: from_.size()]
            self._arr[: from_.size()] += c * from_arr[: from_.size()]
        else:
            tmp = from_.copy()
            tmp._arr[: self.size()] += c * self._arr[: self.size()]
            self.setArr(tmp._arr)

    # -------- reductions --------------------------------------------------
    def min(self) -> float:
        return float(self._arr[: self._size].min())

    def max(self) -> float:
        return float(self._arr[: self._size].max())

    def minIdx(self) -> int:
        return int(self._arr[: self._size].argmin())

    def maxIdx(self) -> int:
        return int(self._arr[: self._size].argmax())

    # -------- dot products -----------------------------------------------
    def dot(self, v2) -> float:
        if isinstance(v2, Vec):
            arr2 = v2._arr[: v2._size]
        else:  # assume raw array
            arr2 = np.asarray(v2, dtype=np.float64)
        return float(np.dot(self._arr[: self._size], arr2))

    # -------- addSafe -----------------------------------------------------
    def addSafe(self, from_: "Vec") -> None:
        if self.size() >= from_.size():
            self._arr[: from_.size()] += from_._arr[: from_.size()]
        else:
            tmp = from_.copy()
            tmp._arr[: self.size()] += self._arr[: self.size()]
            self.setArr(tmp._arr)

    # -------- static utilities -------------------------------------------
    @staticmethod
    def calcMedianFromSorted(e: NumberArray) -> float:
        arr = np.asarray(e, dtype=np.float64)
        if arr.size == 0:
            raise ValueError(Vec.log.error("calcMedianFromSorted(e.length==0)"))
        mid = arr.size // 2
        if arr.size % 2 == 0:
            return float(0.5 * (arr[mid - 1] + arr[mid]))
        return float(arr[mid])

    @staticmethod
    def mean(e: NumberArray) -> float:
        arr = np.asarray(e, dtype=np.float64)
        if arr.size == 0:
            raise ValueError(Vec.log.error("Vec.mean(e.length==0)"))
        return float(arr.mean())

    @staticmethod
    def medianSLOW(e: NumberArray) -> float:
        tmp = Vec.copy(e)
        tmp.sort()
        return Vec.calcMedianFromSorted(tmp)

    @staticmethod
    def medianNoCopySLOW(e: NumberArray) -> float:
        arr = np.asarray(e, dtype=np.float64)
        arr.sort()
        return Vec.calcMedianFromSorted(arr)

    @staticmethod
    def copy(from_: NumberArray) -> np.ndarray:
        return np.asarray(from_, dtype=np.float64).copy()

    @staticmethod
    def convert(arr: Union[List[float], List[int], np.ndarray]) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64).copy()

    # -------- misc --------------------------------------------------------
    def sortSelf(self) -> "Vec":
        self._arr[: self._size].sort()
        return self



log = Log.getLog(Vec)  # todo: both ways are ok: log = Log.getLog("Vec")
# -- pretty-printer plug-in for KissLog -------------------------------
KissLog.register_formatter(Vec, lambda v: str(v))
# from qm_math.vec.VecDbgView import VecDbgView
# KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))


if __name__ == '__main__':
    # import numpy as np
    # from vec import Vec

    v1 = Vec([1, 2, 3])
    v2 = Vec(np.arange(3) + 0.5)

    print("size:", v1.size())
    print("dot :", v1.dot(v2))
    v1.add(10)
    print("after add:", v1)

    v3 = v1.append(v2)
    # print("appended:", v3.toCSV())
