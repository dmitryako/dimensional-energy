# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncVec.py Created : 2025-06-26 at 2:41 pm by Dmitry.A.Konovalov@gmail.com

# func_vec.py  –– Port of math.func.FuncVec
# (Java original: 09 Jul 2008, 15 : 51 : 49)
#
# • Extends the previously-ported Vec.
# • Replaces   FastLoop.calc(arr, x.getArr(), f)   with a NumPy/pythonic
#   element-wise evaluation:
#         y[i] = f.calc(x[i])
# • Multiple Java constructors folded into one __init__(*args) that
#   inspects the argument pattern (see doc-string below).
# • Simple stubs for DerivFactory and FuncVecToString so the file runs.
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Union, Sequence, Any
import numpy as np

from qm_math.func.Func import Func
from qm_math.func.deriv.DerivFactory import DerivFactory
from qm_math.vec.Vec import Vec

NumberArray = Union[Sequence[Union[int, float]], np.ndarray]
# class FuncVec(Vec):

class FuncVec:
    # ---------------- constructor dispatcher -----------------------------
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Vec):  # (x_vec)
            x = args[0]
            assert not isinstance(args[0], FuncVec)
            # super().__init__(x.size())
            self._y = Vec(x.size())  # NEW
            self._x = x

        elif len(args) == 1 and isinstance(args[0], FuncVec):  # (FuncVec)
            func_vec = args[0]
            # super().__init__(func_vec.getY())
            # self._x: Vec = func_vec._x
            self._y = func_vec.getY()
            self._x: Vec = func_vec.getX()

        elif len(args) == 2 and isinstance(args[1], (Vec, np.ndarray, Sequence)):
            # (x_vec, y_vec_or_array)
            x, y = args
            # super().__init__(y)
            self._y = Vec(y)
            self._x = x

        elif len(args) == 2 and isinstance(args[1], Func):  # (x_vec, func)
            x, f = args
            # super().__init__(x.size())
            self._y = Vec(x.size())  # NEW
            self._x = x
            self.calc(f)

        else:
            raise TypeError("Unsupported constructor arguments for FuncVec.")

        # derivatives (lazy)
        self._drv: Vec | None = None
        self._drv2: Vec | None = None

    # ---------------- calculation ----------------------------------------
    def calc(self, f: Func) -> None:
        """Compute y[i] = f.calc(x[i]) in-place."""
        # x_arr = self._x.getArr()[: self._x.size()]
        x_arr = self._x.getArr()
        # self._arr[: self._size] = np.array(
        #     [f.calc(v) for v in x_arr], dtype=np.float64
        # )
        tmp = np.array(
            [f.calc(v) for v in x_arr], dtype=np.float64
        )
        # self._arr[:] = tmp[:]  # make sure it's in place
        self._y._arr[:] = tmp[:]  # make sure it's in place

    # ---------------- getters / setters ----------------------------------
    def setX(self, xGrid: Vec) -> None:
        self._x = xGrid

    def getX(self) -> Vec:
        return self._x

    def getY(self) -> "Vec":
        return self._y
    def setY(self, new_y) :
        assert isinstance(new_y, Vec)
        self._y = new_y

    # ---------------- derivative accessors -------------------------------
    def getDrv(self) -> Vec:
        if self._drv is None:
            self._drv = DerivFactory.makeDeriv(self)
        return self._drv

    def getDrv2(self) -> Vec:
        if self._drv is None:
            self._drv = DerivFactory.makeDeriv(self)
        if self._drv2 is None:
            drv = FuncVec(self.getX(), self._drv)
            # self._drv2 = DerivFactory.makeDeriv(self._drv)
            self._drv2 = DerivFactory.makeDeriv(drv)
        return self._drv2

    # ---------------- string helpers -------------------------------------
    # def toTab(self) -> str:
    #     return FuncVecToString(self).toTab()
    def toCSV(self) -> str:
        # return ", ".join(f"{self._arr[i]:.6g}" for i in range(self._size))
        return ", ".join(f"{self.getArr()[i]:.6g}" for i in range(self.size()))

    def __str__(self) -> str:  # overrides Vec.__str__
        # return "x=" + self._x.__str__() + "\ny=" + super().__str__()
        return "x=" + self._x.__str__() + "\ny=" + self._y.__str__()

    # Replicating all before was FuncVec(Vec)
    def mult(self, val):
        self._y.mult(val)

    def multSelf(self, s: "Vec", s2=None) -> None:
        self._y.multSelf(s, s2)

    def get(self, i):
        return self._y.get(i)

    def set(self, idx, val):
        self._y.set(idx, val)

    @property
    def arr(self): return self._y.getArr()  # python way
    # redirecting all calls to self._y; before was FuncVec(Vec):
    def getArr(self): return self._y.getArr()

    def size(self) -> int:  # old was class FuncVec(Vec):
        return self._y._size

    def addMultSafe(self, c: float, from_: "Vec") -> None:
        self._y.addMultSafe(c, from_)

    def add(self, s: float) -> None:  # in-place
        self._y.add(s)

    def getFirst(self):
        return self._y.getFirst()

    def getLast(self):
        return self._y.getLast()


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # grid x = [-1, -0.5, 0, 0.5, 1]
    x_vec = Vec(np.linspace(-1.0, 1.0, 5))


    # simple function  f(x) = x²
    class Square(Func):
        def calc(self, x: float) -> float: return x * x


    fv = FuncVec(x_vec, Square())  # (x_vec, Func) constructor
    print("Table:\n" + fv.toTab())

    print("\nFirst derivative approx:", fv.getDrv().toCSV())
    print("Second derivative approx:", fv.getDrv2().toCSV())
