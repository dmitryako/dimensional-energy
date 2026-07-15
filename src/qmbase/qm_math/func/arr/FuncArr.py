# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncArr.py Created : 2025-06-27 at 12:43 pm by Dmitry.A.Konovalov@gmail.com

# func_arr.py  –– Port of math.func.arr.FuncArr
# ----------------------------------------------------------------------
# * Inherits  VecArr  and implements the IFuncArr interface.
# * All **Java** method names are kept when there is no conflict.
#   The one overloaded pair,  mult(Vec) vs mult(Func),
#   must be renamed in Python – we keep VecArr.mult(Vec) unchanged and
#   call the second form **mult_func(Func)** (see comment below).
# ----------------------------------------------------------------------

from __future__ import annotations
from typing import Optional

from javax.utilx.log.Log import Log
from qm_math.func.Func import Func
from qm_math.func.FuncVec import FuncVec
from qm_math.func.arr.IFuncArr import IFuncArr
from qm_math.vec.Vec import Vec
from qm_math.vec.VecArr import VecArr

log = Log.getLog("FuncArr")

class FuncArr(VecArr, IFuncArr):
    # Container of FuncVec objects that share a common X–grid.
    def __init__(self,
                 x: Vec | "FuncArr",
                 arrSize: Optional[int] = None) -> None:
        if isinstance(x, FuncArr):                      # copy-constructor
            super().__init__(x)
            self._x = x._x
            return
        # x is a Vec
        super().__init__()                              # create empty list
        self._x = x
        if arrSize is not None:
            self._init(x, arrSize)

    # ---- Java-style accessors ----------------------------------------
    def getFunc(self, i: int) -> FuncVec:          # Java getFunc(int)
        return super().get(i)
    def __getitem__(self, i: int) -> FuncVec:  # now can be called via [i]
        return super().get(i)

    def setFunc(self, i: int, fv: FuncVec) -> None:
        super().set(i, fv)

    # These return types are narrowed just like the Java overrides
    def get(self, i: int) -> FuncVec:              # hides VecArr.get
        return super().get(i)                      # type: ignore[return-value]

    def getLast(self) -> FuncVec:
        return super().getLast()                   # type: ignore[return-value]

    def getFirst(self) -> FuncVec:
        return super().getFirst()                  # type: ignore[return-value]

    # ---- internal helper ---------------------------------------------
    def _init(self, x: Vec, arrSize: int) -> None:
        shared_x = x
        for _ in range(arrSize):
            self.add(FuncVec(x))

    # ---- grid access --------------------------------------------------
    def getX(self) -> Vec:
        return self._x

    def setX(self, x: Vec) -> None:
        self._x = x
        for i in range(self.size()):
            self.getFunc(i).setX(x)

    # ---- pretty printing ---------------------------------------------
    def toString(self) -> str:                     # Java toString()
        return f"x={self._x}\n" + super().toString()

    # def toTab(self) -> str:  # TODO?
    #     return FuncArrToString(self).toTab()

    def toCSV(self) -> str:
        return self._x.toCSV() + "\n" + super().toCSV()

    # ---- multiplication helpers --------------------------------------
    # VecArr.mult(Vec) is inherited unchanged.  # BUG!!
    # The Java overload  mult(Func)  is renamed to avoid clash.
    def mult(self, a: Func) -> None:  # my fix dk250630
        if isinstance(a, Func):
            return self.mult_func(a)
        return super().mult(a)

    def mult_func(self, func: Func) -> None:
        """
        multiply each stored FuncVec by *func(x)* element-wise.
        (Renamed from Java's  mult(Func)  because Python cannot overload.)
        """
        f_vals = FuncVec(self._x, func)
        self.mult(f_vals)                          # Vec-wise multiply

    # ---- copy utilities ----------------------------------------------
    def copyFrom(self, idxDest: int,
                 fromArr: "FuncArr",
                 idxFirst: int,
                 idxLastExcl: int) -> None:
        idx = idxDest
        for i in range(idxFirst, idxLastExcl):
            self.set(idx, fromArr.get(i))
            idx += 1

    def copyDeepFromY(self, idxDest: int,
                      fromArr: "FuncArr",
                      idxFirst: int,
                      idxLastExcl: int) -> None:
        idx = idxDest
        for i in range(idxFirst, idxLastExcl):
            self.setFunc(idx, fromArr.getFunc(i).copyY())
            idx += 1

    def copyDeepY(self) -> "FuncArr":
        res = FuncArr(self._x, self.size())
        for i in range(self.size()):
            res.set(i, self.getFunc(i).copyY())
        return res
