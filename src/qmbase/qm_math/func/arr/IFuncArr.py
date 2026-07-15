# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : IFuncArr.py Created : 2025-06-27 at 12:28 pm by Dmitry.A.Konovalov@gmail.com

# ifunc_arr.py  –– Python interface mirroring math.func.arr.IFuncArr
# -------------------------------------------------------------------
# The names and signatures match the original Java interface exactly,
# so existing code that calls getX(), getFunc(i), setFunc(i,fv), size()
# will keep working without edits.

from __future__ import annotations
from abc import ABC, abstractmethod

from qm_math.func.FuncVec import FuncVec
from qm_math.vec.Vec import Vec

class IFuncArr(ABC):
    """
    Abstract base class corresponding to the Java interface
    `math.func.arr.IFuncArr`.

    * getX()       – return the shared X-grid (Vec)
    * getFunc(i)   – i-th FuncVec in the container
    * setFunc(i,f) – replace i-th element
    * size()       – number of FuncVec objects stored
    """

    # -----------------------------------------------------------
    @abstractmethod
    def getX(self) -> Vec: ...

    @abstractmethod
    def getFunc(self, i: int) -> FuncVec: ...

    @abstractmethod
    def __getitem__(self, i: int) -> FuncVec:
        pass

    @abstractmethod
    def setFunc(self, i: int, fv: FuncVec) -> None: ...

    @abstractmethod
    def size(self) -> int: ...

    # Optional Python conveniences ------------------------------
    def __len__(self) -> int:          # so len(obj) works
        return self.size()

    def __getitem__(self, idx: int) -> FuncVec:
        return self.getFunc(idx)
