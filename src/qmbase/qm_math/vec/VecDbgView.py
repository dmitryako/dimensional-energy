# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : VecDbgView.py Created : 2025-06-30 at 12:54 pm by Dmitry.A.Konovalov@gmail.com
# Copyright dmitry.konovalov@jcu.edu.au Date: 27/10/2008, Time: 11:10:09
import numpy as np

from javax.utilx.log.Log import Log
from javax.utilx.log.kiss.KissLog import KissLog
from qm_math.func.FuncVec import FuncVec
from qm_math.vec.DbgView import DbgView
from qm_math.vec.Vec import Vec
from qm_math.vec.grid.StepGrid import StepGrid


class VecDbgView(Vec):
    def __init__(self, from_):
        if isinstance(from_, Vec):
            super().__init__(from_.getArr())  # Copy from Vec
        elif isinstance(from_, FuncVec):  # Or from FuncVec
            super().__init__(from_.getY())
        elif isinstance(from_, np.ndarray):  # Or from FuncVec
            super().__init__(from_)
        elif isinstance(from_, (list, tuple)):  # Or from array
            super().__init__(from_)
        else:
            raise TypeError(f"VecDbgView constructor expected Vec or list, got {type(from_)}")

    def __str__(self):
        if DbgView.getNumShow() >= self.size():
            return self.toString(self.getArr(), self.size())
        n = DbgView.getNumShow() // 2
        start = self.toString(self._arr, 0, n)
        tail = self.toString(self._arr, len(self._arr) - n, n)
        head = f"Vec[{self.size()}] = {{"
        return f"{head}{start}, ..., {tail}}}"

    @staticmethod
    def toString(a, firstIdx, size=None):
        if size is None:
            size = len(a)
            firstIdx = 0
        len_limit = min(len(a), firstIdx + size)
        buff = []
        for i in range(firstIdx, len_limit):
            buff.append(DbgView.toStr(a[i]))
            # buff.append(str(a[i]))
        return ", ".join(buff)

    # def __str__(self) -> str:
    #     return self.toString(self)

    __repr__ = __str__



# printVec
KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))

# -------------------- quick demo / self-test -------------------------
if __name__ == "__main__":
    log = Log.getLog(StepGrid)

    g1 = StepGrid(first=0.0, last=10.0, size=101)  # primary ctor
    print("g1:", g1.toCSV())
    log.info("g1", g1)


