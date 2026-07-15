# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : MtrxDbgView.py Created : 2025-07-08 at 2:19 pm by Dmitry.A.Konovalov@gmail.com
import os

from qm_math.mtrx.api.Mtrx import Mtrx
from qm_math.vec.DbgView import DbgView
from qm_math.vec.Vec import Vec
from qm_math.vec.VecDbgView import VecDbgView
import os

class MtrxDbgView(Mtrx):
    def __init__(self, from_mtrx: Mtrx):
        super().__init__(data=from_mtrx)

    def toString(self, *args) -> str:
        if len(args) == 0:
            return self._toString()
        elif len(args) == 3:
            return self._toString3(args[0], args[1], args[2])
        else:
            raise TypeError("Unsupported constructor arguments for Quadr.")

    def _toString(self) -> str:
        nr = self.getNumRows()
        head = f"Mtrx[{self.getNumRows()}][{self.getNumCols()}] = {{{os.linesep}"
        if DbgView.getNumShow() >= nr:
            return head + MtrxDbgView._toString3(self, 0, nr)
        n = DbgView.getNumShow() // 2
        start = MtrxDbgView._toString3(self, 0, n)
        tail = MtrxDbgView._toString3(self, nr - n, n)
        return head + start + os.linesep + ", ..., " + os.linesep + tail + os.linesep + "}"

    @staticmethod
    def _toString3(m: Mtrx, firstIdx: int, size: int) -> str:
        nr = m.getNumRows()
        length = min(nr, firstIdx + size)
        lines = []
        for i in range(firstIdx, length):
            vec = Vec(m.getRowCopy(i))
            lines.append(str(VecDbgView(vec)))
        return os.linesep.join(lines)

    def __str__(self) -> str:
        return self._toString()

    __repr__ = __str__
