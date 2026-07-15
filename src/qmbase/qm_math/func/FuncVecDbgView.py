# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncVecDbgView.py Created : 2025-07-04 at 4:43 pm by Dmitry.A.Konovalov@gmail.com
from javax.utilx.log.kiss.KissLog import KissLog
from qm_math.func.FuncVec import FuncVec
from qm_math.vec.VecDbgView import VecDbgView


class FuncVecDbgView:
    def __init__(self, vec):
        if isinstance(vec, FuncVec):
            self._func_vec = vec  # Copy from Vec
        # elif isinstance(from_vec, (list, tuple)):  # Or from array
        #     super().__init__(from_vec)
        else:
            raise TypeError(f"VecDbgView constructor expected Vec or list, got {type(vec)}")

    # @staticmethod
    def toString(self):
        mssg_x = 'x = ' + str(VecDbgView(self._func_vec.getX()))
        mssg_y = 'y = ' + str(VecDbgView(self._func_vec.getY()))
        mssg = '\n' + mssg_x + '\n' + mssg_y
        return mssg

    # Python dunders
    __str__  = toString
    __repr__ = toString


# printVec
KissLog.register_formatter(FuncVec, lambda v: str(FuncVecDbgView(v)))

