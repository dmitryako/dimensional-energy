# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : func_arr_2d.py Created : 2025-07-05 at 12:53 pm by Dmitry.A.Konovalov@gmail.com
from atom.wf.WFQuadrD1 import WFQuadrD1
from qm_math.func.FuncVec import FuncVec
from qm_math.func.arr.FuncArr import FuncArr
from scatt.jm_2008.jm.laguerre.IWFuncArr import IWFuncArr

class FuncArr2D_item:
    def __init__(self, *, i1, fv1:FuncVec, i2, fv2:FuncVec):
        # assert isinstance(i1, int)
        # assert isinstance(i2, int)
        assert isinstance(i1, int) or isinstance(i1, str)  # 250817: int is not used as 'int'
        assert isinstance(i2, int) or isinstance(i2, str)
        assert isinstance(fv1, FuncVec)
        assert isinstance(fv2, FuncVec)
        self.i1 = i1
        self.fv1 = fv1  # FuncVec
        self.i2 = i2
        self.fv2 = fv2

# class FuncArr2D(LgrrOrth, IWFuncArr):
class FuncArr2D:
    HELP = (
        "2d expansion in one basis"
    )

    def __init__(self, *args):  # both IWFuncArr(IFuncArr) and FuncArr(IFuncArr)
        if len(args) == 2:
            orth1, orth2 = args

        if len(args) == 1:
            basis_1d = args[0]
            assert isinstance(basis_1d, IWFuncArr)  # could be any FuncArr???
            # assert isinstance(basis_1d, FuncArr)  # could be any FuncArr???
            self._basis_1d = basis_1d
            self._quadr = basis_1d.getQuadr()  # no need? can always self._basis_1d.getQuadr()
            self._items = None
            self.load_basis_2d()

    @property
    def items(self): return self._items  # python way
    @property
    def basis_1d(self) -> IWFuncArr: return self._basis_1d  # python way .wf
    @property
    def quadr_1d(self) -> WFQuadrD1: return self._quadr  # python way .wf
    @property
    def grid_1d(self) -> WFQuadrD1: return self._quadr.getX()  # python way .wf

    def getQuadr(self):
        return self._quadr

    def __len__(self):  # started adding python-like
        return len(self._items)

    def size(self):
        return len(self._items)
    def load_basis_2d(self):
        n = self._basis_1d.size()
        # todo? symmetric?
        self._items = []
        for i in range(n):
            wf1 = self._basis_1d.getFunc(i)
            for j in range(n):
                wf2 = self._basis_1d.getFunc(j)
                item = FuncArr2D_item(i1=i, i2=j, fv1=wf1, fv2=wf2)
                self._items.append(item)
