# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncArr2e.py Created : 2025-07-10 at 7:39 am by Dmitry.A.Konovalov@gmail.com
from typing import cast

import numpy as np

from _new25.dbg import dbg
from d2_py.FuncArr2D import FuncArr2D, FuncArr2D_item
from javax.utilx.log.Log import Log

log = Log.getLog('FuncArr1d2e')

# NOTE!!! 2d item is f1(x), f2(y)
# NOTE!!! 2e item is f1(x1), f2(x2)
class FuncArr1d2e_item(FuncArr2D_item):
    pass

class FuncArr1d2e(FuncArr2D):
    def __init__(self, *args, load_symm_half=False):  # both IWFuncArr(IFuncArr) and FuncArr(IFuncArr)
        self._items = []
        self._map_ij_to_idx = {}
        assert not load_symm_half
        if len(args) == 2:
            orth1, orth2 = args[:2]

            from qm_math.func.arr.FuncArr import FuncArr
            assert isinstance(orth1, FuncArr), 'need FuncArr'
            assert isinstance(orth2, FuncArr), 'need FuncArr'
            from scatt.jm_2008.jm.laguerre.IWFuncArr import IWFuncArr
            assert isinstance(orth1, IWFuncArr), 'need? IWFuncArr'
            assert isinstance(orth2, IWFuncArr), 'need? IWFuncArr'
            self._basis_1d = cast(FuncArr, orth1)
            self._basis2_1d = cast(FuncArr, orth2)

            # load_symm_half = False
            # if len(args) == 3:
            #     load_symm_half = args[2]
            #     assert isinstance(load_symm_half, bool)
            self.load_basis_1and2(load_symm_half)

        if len(args) == 1:
            super().__init__(*args)

    @property
    def basis2_1d(self): return self._basis2_1d  # python way .wf

    @property
    def shape(self):
        res = (self._basis_1d.size(), self._basis2_1d.size())
        return res  # python way
    @property
    def map_ij_to_idx(self): return self._map_ij_to_idx

    def calc_norm_ij(self, i, j):
        return 0.5 if i == j else 1 / np.sqrt(2)
    def calc_anti_norm_ij(self, i, j):
        return 0.0 if i == j else 1 / np.sqrt(2)

    def load_basis_1and2(self, load_symm_half=False):
        n1 = self._basis_1d.size()
        n2 = self._basis2_1d.size()
        self._items = []
        self._map_ij_to_idx = {}

        for i1 in range(n1):
            wf1 = self._basis_1d.getFunc(i1)
            for i2 in range(n2):
                if load_symm_half and i2 > i1:
                    continue
                wf2 = self._basis2_1d.getFunc(i2)
                item = FuncArr2D_item(i1=i1, i2=i2, fv1=wf1, fv2=wf2)
                # key = (i1, i2)
                key = self.make_key(i1, i2)
                # dbg('key')
                self._map_ij_to_idx[key] = len(self._items)
                self._items.append(item)

    @staticmethod
    def make_key(i1, i2):
        key = f'({i1}, {i2})'  # (i, j)
        dbg('key')
        return key

    def append_from(self, basis2e, key_tag):
        #         self._items = []
        #         self._map_ij_to_idx = {}
        for _, from_item in enumerate(basis2e.items):
            from_item = cast(FuncArr1d2e_item, from_item)
            #         self.i1 = i1
            #         self.fv1 = fv1  # FuncVec
            #         self.i2 = i2
            #         self.fv2 = fv2
            # print(from_item)
            new_i1 = f'{key_tag}_{from_item.i1}'
            new_i2 = f'{key_tag}_{from_item.i2}'
            new_item = FuncArr2D_item(i1=new_i1, i2=new_i2, fv1=from_item.fv1, fv2=from_item.fv2)
            # new_key =  f'({new_i1}, {new_i2})' # (i, j)
            new_key =  self.make_key(new_i1, new_i2) # (i, j)
            dbg('new_key')
            log.dbg('new_key =', new_key)
            self._map_ij_to_idx[new_key] = len(self._items)
            self._items.append(new_item)

def append_two_orth1d2e(basis1, basis2, *, key_tag1, key_tag2):
    ret = FuncArr1d2e()
    ret.append_from(basis1, key_tag=key_tag1)
    ret.append_from(basis2, key_tag=key_tag2)
    return ret






