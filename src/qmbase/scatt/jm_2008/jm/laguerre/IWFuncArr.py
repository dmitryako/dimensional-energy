# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : IWFuncArr.py Created : 2025-06-30 at 12:15 pm by Dmitry.A.Konovalov@gmail.com
# Copyright dmitry.konovalov@jcu.edu.au Date: 21/11/2008, Time: 13:58:47

from abc import ABC, abstractmethod
from qm_math.func.arr.IFuncArr import IFuncArr

class IWFuncArr(IFuncArr, ABC):
    @abstractmethod
    def getQuadr(self):
        """Return WFQuadrD1"""
        pass
