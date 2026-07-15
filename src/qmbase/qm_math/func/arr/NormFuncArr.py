# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : NormFuncArr.py Created : 2025-06-30 at 1:24 pm by Dmitry.A.Konovalov@gmail.com
# Copyright dmitry.konovalov@jcu.edu.au Date: 30/07/2008, Time: 13:52:36

from qm_math.func.arr.FuncArr import FuncArr
from qm_math.integral.Quadr import Quadr
from qm_math.vec.Vec import Vec
from javax.utilx.log.Log import Log

class NormFuncArr(FuncArr):
    log = Log.getLog('NormFuncArr')

    def __init__(self, *args):
        """
        Constructor overload:
        - NormFuncArr(Vec x, int arrSize)
        - NormFuncArr(Quadr w, FuncArr from_arr)
        - NormFuncArr(Quadr w)
        """
        if len(args) == 2 and isinstance(args[0], Vec) and isinstance(args[1], int):
            super().__init__(args[0], args[1])
            self.normQuadr = None
            self.refQuadr = None
        elif len(args) == 2 and isinstance(args[0], Quadr) and isinstance(args[1], FuncArr):
            super().__init__(args[1])
            self.normQuadr = args[0]
            self.refQuadr = args[0]
        elif len(args) == 1 and isinstance(args[0], Quadr):
            super().__init__(args[0].getX())
            self.normQuadr = args[0]
            self.refQuadr = args[0]
        else:
            raise ValueError("Invalid constructor arguments for NormFuncArr")

    def getNormQuadr(self):
        return self.normQuadr

    def setNormQuadr(self, w):
        self.normQuadr = w

    def getRefQuadr(self):
        return self.refQuadr

    def setRefQuadr(self, w):
        self.refQuadr = w

    # public void makeOrthRotate() {
    #   OrthFactory.makeOrthRotate(this, normQuadr);
    # }

    # public double calcMaxOrthonErr() {
    #   double normErr = OrthFactory.calcMaxOrthonErr(this);
    #   return normErr;
    # }
