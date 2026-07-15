# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LgrrOrthLcr.py Created : 2025-06-30 at 12:34 pm by Dmitry.A.Konovalov@gmail.com
from _new25.dbg import dbg
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from javax.utilx.log.Log import Log
from qm_math.integral.OrthFactory import OrthFactory
from scatt.jm_2008.jm.laguerre.IWFuncArr import IWFuncArr
from scatt.jm_2008.jm.laguerre.LgrrOpt import LgrrOpt
from scatt.jm_2008.jm.laguerre.LgrrOrthR import LgrrOrthR
from atom.wf.lcr.func.FuncRToDivSqrtCR import FuncRToDivSqrtCR
from qm_math.func.polynom.laguerre.LgrrOrth import LgrrOrth
from qm_math.vec.Vec import Vec

"""
Copyright dmitry.konovalov@jcu.edu.au Date: 16/09/2008, Time: 16:42:53
"""
log = Log.getLog('LgrrOrthLcr')


class LgrrOrthLcr(LgrrOrth, IWFuncArr):
    HELP = "The LCT transform of\n" + LgrrOrthR.HELP

    def __init__(self, w: WFQuadrLcr, model: LgrrOpt, max_orth_err=1e-10):
        super().__init__(
            w.getR(),
            model.getN(),
            2 * model.getL() + 2,
            model.getLambda()
        )  # NOTE!!! calculated on r
        self.mult(FuncRToDivSqrtCR(w.getLcrToRFunc()))    # NOTE!!!  /sqrt(c+r)
        self.setX(w.getX())             # NOTE!!! but stores LCR as x
        self._quadr = w

        orth_err = OrthFactory.calcMaxOrthErr(self, self._quadr)
        dbg('orth_err')
        print('orth_err=', orth_err)
        assert abs(orth_err) < max_orth_err, f'Fix orth_err={orth_err} < max_orth_err={max_orth_err}'


    def getQuadr(self) -> WFQuadrLcr:
        return self._quadr

    def getR(self) -> Vec:
        return self._quadr.getR()
