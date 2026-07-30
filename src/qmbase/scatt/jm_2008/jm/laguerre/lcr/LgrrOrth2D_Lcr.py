# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LgrrOrthLcr.py Created : 2025-06-30 at 12:34 pm by Dmitry.A.Konovalov@gmail.com
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from javax.utilx.log.Log import Log
from scatt.jm_2008.jm.laguerre.IWFuncArr import IWFuncArr
from scatt.jm_2008.jm.laguerre.LgrrOpt import LgrrOpt
from scatt.jm_2008.jm.laguerre.LgrrOrthR import LgrrOrthR
from atom.wf.lcr.func.FuncRToDivSqrtCR import FuncRToDivSqrtCR
from qm_math.func.polynom.laguerre.LgrrOrth import LgrrOrth
from qm_math.vec.Vec import Vec
from scatt.jm_2008.jm.laguerre.lcr.LgrrOrthLcr import LgrrOrthLcr

"""
Copyright dmitry.konovalov@jcu.edu.au Date: 2008-2026
"""


class LgrrOrth_anyD_Lcr(LgrrOrth, IWFuncArr):  # todo: NOTE! skipping LgrrOrthLcr
    # v260705
    log = Log.getLog('LgrrOrth_anyD_Lcr')
    HELP = "The LCT transform of\n" + LgrrOrthR.HELP

    def __init__(self, w: WFQuadrLcr, model: LgrrOpt):
        super().__init__(
            w.getR(),
            model.getN(),
            2 * model.getL() + 2,
            model.getLambda()
        )  # NOTE!!! calculated on r
        self.mult(FuncRToDivSqrtCR(w.getLcrToRFunc()))    # NOTE!!!  /sqrt(c+r)
        self.setX(w.getX())             # NOTE!!! but stores LCR as x
        self._quadr = w

    def getQuadr(self) -> WFQuadrLcr:
        return self._quadr

    def getR(self) -> Vec:
        return self._quadr.getR()


class LgrrOrth2D_Lcr(LgrrOrth, IWFuncArr):  # todo: NOTE! skipping LgrrOrthLcr
    # NOTE! This is actually anyD
    log = Log.getLog('LgrrOrth2D_Lcr')
    HELP = "The LCT transform of\n" + LgrrOrthR.HELP

    def __init__(self, w: WFQuadrLcr, model: LgrrOpt):
        super().__init__(
            w.getR(),
            model.getN(),
            2 * abs(model.get_m_2d()) + 2 * model.getL() + 2,
            model.getLambda()
        )  # NOTE!!! calculated on r
        self.mult(FuncRToDivSqrtCR(w.getLcrToRFunc()))    # NOTE!!!  /sqrt(c+r)
        self.setX(w.getX())             # NOTE!!! but stores LCR as x
        self._quadr = w

    def getQuadr(self) -> WFQuadrLcr:
        return self._quadr

    def getR(self) -> Vec:
        return self._quadr.getR()
