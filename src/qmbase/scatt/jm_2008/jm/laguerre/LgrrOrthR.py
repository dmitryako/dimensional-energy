# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LgrrOrthR.py Created : 2025-06-30 at 12:28 pm by Dmitry.A.Konovalov@gmail.com
# Copyright dmitry.konovalov@jcu.edu.au Date: 16/09/2008, Time: 14:12:21

from qm_math.func.polynom.laguerre.LgrrOrth import LgrrOrth
from atom.wf.WFQuadrR import WFQuadrR
from qm_math.vec.Vec import Vec
from scatt.jm_2008.jm.laguerre.LgrrOpt import LgrrOpt
from scatt.jm_2008.jm.laguerre.IWFuncArr import IWFuncArr

class LgrrOrthR(LgrrOrth, IWFuncArr):
    HELP = (
        "Orthonormal Laguerre lgrrN:\n"
        "R(n, a, lambda, r) = C_n * exp(-x/2) x^(a/2) L^a_n(x),\n"
        "where x = lambda * r,  a = alpha = 2*l+2, l - angular momentum, "
        "L^a_n - the associated Laguerre polynomials."
    )

    def __init__(self, w: WFQuadrR, lgrrOpt: LgrrOpt, how='any', scale=1):
        super().__init__(w.getX(), lgrrOpt.getN(), 2 * lgrrOpt.getL() + 2, lgrrOpt.getLambda())
        self.quadr = w
        self._scale = scale
        assert how in ['any', 'all', 'x_neg', 'x_pos', 'x_not_neg', 'x_not_pos']
        if scale != 1:
            self.mult(scale)
        if how in ['any', 'all']:
            return
        x = self.getX().arr
        if how == 'x_neg':
            mask = x < 0
        if how == 'x_pos':
            mask = x > 0
        if how == 'x_not_neg':
            mask = x >= 0
        if how == 'x_not_pos':
            mask = x <= 0
        self.mult(Vec(mask))

    #  public LgrrOrthR(Vec r, LgrrOpt model) {
    #    super(r, model.getMomN(), 2 * model.getTotL() + 2, model.getLambda());
    #  }

    def getQuadr(self):
        return self.quadr
