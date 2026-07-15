# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : H1d1e_Lr.py Created : 2025-07-13 at 6:55 am by Dmitry.A.Konovalov@gmail.com

from _new25.dbg import dbg
from d1.d1e1.H1d1e import H1d1e
from javax.utilx.log.Log import Log

log = Log.getLog('H1d1e_Lcr')

class H1d1e_Lcr(H1d1e):
    # x = ln r, r = exp(x)
    # todo: NOTE! and P(r) = F(x)/sqrt(c+r)
    # D1x = d/dx,  D2x = d^2/dx^2
    # D2r = d^2/dr^2 P(r) =  = [D2x - 1/4] F(x)
    # Kin = (-0.5) D2r
    @staticmethod
    def apply_HE_atomZ_inX_np(*, atom_z, wLcr: 'WFQuadrLcr', wf, eng):
        # TODO: BUG at r=0
        #         # H1d1eDiagLcr_Test: res = Vec[301] = {0.00496, 0.00000, -0.00000,
        #         # DistMaxAbsErr: MAX dist[i=0] = 0.0049575  v=0  v2=0.0049575
        #         # max_err = 0.004957504348112707
        #
        from qm_math.mtrx.api.Mtrx import Mtrx
        from qm_math.mtrx.MtrxDbgView import MtrxDbgView
        from qm_math.vec.VecDbgView import VecDbgView
        from qm_math.vec.Vec import Vec
        from qm_math.func.deriv.DerivFactory import DerivFactory
        from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
        assert isinstance(wLcr, WFQuadrLcr), 'need WFQuadrLcr'
        x_grid = wLcr.getX().arr
        r = wLcr.getR().arr  # todo NOTT USED !!!  just checking
        dbg(r)
        # r = Vec[301] = {0.00000, 0.00053, 0.00108, 0.00165, 0.00223, ..., 89.15798, 91.75300, 94.42353, 97.17178, 100.00000}
        log.dbg("r =", VecDbgView(Vec(r)))

        # D1 = DerivFactory.makeDerivMtrxV2(x_grid)
        # todo: sparse
        mtrx_D1 = DerivFactory.calcDerivMtrxPts9(wLcr.getX())
        D1 = mtrx_D1.mtrx
        # D1 = DerivFactory.make_deriv_mtrx_5pt(x_grid)
        # D1 = DerivFactory.make_deriv_mtrx_9pt(x_grid)
        dbg(D1)
        # H1d1e_Lcr: D1 = Mtrx[301][301] = {  (301, 301) float64 min= -650.757707221244 mean= -5.85740340361392e-19 max= 650.757707221244
        # Vec[301] = {-94.74999, 278.89616, -488.06828, 650.75771, -610.08535, ..., 0.00000, 0.00000, 0.00000, 0.00000, 0.00000}
        log.dbg("D1 =", MtrxDbgView(Mtrx(data=D1)))

        D2 = D1 @ D1
        dbg(D2)
        # H1d1e_Lcr: D2 = Mtrx[301][301] = {  (301, 301) float64 min= -119970.02043446284 mean= 1.0656528412368207e-15 max= 123713.78704050765
        # Vec[301] = {7247.36001, -34571.79585, 80302.91551, -119970.02043, 123713.78704, ..., 0.00000, 0.00000, 0.00000, 0.00000, 0.00000}
        log.dbg("D2 =", MtrxDbgView(Mtrx(data=D2)))
        # n = D2.shape[0]
        # K = (-0.5)*(D2 - (1./4) * np.eye(n, dtype=np.float64))
        K1 = (-0.5)*D2
        dbg(K1)
        # H1d1e_Lcr: K = Mtrx[301][301] = {  (301, 301) float64 min= -61856.89352025383 mean= 0.00041528239202604525 max= 59985.01021723142
        # Vec[301] = {-3623.55500, 17285.89792, -40151.45775, 59985.01022, -61856.89352, ..., -0.00000, -0.00000, -0.00000, -0.00000, -0.00000}
        log.dbg("K1 =", MtrxDbgView(Mtrx(data=K1)))

        # todo!!! Err K_wf[0]=0.00496
        K1_wf = K1 @ wf
        K2_wf = (-0.5) * (-1./4) * wf
        dbg(K1_wf)
        dbg(K2_wf)
        # (301,) float64 min= -0.30137537753352106 mean= 0.05380622935398439 max= 0.3990756752219795
        # H1d1e_Lcr: K_wf = Vec[301] = {0.00496, 0.00517, 0.00539, 0.00563, 0.00587, ..., 0.00000, -0.00000, -0.00000, 0.00000, -0.00000}
        log.dbg("K1_wf =", VecDbgView(Vec(K1_wf)))
        log.dbg("K2_wf =", VecDbgView(Vec(K2_wf)))
        K_wf = K1_wf + K2_wf
        dbg(K_wf)
        log.dbg("K_wf =", VecDbgView(Vec(K_wf)))

        cr2DivR = wLcr.getCR2DivR().arr
        dbg(cr2DivR)
        # (301,) float64 min= 0.0 mean= 11.776001302435912 max= 100.03663463240375
        # H1d1e_Lcr: cr2DivR = Vec[301] = {0.00000, 0.66657, 0.34790, 0.24209, 0.18951, ..., 89.19462, 91.78964, 94.46017, 97.20842, 100.03663}
        log.dbg("cr2DivR =", VecDbgView(Vec(cr2DivR)))

        pot = -atom_z * cr2DivR  # todo: NOTE orig was 1/r, now it's * r
        dbg(pot)
        # (301,) float64 min= -100.03663463240375 mean= -11.776001302435912 max= -0.0
        # H1d1e_Lcr: pot = Vec[301] = {-0.00000, -0.66657, -0.34790, -0.24209, -0.18951, ..., -89.19462, -91.78964, -94.46017, -97.20842, -100.03663}
        log.dbg("pot =", VecDbgView(Vec(pot)))

        # todo: BUG V_wf[0] should be -0.00496
        # V_wf = Vec[301] = {-0.00000, -0.00517, -0.00540,
        V_wf = pot * wf
        dbg(V_wf)
        # (301,) float64 min= -0.8348968948860385 mean= -0.20868748553318706 max= -0.0
        # H1d1e_Lcr: V_wf = Vec[301] = {-0.00000, -0.00517, -0.00540, -0.00563, -0.00588, ..., -0.00000, -0.00000, -0.00000, -0.00000, -0.00000}
        log.dbg("V_wf =", VecDbgView(Vec(V_wf)))

        cr2 = wLcr.getCR2().arr
        dbg(cr2)
        # (301,) float64 min= 0.0003354626279025118 mean= 596.0917158561417 max= 10003.663463240375
        # H1d1e_Lcr: cr2 = Vec[301] = {0.00034, 0.00036, 0.00038, 0.00040, 0.00042, ..., 7952.41204, 8421.97453, 8919.26307, 9445.91478, 10003.66346}
        log.dbg("cr2 =", VecDbgView(Vec(cr2)))

        E_wf = eng * wf * cr2  # todo: NOTE! orig was *1., now *r2
        dbg(E_wf)
        # (301,) float64 min= -0.8199175516319266 mean= -0.1548977262932909 max= -0.0
        # H1d1e_Lcr: E_wf = Vec[301] = {-0.00000, -0.00000, -0.00000, -0.00000, -0.00001, ..., -0.00000, -0.00000, -0.00000, -0.00000, -0.00000}
        log.dbg("E_wf =", VecDbgView(Vec(E_wf)))

        pred_zero = K_wf + V_wf - E_wf
        dbg(pred_zero)
        # (301,) float64 min= -2.6961960442051236e-11 mean= 1.6470114088245117e-05 max= 0.004957504347203212
        # H1d1e_Lcr: pred_zero = Vec[301] = {0.00496, 0.00000, -0.00000, -0.00000, 0.00000, ..., 0.00000, -0.00000, -0.00000, 0.00000, -0.00000}
        log.dbg("pred_zero =", VecDbgView(Vec(pred_zero)))
        # true_zero = np.zeros_like(pot)
        return pred_zero


if __name__ == "__main__":
    from _new25.v250712_H1d1e_diag_OK.v250714a_Hy1d1eDiagLcr_TEST import H1d1eDiagLcr_Test
    H1d1eDiagLcr_Test().test_1()


