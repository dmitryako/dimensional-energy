# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : He1d2e_LgrrLcr.py Created : 2025-07-15 at 6:17 pm by Dmitry.A.Konovalov@gmail.com
import numpy as np
from typing import cast

from _new25.dbg import dbg
from atom.energy.pw.lcr.PotHLcr import PotHLcr
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from d1.d1e2.H2d2e_LcrAnti import H2d2e_LcrAnti
from javax.utilx.log.Log import Log
from qm_math.mtrx.MtrxDbgView import MtrxDbgView
from atom.energy.HMtrx import HMtrx
from d1.d1e1.H1d1e import H1d1e
from qm_math.mtrx.api.Mtrx import Mtrx

log = Log.getLog('H_3d_2e_LcrAnti')


# todo: old, make (H1d2e_LgrrLcr) to make sure not using anthing from H1d2e_LgrrLcrSym
class H_3d_2e_LcrAnti(H2d2e_LcrAnti):  # H2d2e_LcrAnti(H1d2e_LgrrLcrAnti)

    # todo: the same as H1d2e_LgrrLcrAnti

    def diag_on_2orth1(self, sign=1):
        cfg = self.cfg
        plot_on = cfg.plot_on

        h2x1e, pot2e = self.load_mats()
        dbg(h2x1e.mtrx)
        dbg(pot2e.mtrx)

        # if is_symm:  # symmetric
        #     from d1.d1e2.H_1d_2e_LgrrLcrSym import fix_sym
        #     h2x1e, pot2e = fix_sym(h2x1e, pot2e, orth2e=self.cfg.orth2e, ok_1e_diag=True, ex_sign=1)
        # else:
        #     from d1.d1e2.H_1d_2e_LgrrLcrSym import fix_sym
        #     h2x1e, pot2e = fix_sym(h2x1e, pot2e, orth2e=self.cfg.orth2e, ok_1e_diag=False, ex_sign=-1)
        # dbg([h2x1e.mtrx, pot2e.mtrx])

        # todo:  1D H^(0)-2e --------------------------
        from d1.d1e2.H_1d_2e_LgrrLcrSym import fix_sym
        # todo: see: H1d2e_AnyCoreLcr
        h2x1e, pot2e = fix_sym(h2x1e, pot2e, orth2e=self.cfg.orth2e, ok_1e_diag=False, ex_sign=-1)
        dbg([h2x1e.mtrx, pot2e.mtrx])

        h2x1e_engs = h2x1e.getEigEngs()
        vecs = h2x1e.getEigVec()
        log.dbg("h2x1e_anti.getEigEngs=", h2x1e_engs)
        dbg('self.cfg.eng_2x2d1e_anti')
        err_eng = h2x1e_engs.arr[0] - cfg.eng_2x2d1e_anti
        print('err_eng = ', err_eng)
        print('h2x1e_anti.engs', h2x1e_engs.arr[:4])
        assert abs(err_eng) < 1e-1, 'quick check'
        tag = f'psi2x1e_ANTI_idx{cfg.plot_evec_idx}'

        if plot_on:
            self.build_and_plot_wf_2e(vecs, tag=tag, sign=sign)

        # todo: FULL e+e
        h2e = h2x1e.mtrx + pot2e.mtrx
        dbg(h2e)
        h2e = HMtrx(mh=h2e)
        engs = h2e.getEigEngs()
        log.dbg("h2x1e_anti.getEigEngs=", h2x1e_engs.arr[:4])
        log.dbg("h2e_anti.getEigEngs=", engs.arr[:4])
        print('err_eng = ', err_eng)
        exit(0)

        vecs = h2e.getEigVec()
        dbg(vecs.mtrx)
        log.dbg("h2e_anti =", MtrxDbgView(vecs))
        tag = f'psi2e_ANTI_idx{cfg.plot_evec_idx}'
        if plot_on:
            self.build_and_plot_wf_2e(vecs, tag=tag)

        # todo:
        vec_idx = cfg.plot_evec_idx
        psi2e_grid = self.build_r1r2(vecs.mtrx[:, vec_idx])
        dbg(psi2e_grid)
        # fpath = f'psi2e_ANTI_{cfg.wf_saved_label_fpath}_vec{vec_idx}.npy'
        # np.save(fpath, psi2e_grid)
        psi_swap, anti_err = self.swap_psi_r1_r2(psi2e_grid, check_err_sign=1)
        dbg([psi_swap, anti_err])
        max_anti_err = np.max(np.abs(anti_err))
        dbg('max_anti_err')

        if plot_on:
            H1d1e.plot_wf_2e(psi2e_grid, title='psi2e_grid')
            H1d1e.plot_wf_2e(psi_swap, title='psi_swap')
            H1d1e.plot_wf_2e(anti_err, title=f'max_anti_err={max_anti_err}')

        dbg('engs.arr[0]')
        # dbg('self.cfg.eng_1d2e_anti')
        dbg('self.cfg.eng_2d2e_anti')
        log.dbg(f"M={cfg.m_2d} h2x1e_anti.getEigEngs=", h2x1e_engs.arr[:5])
        log.dbg(f"M={cfg.m_2d} h2e_anti.getEigEngs=", engs.arr[:5])
        assert abs(engs.arr[0] - self.cfg.eng_2d2e_anti) < 1e-4, 'quick check'
        return engs.arr[0]


    def init_all(self):
        super().init_all()  # most of them are the same
        cfg = self.cfg
        wLcr1 = cast(WFQuadrLcr, cfg.wLcr1)
        wLcr2 = cast(WFQuadrLcr, cfg.wLcr2)
        self._vX1 = wLcr1.getX()
        self._vX2 = wLcr2.getX()
        self._vR1 = wLcr1.getR()
        self._vR2 = wLcr2.getR()
        self._sysPotHLcr1 = PotHLcr(wLcr1)
        self._sysPotHLcr2 = PotHLcr(wLcr2)

        from qm_math.func.simple.FuncPowAbsInt import FuncPowAbsInt
        potFunc = FuncPowAbsInt(-cfg.atom_z, -1)  # // f(r)=-1./r
        # from d2_py.FuncAtom2DLn import FuncAtom2DLn
        # potFunc = FuncAtom2DLn(z=-cfg.atom_z)  # // f(r)=-2*Z*Ln(r)
        from qm_math.func.FuncVec import FuncVec
        self._vPot1 = FuncVec(self._vR1, potFunc)
        self._vPot2 = FuncVec(self._vR2, potFunc)
        log.dbg("-2 Z log(r1)=", self._vPot1)
        log.dbg("-2 Z log(r)2=", self._vPot2)

        cr2w1 = wLcr1.getWithCR2().arr
        cr2w2 = wLcr2.getWithCR2().arr
        W2d = cr2w1[:, None] * cr2w2[None, :]
        dbg([cr2w1, cr2w2, W2d])
        eps = cfg.vee_1r12_eps
        rr1 = self._r1[:, None]
        rr2 = self._r2[None, :]
        dbg([rr1, rr2])
        # self._V2e_grid = W2d / np.maximum(np.abs(r12), eps)
        # V2e_grid = W2d / np.maximum(np.abs(rr1 - rr2), eps)  # for 1/r12
        # V2e_grid = np.maximum(np.abs(rr1 - rr2), eps)
        V2e_grid = np.maximum(rr1, rr2)   # r_>
        dbg(V2e_grid)
        V2e_grid = np.maximum(V2e_grid, eps)    # before calling log(r12)
        dbg(V2e_grid)
        # V2e_grid = W2d * (-2) * np.log(V2e_grid)  # Vee,2d = -2 log(r_>)
        V2e_grid = W2d / V2e_grid  # Vee,3d = 1/(r_>)

        # (1001, 1001) float64 min= 4.450980923097418e-20 mean= 241234949.12224776 max= 26496613843859.445
        dbg(V2e_grid)

        # from d2_py.tests.Vee2D_Kw_Test import calc_2d_1over_r12_np
        # dbg(self._V2e_grid)
        # 2d case for 1/r12
        # calc_2d_1over_r12_np = Int_0^pi 1/\sqrt{r_1^2 + r_2^2 - 2 r_1 r_2 \cos \phi},
        # # todo BUT. Vee = 2/(2 \pi) * that Int
        # K_m_to_vee = 1 / np.pi
        # self._V2e_grid = K_m_to_vee * W2d * calc_2d_1over_r12_np(rr1, rr2, eps)
        self._V2e_grid = V2e_grid

        # (1001, 1001) float64 min= 5.518972608507955e-20 mean= 4.411136120413659 max= 19242.829793236764
        dbg(self._V2e_grid)
        log.dbg("_V2e_grid =", MtrxDbgView(Mtrx(data=self._V2e_grid)))


if __name__ == "__main__":
    # from _new25.v250712_H1d1e_diag_OK.v250714c_He1d2eDiagLcr_TEST import He1d2eDiagLcr_Test
    # from _new25.v250715_He1d.v250716b_He1d_LgrrLcrAnti_Test import He1d_LgrrLcrAnti_Test
    # He1d_LgrrLcrAnti_Test().test_1()

    # from _new25.v250715_He1d_submitted250802.v250717a_He1dPaper_LgrrLcrAnti_Test import He1d_LgrrLcrAnti_Test
    from _new25.v260614_He_2d_Gauss.v260627a_He3d_LcrAnti_run import He3d_LcrAnti_run
    He3d_LcrAnti_run().test_1()
