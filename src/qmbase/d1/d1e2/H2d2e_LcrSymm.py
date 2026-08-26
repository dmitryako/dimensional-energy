# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : He1d2e_LgrrLcr.py Created : 2025-07-15 at 6:17 pm by Dmitry.A.Konovalov@gmail.com
import numpy as np
from typing import cast

from _new25.dbg import set_dbg, dbg
from d1.d1e2.H2d2e_LcrAnti import H2d2e_LcrAnti
from javax.utilx.log.Log import Log
from qm_math.mtrx.MtrxDbgView import MtrxDbgView
from atom.energy.HMtrx import HMtrx
from d1.d1e1.H1d1e import H1d1e

log = Log.getLog('H2d2e_LcrSymm')


# todo: old, make (H1d2e_LgrrLcr) to make sure not using anthing from H1d2e_LgrrLcrSym
class H2d2e_LcrSymm(H2d2e_LcrAnti):
    # todo: the same as H1d2e_LgrrLcrAnti

    def diag_on_2orth1(self, sign=-1):
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
        h2x1e, pot2e = fix_sym(h2x1e, pot2e, orth2e=self.cfg.orth2e, ok_1e_diag=True, ex_sign=1)
        dbg([h2x1e.mtrx, pot2e.mtrx])

        h2x1e_engs = h2x1e.getEigEngs()
        vecs = h2x1e.getEigVec()
        log.dbg("h2x1e_symm.getEigEngs=", h2x1e_engs)
        dbg('self.cfg.eng_2x2d1e_symm')
        err_eng = h2x1e_engs.arr[0] - cfg.eng_2x2d1e_symm
        print('err_eng = ', err_eng)
        print('h2x1e_symm.engs', h2x1e_engs.arr[:4])
        assert abs(err_eng) < 1e-1, 'quick check'
        tag = f'psi2x1e_ANTI_idx{cfg.plot_evec_idx}'

        if plot_on:
            self.build_and_plot_wf_2e(vecs, tag=tag, sign=sign)

        # todo: FULL e+e
        h2e = h2x1e.mtrx + pot2e.mtrx
        dbg(h2e)
        h2e = HMtrx(mh=h2e)
        engs = h2e.getEigEngs()
        print("h2x1e_symm.getEigEngs=", h2x1e_engs.arr[:4])
        print("h2e_symm.getEigEngs=", engs.arr[:4])
        log.dbg("h2x1e_symm.getEigEngs=", h2x1e_engs.arr[:4])
        log.dbg("h2e_symm.getEigEngs=", engs.arr[:4])
        print('err_eng = ', err_eng)
        exit(0)

        vecs = h2e.getEigVec()
        # (190, 190) float64 min= -0.424073336447102 mean= -0.0001675790581064721 max= 0.4093013160525687
        dbg(vecs.mtrx)
        log.dbg("h2e_symm =", MtrxDbgView(vecs))
        tag = f'psi2e_SYMM_idx{cfg.plot_evec_idx}'
        self.build_and_plot_wf_2e(vecs, tag=tag)

        # todo:
        vec_idx = cfg.plot_evec_idx
        psi2e_grid = self.build_r1r2(vecs.mtrx[:, vec_idx])
        diag_cii = self.cfg.diag_cii
        dbg('diag_cii')
        dbg(psi2e_grid)
        psi_swap, swap_err = self.swap_psi_r1_r2(psi2e_grid, check_err_sign=-1)
        dbg([psi_swap, swap_err])
        max_anti_err = np.max(np.abs(swap_err))
        dbg('swap_err')

        if plot_on:
            H1d1e.plot_wf_2e(psi2e_grid, title='psi2e_grid')
            H1d1e.plot_wf_2e(psi_swap, title='psi_swap')
            H1d1e.plot_wf_2e(swap_err, title=f'max_anti_err={max_anti_err}')

        dbg('diag_cii')
        dbg('engs.arr[0]')
        # dbg('self.cfg.eng_1d2e_anti')
        dbg('self.cfg.eng_2d2e_symm')
        log.dbg("h2x1e_symm.getEigEngs=", h2x1e_engs.arr[:5])
        log.dbg("h2e_symm.getEigEngs=", engs.arr[:5])
        assert abs(engs.arr[0] - self.cfg.eng_2d2e_symm) < 1e-4, 'quick check'
        return engs.arr[0]


if __name__ == "__main__":
    # from _new25.v250712_H1d1e_diag_OK.v250714c_He1d2eDiagLcr_TEST import He1d2eDiagLcr_Test
    # from _new25.v250715_He1d.v250716b_He1d_LgrrLcrAnti_Test import He1d_LgrrLcrAnti_Test
    # He1d_LgrrLcrAnti_Test().test_1()

    # from _new25.v250715_He1d_submitted250802.v250717a_He1dPaper_LgrrLcrAnti_Test import He1d_LgrrLcrAnti_Test
    # from _new25.v250726_2d_anyZ.v250802b_He2dPaper_LcrSymm_run import He2d_LcrSymm_run
    # He2d_LcrSymm_run().test_1()

    from _new25.v260614_He_2d_Gauss.v260625a_He2d_LcrSymm_rerun import He2d_LcrSymm_rerun
    He2d_LcrSymm_rerun().test_1()

