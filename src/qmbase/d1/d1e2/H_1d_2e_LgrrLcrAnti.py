# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : He1d2e_LgrrLcr.py Created : 2025-07-15 at 6:17 pm by Dmitry.A.Konovalov@gmail.com
import numpy as np
from typing import cast

from _new25.dbg import set_dbg, dbg
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from d1.d1e2.FuncArr1d2e import FuncArr1d2e, FuncArr1d2e_item
from d1.d1e2.H_1d_2e_LgrrLcr import H_1d_2e_LgrrLcr
from d1.d1e2.H_1d_2e_LgrrLcrSym import fix_sym
from d2_py.FuncArr2D import FuncArr2D_item
from javax.utilx.log.Log import Log
from qm_math.mtrx.MtrxDbgView import MtrxDbgView
from atom.energy.HMtrx import HMtrx, hermitian_error
from d1.d1e1.H1d1e import H1d1e

log = Log.getLog('H1d2e_LgrrLcrAnti')
# DBG_ON = True
DBG_ON = False
log.setDbg(DBG_ON)

class H_1d_2e_LgrrLcrAnti(H_1d_2e_LgrrLcr):
    def __init__(self, cfg):
        super().__init__(cfg)  # Copy from Vec
        # log.setDbg(False)

    def diag_on_2orth1(self, ASSUME_SYMM=True):
        cfg = self.cfg
        plot_on = cfg.plot_on
        # todo: remember! ASSUME_SYMM=True can only be used if orth1 and orth2 are identical
        h2x1e, pot2e = self.load_mats(ASSUME_SYMM=ASSUME_SYMM)
        dbg(h2x1e.mtrx)
        dbg(pot2e.mtrx)
        herm_err = hermitian_error(h2x1e)
        assert abs(herm_err) < 1e-12
        herm_err = hermitian_error(pot2e)
        assert abs(herm_err) < 1e-12

        # todo: one place needed for wf figs
        cfg.ok_1e_diag = False  # new 2060710
        cfg.ex_sign = -1

        # if is_symm:  # symmetric
        #     from d1.d1e2.H_1d_2e_LgrrLcrSym import fix_sym
        #     h2x1e, pot2e = fix_sym(h2x1e, pot2e, orth2e=self.cfg.orth2e, ok_1e_diag=True, ex_sign=1)
        # else:
        #     from d1.d1e2.H_1d_2e_LgrrLcrSym import fix_sym
        #     h2x1e, pot2e = fix_sym(h2x1e, pot2e, orth2e=self.cfg.orth2e, ok_1e_diag=False, ex_sign=-1)

        # todo:  1D H^(0)-2e --------------------------
        h2x1e, pot2e = fix_sym(h2x1e, pot2e, orth2e=self.cfg.orth2e, ok_1e_diag=cfg.ok_1e_diag, ex_sign=cfg.ex_sign)
        engs = h2x1e.getEigEngs()
        vecs = h2x1e.getEigVec()
        vecs_2x1e = vecs
        np.set_printoptions(precision=6)
        wLcr1 = cast(WFQuadrLcr, cfg.wLcr1)
        print(f'wLcr1:', wLcr1.get_info())
        print(f'ANTI h2x1e.engs(orth1N={cfg.orth1N}) =', engs.arr[:5])
        print(f'ANTI eng_2x1d1e_anti={cfg.eng_2x1d1e_anti}')
        err_eng = abs(engs.arr[0] - cfg.eng_2x1d1e_anti)
        print('err_eng_2x1e = ', err_eng)
        eng_err = abs(engs.arr[0] - self.cfg.eng_2x1e)
        assert eng_err < cfg.max_err_2x1e, f'quick check: eng_err={eng_err}, max_err={cfg.max_err_2x1e}'

        tag = f'psi2x1e_ANTI_idx{cfg.plot_evec_idx}_N{cfg.orth1N}'
        self.build_and_plot_wf_2e_onlyColors(vecs, tag=tag)

        # todo:  1D H-2e --------------------------
        h2e = h2x1e.mtrx + pot2e.mtrx
        dbg(h2e)
        h2e = HMtrx(mh=h2e)
        engs = h2e.getEigEngs()
        print(f'ANTI h2e_engs(orth1N={cfg.orth1N}) =', engs.arr[:5])
        eng2e_diff = engs.arr - cfg.eng_1d1e_n1
        print(f'eng_1d1e_n1(Z={cfg.atom_z}) =', cfg.eng_1d1e_n1)
        print('ANTI Delta(h2e_engs - eng_1d1e_n1) =', eng2e_diff[:5])
        print('cfg.eng_2e_check=', cfg.eng_2e)
        eng_err = abs(engs.arr[0] - cfg.eng_2e)
        assert eng_err < cfg.max_err_2e, f'quick check: eng_err={eng_err}, max_err={cfg.max_err_2e}'
        if not cfg.plot_on:
            return

        vecs = h2e.getEigVec()
        vecs_2e = vecs
        dbg(vecs.mtrx)
        log.dbg("h2e_anti =", MtrxDbgView(vecs))
        tag = f'psi2e_ANTI_idx{cfg.plot_evec_idx}_N{cfg.orth1N}'
        # self.build_and_plot_wf_2e(vecs, tag=tag)
        self.build_and_plot_wf_2e_onlyColors(vecs, tag=tag)

        tag = f'psi2x1e_and_2e_ANTI_idx{cfg.plot_evec_idx}_N{cfg.orth1N}'
        self.build_and_plot_wf_2x1e_AND_2e_onlyColors(vecs_2x1e=vecs_2x1e, vecs_2e=vecs_2e, tag=tag)

        tag = f'psi2e_SYMM_from_ANTI_idx{cfg.plot_evec_idx}_orth1N{cfg.orth1N}'
        self.build_and_plot_wf_2e_forceSymm(vecs, tag=tag)

        tag = f'psi2e_ANTI_1s3s_1s4s_orth1N{cfg.orth1N}'
        self.build_and_plot_wf_2e_1s3s_and_1s4s_onlyColors(vecs_2e=vecs_2e, tag=tag)

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

        # if plot_on:
        #     H1d1e.plot_wf_2e(psi2e_grid, title='psi2e_grid')
        #     H1d1e.plot_wf_2e(psi_swap, title='psi_swap')
        #     H1d1e.plot_wf_2e(anti_err, title=f'max_anti_err={max_anti_err}')

        return engs.arr[0]


# if __name__ == "__main__":
#     from _new25.v250812_He1Dp.v250715_He1d_submitted250802 import He1d_LgrrLcrAnti_Test
#     He1d_LgrrLcrAnti_Test().test_1()
