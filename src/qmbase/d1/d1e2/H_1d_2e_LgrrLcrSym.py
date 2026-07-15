# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : He1d2e_LgrrLcr.py Created : 2025-07-15 at 6:17 pm by Dmitry.A.Konovalov@gmail.com
import numpy as np
from typing import cast

from _new25.dbg import set_dbg, dbg
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from d1.d1e2.FuncArr1d2e import FuncArr1d2e, FuncArr1d2e_item
from d1.d1e2.H_1d_2e_LgrrLcr import H_1d_2e_LgrrLcr
from javax.utilx.log.Log import Log
from qm_math.mtrx.MtrxDbgView import MtrxDbgView
from atom.energy.HMtrx import HMtrx, hermitian_error

log = Log.getLog('H1d2e_LgrrLcrSym')
# DBG_ON = True
DBG_ON = False
log.setDbg(DBG_ON)


def fix_sym(h2x1e_mat, pot2e_mat, orth2e, ok_1e_diag, ex_sign):
    # return fix_sym_v1_using_orth1(h2x1e_mat, pot2e_mat, orth2e)
    return fix_sym_v2_only_orth2e(h2x1e_mat, pot2e_mat, orth2e, ok_1e_diag, ex_sign)

def fix_sym_v2_only_orth2e(h2x1e_mat, pot2e_mat, orth2e, ok_1e_diag, ex_sign):
    ASSUME_SYMM = True
    orth2e = cast(FuncArr1d2e, orth2e)
    # new_n = (n - nb1) // 2 + nb1  # for symm
    # new_n = (n - nb1) // 2   # for anti  #todo
    used_rows = {}
    for _, row in enumerate(orth2e.items):
        row = cast(FuncArr1d2e_item, row)
        if row.i1 == row.i2 and not ok_1e_diag:
            continue
        in_order = sorted([row.i1, row.i2])
        key_new_row = FuncArr1d2e.make_key(*in_order) # (i, j)
        if key_new_row in used_rows:
            continue  # already done
        used_rows[key_new_row] = row
    new_n = len(used_rows)

    h2x1e = h2x1e_mat.mtrx
    pot2e = pot2e_mat.mtrx
    h2x1e_sym = np.zeros((new_n, new_n), dtype=float)
    pot2e_sym = np.zeros((new_n, new_n), dtype=float)

    # NEW_DBG = True
    NEW_DBG = False
    saved_dbg = set_dbg(NEW_DBG)
    log.setDbg(NEW_DBG)
    map_to_idx = orth2e.map_ij_to_idx

    used_rows = {}
    new_r_di = -1    # for new_r_di, (i1, j1) in enumerate(zip(i_arr, j_arr)):
    for _, row in enumerate(orth2e.items):
        row = cast(FuncArr1d2e_item, row)
        if row.i1 == row.i2 and not ok_1e_diag:
            continue
        i1, j1 = row.i1, row.i2
        key_i1_j1 = FuncArr1d2e.make_key(i1, j1)  # direct
        key_j1_i1 = FuncArr1d2e.make_key(j1, i1)  # exchange
        r_di = map_to_idx[key_i1_j1]      # r_di = map_to_idx[(i1, j1)]
        r_ex = map_to_idx[key_j1_i1]
        row_diag = (i1 == j1)   # row_diag = (i1 == j1)

        in_order = sorted([i1, j1])
        key_new_row = FuncArr1d2e.make_key(*in_order) # (i, j)
        if key_new_row in used_rows:
            continue  # already done
        used_rows[key_new_row] = row
        new_r_di += 1

        used_cols = {}
        new_c_di = -1    # for new_c_di, (i2, j2) in enumerate(zip(i_arr, j_arr)):
        for _, col in enumerate(orth2e.items):
            col = cast(FuncArr1d2e_item, col)
            if col.i1 == col.i2 and not ok_1e_diag:
                continue
            i2, j2 = col.i1, col.i2
            key_i2_j2 = FuncArr1d2e.make_key(i2, j2) #
            key_j2_i2 = FuncArr1d2e.make_key(j2, i2) #
            c_di = map_to_idx[key_i2_j2]         # c_di = map_to_idx[(i2, j2)]

            # todo: NOTE!!! BUG in fix_sym_v2_only_orth2e for orth2N != orth1N
            c_ex = map_to_idx[key_j2_i2]
            #  c_ex = map_to_idx[key_j2_i2]
            # KeyError: '(10, 0)'

            col_diag = (i2 == j2)      # col_diag = (i2 == j2)

            in_order = sorted([i2, j2])
            key_NEW_col = FuncArr1d2e.make_key(*in_order) # (i, j)
            if key_NEW_col in used_cols:
                continue  # already done
            used_cols[key_NEW_col] = col
            new_c_di += 1

            if row_diag and col_diag:
                # G_ii vs G_jj
                val = h2x1e[r_di, c_di]
                pot = pot2e[r_di, c_di]
            elif row_diag and not col_diag:
                # G_ii vs G_mn (m<n)
                val = (h2x1e[r_di, c_di] + ex_sign * h2x1e[r_di, c_ex]) / np.sqrt(2.0)
                pot = (pot2e[r_di, c_di] + ex_sign * pot2e[r_di, c_ex]) / np.sqrt(2.0)
            elif (not row_diag) and col_diag:
                # G_mn vs G_ii
                val = (h2x1e[r_di, c_di] + ex_sign * h2x1e[r_ex, c_di]) / np.sqrt(2.0)
                pot = (pot2e[r_di, c_di] + ex_sign * pot2e[r_ex, c_di]) / np.sqrt(2.0)
            else:
                # both off-diagonal
                val = (
                              h2x1e[r_di, c_di] +
                              h2x1e[r_di, c_ex] * ex_sign +
                              h2x1e[r_ex, c_di] * ex_sign +
                              h2x1e[r_ex, c_ex]) * 0.5  # (= 1/(√2 * √2))
                pot = (
                              pot2e[r_di, c_di] +
                              pot2e[r_di, c_ex] * ex_sign +
                              pot2e[r_ex, c_di] * ex_sign +
                              pot2e[r_ex, c_ex]) * 0.5  # (= 1/(√2 * √2))

            h2x1e_sym[new_r_di, new_c_di] = val
            h2x1e_sym[new_c_di, new_r_di] = val  # enforce symmetry
            pot2e_sym[new_r_di, new_c_di] = pot
            pot2e_sym[new_c_di, new_r_di] = pot  # enforce symmetry

    set_dbg(saved_dbg)
    log.setDbg(saved_dbg)
    h2x1e_sym = HMtrx(mh=h2x1e_sym)
    pot2e_sym = HMtrx(mh=pot2e_sym)
    dbg(h2x1e_sym.mtrx)
    dbg(h2x1e_sym.mtrx)
    log.dbg("h2x1e_sym =", MtrxDbgView(h2x1e_sym))
    log.dbg("pot2e_sym =", MtrxDbgView(pot2e_sym))
    return h2x1e_sym, pot2e_sym


class H_1d_2e_LgrrLcrSym(H_1d_2e_LgrrLcr):
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
        cfg.ok_1e_diag = True  # new 2060710
        cfg.ex_sign = 1

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
        print(f'SYM h2x1e.engs(orth1N={cfg.orth1N}) =', engs.arr[:5])
        print(f'SYM eng_2x1d1e={cfg.eng_2x1d1e_symm}')
        err_eng = abs(engs.arr[0] - cfg.eng_2x1d1e_symm)
        print('err_eng_2x1e = ', err_eng)
        eng_err = abs(engs.arr[0] - self.cfg.eng_2x1e)
        assert eng_err < cfg.max_err_2x1e, f'quick check: eng_err={eng_err}, max_err={cfg.max_err_2x1e}'

        tag = f'psi2x1e_SYM_idx{cfg.plot_evec_idx}_N{cfg.orth1N}'
        self.build_and_plot_wf_2e_onlyColors(vecs, tag=tag)

        # todo:  1D H-2e --------------------------
        h2e = h2x1e.mtrx + pot2e.mtrx
        dbg(h2e)
        h2e = HMtrx(mh=h2e)
        engs = h2e.getEigEngs()
        print(f'SYM h2e_engs(orth1N={cfg.orth1N}) =', engs.arr[:5])
        eng2e_diff = engs.arr - cfg.eng_1d1e_n1
        print(f'eng_1d1e_n1(Z={cfg.atom_z}) =', cfg.eng_1d1e_n1)
        print('SYM Delta(h2e_engs - eng_1d1e_n1) =', eng2e_diff[:5])
        print('cfg.eng_2e_check=', cfg.eng_2e)
        eng_err = abs(engs.arr[0] - cfg.eng_2e)
        assert eng_err < cfg.max_err_2e, f'quick check: eng_err={eng_err}, max_err={cfg.max_err_2e}'
        if not cfg.plot_on:
            return

        vecs = h2e.getEigVec()
        vecs_2e = vecs
        dbg(vecs.mtrx)
        log.dbg("h2e_anti =", MtrxDbgView(vecs))
        tag = f'psi2e_SYM_idx{cfg.plot_evec_idx}_N{cfg.orth1N}'
        # self.build_and_plot_wf_2e(vecs, tag=tag)
        self.build_and_plot_wf_2e_onlyColors(vecs, tag=tag)

        tag = f'psi2x1e_and_2e_SYM_idx{cfg.plot_evec_idx}_N{cfg.orth1N}'
        self.build_and_plot_wf_2x1e_AND_2e_onlyColors(vecs_2x1e=vecs_2x1e, vecs_2e=vecs_2e, tag=tag)

        # todo: read from given path: to compare SYM_from_ANTI
        tag = f'psi2e_SymFromAnti_and_SYM_idx{cfg.plot_evec_idx}_N{cfg.orth1N}'
        # todo: reuse build_and_plot_wf_2x1e_AND_2e_onlyColors:
        # load SYM_FROM_ANTI into vecs_sym_from_anti
        # todo: maybe generic?
        # fpath = cfg.current_dir_path / f'{tag}_{cfg.cfg_file_label}_psi2d_2x1e_idx{vec_idx}.npy'
        fpath = cfg.current_dir_path / cfg.npy_fpath_to_plot_compare
        print(fpath)
        psi2d_2e_from_anti = np.load(fpath)
        self.build_and_plot_wf_2e_SymFromAnti_and_Symm_onlyColors(
            psi2d_2e_from_anti=psi2d_2e_from_anti, vecs_2e=vecs_2e, tag=tag)

        # no point in symm:
        # tag = f'psi2e_SYMM_from_SYM_idx{cfg.plot_evec_idx}_orth1N{cfg.orth1N}'
        # self.build_and_plot_wf_2e_forceSymm(vecs, tag=tag)

        # todo:
        vec_idx = cfg.plot_evec_idx
        psi2e_grid = self.build_r1r2(vecs.mtrx[:, vec_idx])
        dbg(psi2e_grid)
        # fpath = f'psi2e_SYM_{cfg.wf_saved_label_fpath}_vec{vec_idx}.npy'
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
#     from _new25.v250812_He1Dp.v250715_He1d_submitted250802.v250716a_He1d_LgrrLcrSym_Test import He1d_LgrrLcrSym_Test
#
#     He1d_LgrrLcrSym_Test().test_1()
