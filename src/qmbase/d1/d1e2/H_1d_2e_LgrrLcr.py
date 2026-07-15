# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : He1d2e_LgrrLcr.py Created : 2025-07-15 at 6:17 pm by Dmitry.A.Konovalov@gmail.com
import os
from os.path import dirname

from tqdm import tqdm
from typing import cast
import numpy as np

from _new25.dbg import set_dbg, dbg
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from d1.d1e1.H1d1e import H1d1e
from d1.d1e2.FuncArr1d2e import FuncArr1d2e, FuncArr1d2e_item
from d1.d1e1.H1d1e_Lcr import H1d1e_Lcr
from javax.utilx.log.Log import Log
from qm_math.func.deriv.DerivFactory import DerivFactory
from qm_math.mtrx.MtrxDbgView import MtrxDbgView
from qm_math.mtrx.api.Mtrx import Mtrx
from qm_math.vec.Vec import Vec
from atom.energy.HMtrx import HMtrx, hermitian_error
from scatt.jm_2008.jm.laguerre.IWFuncArr import IWFuncArr

log = Log.getLog('H_1d_2e_LgrrLcr')
DBG_ON = True

class H_1d_2e_LgrrLcr(H1d1e_Lcr):
    def __init__(self, cfg):
        self.cfg = cfg
        # self._orth2e = cast(FuncArr1d2e, cfg.orth2e)
        # log.setDbg(DBG_ON)
        self.init_all()
        self._map_H12 = {}

    def diag_on_2orth1(self, ASSUME_SYMM=True):
        cfg = self.cfg
        # todo: remember! ASSUME_SYMM=True can only be used if orth1 and orth2 are identical
        h2x1e, pot2e = self.load_mats(ASSUME_SYMM=ASSUME_SYMM)
        dbg(h2x1e.mtrx)
        dbg(pot2e.mtrx)
        herm_err = hermitian_error(h2x1e)
        assert abs(herm_err) < 1e-12
        herm_err = hermitian_error(pot2e)
        assert abs(herm_err) < 1e-12

        engs = h2x1e.getEigEngs()
        log.dbg("h2x1e.getEigEngs=", engs)
        np.set_printoptions(precision=6)
        print('h2x1e.getEigEngs =', engs.arr[:5])
        dbg('self.cfg.eng_2x1d1e')
        eng_err = abs(engs.arr[0] - self.cfg.eng_2x1e)
        assert eng_err < cfg.max_err_2x1e, f'quick check: eng_err={eng_err}, max_err={cfg.max_err_2x1e}'

        h2e = h2x1e.mtrx + pot2e.mtrx
        dbg(h2e)
        h2e = HMtrx(mh=h2e)
        engs = h2e.getEigEngs()
        log.dbg("h2e.getEigEngs=", engs)
        print('h2e.getEigEngs =', engs.arr[:5])
        eng_err = abs(engs.arr[0] - cfg.eng_2e)
        assert eng_err < cfg.max_err_2e, f'quick check: eng_err={eng_err}, max_err={cfg.max_err_2e}'
        return engs.arr[0]

    # def load_mats(self):
    def load_mats(self, ASSUME_SYMM=True):
        # todo: remember! ASSUME_SYMM=True can only be used if orth1 and orth2 are identical
        from d2_py.FuncArr2D import FuncArr2D_item
        orth2e = cast(FuncArr1d2e, self.cfg.orth2e)
        # orth2e = cast(FuncArr1d2e, self._orth2e)
        n = orth2e.size()
        # h2x1e = HMtrx(rows=nb1, cols=nb2)
        h2x1e = HMtrx(rows=n, cols=n)
        pot2e = HMtrx(rows=n, cols=n)

        # NEW_DBG = True
        NEW_DBG = False
        saved_dbg = set_dbg(NEW_DBG)
        log.setDbg(NEW_DBG)
        # map_H1 = {}
        # count1 = 0
        for row_idx, row in tqdm(enumerate(orth2e.items), total=len(orth2e.items), desc="load_mats rows"):
            row = cast(FuncArr2D_item, row)
            #  item = FuncArr2D_item(i1=i, i2=j, fv1=wf1, fv2=wf2)
            row_f1 = row.fv1.arr
            row_f2 = row.fv2.arr
            dbg([row_f1, row_f2])
            # for col_idx, col in tqdm(enumerate(orth2e.items)):
            for col_idx, col in enumerate(orth2e.items):
                if ASSUME_SYMM and col_idx > row_idx:
                    continue
                col = cast(FuncArr2D_item, col)
                col_f1 = col.fv1.arr
                col_f2 = col.fv2.arr
                dbg([col_f1, col_f2])

                h1 = 0.
                if row.i2 == col.i2:  # i2 is e2, must be the same, but orth
                    h1 = self.calc_H1_with_map(fa=row_f1, fb=col_f1, row_i1=row.i1, col_i1=col.i1)
                    # h1_old = self.calc_H1(fa=row_f1, fb=col_f1)
                    # assert h1 == h1_old

                h2 = 0.
                if row.i1 == col.i1:  # i1 is e1, must be the same, but orth
                    h2 = self.calc_H2_with_map(fa=row_f2, fb=col_f2, row_i2=row.i2, col_i2=col.i2)
                    # h2_old = self.calc_H2(fa=row_f2, fb=col_f2)
                    # assert h2 == h2_old
                h1e = h1 + h2
                h2x1e.set(r=row_idx, c=col_idx, val=h1e)

                v = self.calc_pot2e(row1=row_f1, row2=row_f2, col1=col_f1, col2=col_f2)
                pot2e.set(r=row_idx, c=col_idx, val=v)
                if ASSUME_SYMM:
                    h2x1e.set(r=col_idx, c=row_idx, val=h1e)
                    pot2e.set(r=col_idx, c=row_idx, val=v)

        dbg(h2x1e.mtrx)
        dbg(pot2e.mtrx)
        set_dbg(saved_dbg)
        log.setDbg(saved_dbg)
        log.dbg("h2x1e =", MtrxDbgView(h2x1e))
        log.dbg("pot2e =", MtrxDbgView(pot2e))
        return h2x1e, pot2e

    def swap_psi_r1_r2(self, psi_grid, check_err_sign):
        # return self.swap_psi_r1_r2_v1(psi_grid=psi_grid, check_err_sign=check_err_sign)
        return self.swap_psi_r1_r2_v2(psi_grid=psi_grid, check_err_sign=check_err_sign)

    def swap_psi_r1_r2_v2(self, psi_grid, check_err_sign):
        # psi_diag : ndarray, shape (n_common,)
        #     ψ(r,r) along the diagonal (uses interpolation if n1≠n2).
        # psi_swap : ndarray, shape (n2,n1)
        #     ψ(r2,r1) on the full swapped grid.
        # swap_err : ndarray, same shape as psi_grid
        #     ψ(r1,r2) + sym_sign * ψ(r2,r1);  should be ≈0 for the
        #     required symmetry (sym_sign=+1 for symmetric, –1 for antisymmetric).
        # import numpy as np
        from scipy.interpolate import RectBivariateSpline

        r1, r2 = self._x1, self._x2  #
        interp = RectBivariateSpline(r1, r2, psi_grid)
        dbg([r1, r2, psi_grid])  # (1001, 1101)

        # ---------- diagonal  ψ(r,r) -------------------------------------------
        # Work on the overlap interval so we never extrapolate:
        r_common = r1 if len(r1) <= len(r2) else r2  # the shorter grid
        # Evaluate ψ(r,r) point‑wise with fast .ev:
        psi_diag = np.array([interp.ev(r, r) for r in r_common])
        dbg(psi_diag)  #
        self.cfg.psi_diag = psi_diag
        max_psi_diag = np.max(np.abs(psi_diag))
        dbg('max_psi_diag')
        self.cfg.max_psi_diag = max_psi_diag

        # ---------- swapped grid  ψ(r2,r1) -------------------------------------
        psi_swap = interp(r2, r1)  # shape (n2,n1)
        dbg(psi_swap)  #
        psi_swap = np.moveaxis(psi_swap, source=1, destination=0)
        dbg(psi_swap)  #

        # ---------- symmetry / antisymmetry error ------------------------------
        swap_err = psi_grid + check_err_sign * psi_swap  # + for sym, – for anti
        dbg(swap_err)  #
        max_swap_err = np.max(np.abs(swap_err))
        dbg('max_swap_err')
        self.cfg.max_swap_err = max_swap_err

        return psi_swap, swap_err  # keep as v1

    def swap_psi_r1_r2_v1(self, psi_grid, check_err_sign):
        from scipy.interpolate import RectBivariateSpline
        # r1: shape (n1,)
        # r2: shape (n2,)
        # psi_grid: shape (n1, n2)
        # Interpolate psi(r1, r2)
        r1 = self._x1
        r2 = self._x2
        dbg([r1, r2, psi_grid])  # (1001, 1101)
        interp = RectBivariateSpline(r1, r2, psi_grid)
        # Evaluate psi(r2, r1) on the swapped grid
        # This gives a new array of shape (n2, n1)
        psi_swap = interp(r2, r1)  # careful: order matters!
        dbg(psi_swap)  # (1101, 1001)
        psi_swap = np.moveaxis(psi_swap, source=1, destination=0)
        dbg(psi_swap)  # (1101, 1001)
        # check_err_sign = +1 for anti
        # check_err_sign = -1 for symm
        swap_err = psi_grid + check_err_sign * psi_swap
        return psi_swap, swap_err

    def calc_kin(self, fa, *, fb, D1, wx):
        dbg([fa, fb, D1, wx])
        a1 = D1 @ fa
        b1 = D1 @ fb
        wx_b1 = wx * b1
        dbg([a1, b1, wx_b1])
        kin05 = (-(-0.5)) * a1.T @ wx_b1
        dbg('kin05')
        kin8 = ((-0.5) * (-0.25)) * fa.T @ (wx * fb)
        dbg('kin8')
        kin1e = kin05 + kin8
        dbg('kin1e')
        # B            B               |B
        # I dr RR" = - I dr R'R' + RR' |  = - I drR'R' + RR'(B) - RR'(A)
        # A            A               |A
        corrA = fa[0] * b1[0]
        corrB = fa[-1] * b1[-1]
        dbg('corrA')
        dbg('corrB')
        assert abs(corrA) < 1e-8, f'check corrA={corrA}'
        assert abs(corrB) < 1e-8, f'check corrB={corrB}'
        return kin1e

    def calc_pot(self, fa, *, fb, pot):
        dbg([fa, fb, pot])
        pot1e = fa.T @ (pot * fb)
        dbg('pot1e')
        return pot1e

    def calc_H1_with_map(self, *, fa, fb, row_i1, col_i1):
        key1 = f'row_i1={row_i1}, col_i1={col_i1}'
        if key1 in self._map_H12:
            h1 = self._map_H12[key1]
        else:
            h1 = self.calc_H1(fa=fa, fb=fb)
            self._map_H12[key1] = h1
        return h1

    def calc_H2_with_map(self, *, fa, fb, row_i2, col_i2):
        key2 = f'row_i2={row_i2}, col_i2={col_i2}'
        if key2 in self._map_H12:
            h2 = self._map_H12[key2]
        else:
            h2 = self.calc_H2(fa=fa, fb=fb)
            self._map_H12[key2] = h2
        return h2

    def calc_H1(self, *, fa, fb):
        kin1e = self.calc_kin1(fa=fa, fb=fb)
        pot1e = self.calc_pot1(fa=fa, fb=fb)
        h1e = kin1e + pot1e
        dbg('h1e')
        return h1e

    def calc_H2(self, *, fa, fb):
        kin1e = self.calc_kin2(fa=fa, fb=fb)
        pot1e = self.calc_pot2(fa=fa, fb=fb)
        h1e = kin1e + pot1e
        dbg('h1e')
        return h1e

    def calc_kin1(self, *, fa, fb):
        return self.calc_kin(fa, fb=fb, D1=self._D1_x1, wx=self._wx1)
    def calc_kin2(self, *, fa, fb):
        return self.calc_kin(fa, fb=fb, D1=self._D1_x2, wx=self._wx2)
    def calc_pot1(self, *, fa, fb):
        return self.calc_pot(fa, fb=fb, pot=self._pot1)
    def calc_pot2(self, *, fa, fb):
        return self.calc_pot(fa, fb=fb, pot=self._pot2)

    def init_all(self):
        # test_1
        #         r1_max = 100.
        #         e1_to_e2_diff = 1.1
        #         r1_min = 1e-4
        #         x1_min = -4.
        #         nr1 = 701
        #         orth1N = 2
        #
        #         # e#2
        #         r2_max = r1_max * e1_to_e2_diff
        #         r2_min = r1_min * e1_to_e2_diff
        #         x2_min = x1_min * e1_to_e2_diff
        #         nr2 = nr1 + 100
        #         orth2N = orth1N + 1
        cfg = self.cfg
        #         cfg.w1Lcr = w1
        #         cfg.w2Lcr = w2
        #         orth1 = LgrrOrthLcr(w1, lgrrOpt1)
        #         orth2 = LgrrOrthLcr(w2, lgrrOpt2)
        wLcr1 = cast(WFQuadrLcr, cfg.wLcr1)
        wLcr2 = cast(WFQuadrLcr, cfg.wLcr2)

        self._wx1 = wLcr1.getY().arr
        self._wx2 = wLcr2.getY().arr
        dbg([self._wx1, self._wx2])

        self._x1 = wLcr1.getX().arr
        self._x2 = wLcr2.getX().arr
        # (701,) float64 min= -4.0 mean= 0.3026761628942591 max= 4.605352325788518
        # (801,) float64 min= -4.4 mean= 0.15029548592813827 max= 4.700590971856277
        dbg([self._x1, self._x2])
        # H1d2e_LgrrLcr: x1_grid = Vec[701] = {-4.00000, -3.98771, -3.97541, -3.96312, -3.95083, ..., 4.55618, 4.56847, 4.58077, 4.59306, 4.60535}
        # H1d2e_LgrrLcr: x2_grid = Vec[801] = {-4.40000, -4.38862, -4.37725, -4.36587, -4.35450, ..., 4.65509, 4.66646, 4.67784, 4.68922, 4.70059}
        log.dbg("x1_grid =", wLcr1.getX())
        log.dbg("x2_grid =", wLcr2.getX())

        # (701,) float64 min= 9.99999999999994e-05 mean= 11.657369041353835 max= 99.99999999999997
        # (801,) float64 min= 0.00011000000000000072 mean= 12.128667280372785 max= 110.00000000000003
        self._r1 = wLcr1.getR().arr
        self._r2 = wLcr2.getR().arr
        dbg([self._r1, self._r2])
        # H1d2e_LgrrLcr: r1_grid = Vec[701] = {0.00010, 0.00033, 0.00056, 0.00079, 0.00102, ..., 95.20073, 96.37851, 97.57086, 98.77797, 100.00000}
        # H1d2e_LgrrLcr: r2_grid = Vec[801] = {0.00011, 0.00025, 0.00039, 0.00054, 0.00068, ..., 105.10630, 106.30893, 107.52532, 108.75562, 110.00000}
        log.dbg("r1_grid =", wLcr1.getR())
        log.dbg("r2_grid =", wLcr2.getR())

        # self._cr2DivR1 = wLcr1.getCR2DivR().arr
        # self._cr2DivR2 = wLcr2.getCR2DivR().arr
        self._wCr2DivR1 = wLcr1.getWithCR2DivR().arr
        self._wCr2DivR2 = wLcr2.getWithCR2DivR().arr
        # (701,) float64 min= 0.07286259714283415 mean= 11.707876938546528 max= 100.03643459587245
        # (801,) float64 min= 0.048669484856272205 mean= 12.160297237314786 max= 110.02433602566218
        dbg([self._wCr2DivR1, self._wCr2DivR2])
        log.dbg("_wCr2DivR1 =", wLcr1.getWithCR2DivR())
        log.dbg("_wCr2DivR2 =", wLcr2.getWithCR2DivR())

        self._cr2w1 = wLcr1.getWithCR2().arr
        self._cr2w2 = wLcr2.getWithCR2().arr
        # Outer product → shape (N1, N2)
        self._W2d = self._cr2w1[:, None] * self._cr2w2[None, :]
        dbg([self._cr2w1, self._cr2w2, self._W2d])

        # r1_grid = wLcr1.getR().arr[:, None]
        # r2_grid = wLcr2.getR().arr[:, None]
        # V2e_grid = 1.0 / np.abs(r1_grid - r2_grid.T + 1e-10)
        # 2. Outer difference  r12 = r1[i] − r2[j]
        rr12 = np.subtract.outer(self._r1, self._r2)  # shape (N1, N2)
        # 3. Coulomb kernel  1/|r1−r2|   (add tiny ε so you never divide by 0)
        # eps = 1e-10  # or np.finfo(float).eps
        eps = cfg.vee_1r12_eps
        self._V2e_grid = self._W2d / np.maximum(np.abs(rr12), eps)
        dbg(self._V2e_grid)
        log.dbg("_V2e_grid =", MtrxDbgView(Mtrx(data=self._V2e_grid)))

        self._D1_x1 = DerivFactory.make_deriv_mtrx_11pt(self._x1)
        self._D1_x2 = DerivFactory.make_deriv_mtrx_11pt(self._x2)
        # D1 = DerivFactory.make_deriv_mtrx_9pt(x)
        # (701, 701) float64 min= -4270.597981031682 mean= 1.8919012468135872e-16 max= 4270.597981031682
        # (801, 801) float64 min= -4615.084936443587 mean= 1.841391175072382e-16 max= 4615.084936443587
        dbg([self._D1_x1, self._D1_x2])
        log.dbg("_D1_x1 =", MtrxDbgView(Mtrx(data=self._D1_x1)))
        log.dbg("_D1_x2 =", MtrxDbgView(Mtrx(data=self._D1_x2)))

        self._pot1 = -cfg.atom_z * self._wCr2DivR1  # todo: NOTE orig was 1/r, now it's * r
        self._pot2 = -cfg.atom_z * self._wCr2DivR2  # todo: NOTE orig was 1/r, now it's * r
        # (701,) float64 min= -200.0728691917449 mean= -23.415753877093056 max= -0.1457251942856683
        # (801,) float64 min= -220.04867205132436 mean= -24.320594474629573 max= -0.09733896971254441
        dbg([self._pot1, self._pot2])
        log.dbg("_pot1 =", Vec(self._pot1))
        log.dbg("_pot2 =", Vec(self._pot2))

    def calc_pot2e(self, *, row1, row2, col1, col2):
        f11 = row1 * col1
        f22 = row2 * col2
        dbg([f11, self._V2e_grid, f22])
        # pot2e = np.einsum('i,ij,j->', f11, self._V2e_grid, f22)
        # dbg('pot2e')
        # option B (nested @)
        pot2e = f11 @ (self._V2e_grid @ f22)  # or  f11.T @ V2e_grid @ f22
        dbg('pot2e')
        return pot2e

    # def build_r1r2(self, vec_coeff):
    #     raise NotImplementedError("build_r1r2() must be implemented in a subclass.")

    def build_r1r2(self, vec_coeff):
        """
        Reconstruct the antisymmetric 2e wavefunction on the (r1,r2) grid.
        Iterates orth2e.items with the SAME dedup and the SAME (i1,j1)
        direct/exchange convention as fix_sym_v2_only_orth2e, so that
        vec_coeff[idx] is paired with the correct A_{i1,j1}. (anti: ok_1e_diag=False)
        """
        cfg = self.cfg
        orth2e = cast(FuncArr1d2e, self.cfg.orth2e)
        B1 = orth2e.basis_1d  # electron-1 basis
        B2 = orth2e.basis2_1d  # electron-2 basis
        norm = 1.0 / np.sqrt(2.0)
        # ok_1e_diag = False  # antisymmetric: drop i1==i2 (matches fix_sym ex_sign=-1)
        # ok_1e_diag = True  # symmetric: keep i1==i2 (matches fix_sym ex_sign=+1)

        res = np.zeros_like(self._V2e_grid)
        used_rows = {}
        idx = -1
        for _, row in enumerate(orth2e.items):
            row = cast(FuncArr1d2e_item, row)
            if row.i1 == row.i2 and not cfg.ok_1e_diag:
                continue
            i1, j1 = row.i1, row.i2
            row_diag = (i1 == j1)  # row_diag = (i1 == j1)
            in_order = sorted([row.i1, row.i2])
            # key_new_row = FuncArr1d2e.make_key(*sorted([i1, j1]))  # canonical, dedup only
            key_new_row = FuncArr1d2e.make_key(*in_order)  # canonical, dedup only
            if key_new_row in used_rows:
                continue
            used_rows[key_new_row] = row
            idx += 1
            # A_{i1,j1} = (F_{i1,j1} - F_{j1,i1}) / sqrt2, using ORIGINAL (i1,j1)
            # exactly as fix_sym: r_di=(i1,j1) direct, r_ex=(j1,i1) exchange, ex_sign=-1
            b1_i = B1[i1].arr[:, None]  # B1_{i1}(r1)
            b2_j = B2[j1].arr[None, :]  # B2_{j1}(r2)
            b1_j = B1[j1].arr[:, None]  # B1_{j1}(r1)
            b2_i = B2[i1].arr[None, :]  # B2_{i1}(r2)
            if row_diag:
                fij = b1_i * b2_j
            else:
                fij = norm * (b1_i * b2_j + cfg.ex_sign * b1_j * b2_i)
            res += vec_coeff[idx] * fij

        assert idx + 1 == len(vec_coeff), f"count mismatch: built {idx + 1}, vec has {len(vec_coeff)}"
        self.cfg.diag_cii = 0
        return res



    def build_and_plot_wf_2e(self, vecs, *, tag, sign=1):
        cfg = self.cfg
        if not cfg.plot_on:
            return
        vec_idx = cfg.plot_evec_idx
        psi2d = self.build_r1r2(vecs.mtrx[:, vec_idx])
        # diag_cii = self.cfg.diag_cii
        dbg(psi2d)
        # fpath = cfg.current_dir_path / f'psi2x1e_ANTI_{cfg.cfg_file_label}_vec{vec_idx}.npy'
        fpath = cfg.current_dir_path / f'{tag}_{cfg.cfg_file_label}_vec{vec_idx}.npy'
        print(tag)
        print(fpath)
        os.makedirs(dirname(fpath), exist_ok=True)
        np.save(fpath, psi2d)
        H1d1e.plot_wf_2e_PRA(psi2d, title=tag, cfg=cfg, sign=sign)
        # H1d1e.plot_wf_2e(psi2d, title=tag)


    def build_and_plot_wf_2x1e_AND_2e_onlyColors(self, *, vecs_2x1e, vecs_2e, tag, sign=1):
        # sign just fixes +/-
        cfg = self.cfg
        if not cfg.plot_on:
            return
        vec_idx = cfg.plot_evec_idx
        psi2d_2x1e = self.build_r1r2(vecs_2x1e.mtrx[:, vec_idx])
        psi2d_2e = self.build_r1r2(vecs_2e.mtrx[:, vec_idx])
        dbg([psi2d_2x1e, psi2d_2e])

        fpath = cfg.current_dir_path / f'{tag}_{cfg.cfg_file_label}_psi2d_2x1e_idx{vec_idx}.npy'
        print(tag)
        print(fpath)
        os.makedirs(dirname(fpath), exist_ok=True)
        np.save(fpath, psi2d_2x1e)

        fpath = cfg.current_dir_path / f'{tag}_{cfg.cfg_file_label}_psi2d_2e_idx{vec_idx}.npy'
        print(tag)
        print(fpath)
        os.makedirs(dirname(fpath), exist_ok=True)
        np.save(fpath, psi2d_2e)

        H1d1e.plot_wf_2x1e_and_2e_PRA_v5_withPx1_onlyColors(psi2d_2x1e, psi2d_2e, title=tag, cfg=cfg, sign=sign)

    def build_and_plot_wf_2e_1s3s_and_1s4s_onlyColors(self, *, vecs_2e, tag, sign=1):
        # sign just fixes +/-
        cfg = self.cfg
        if not cfg.plot_on:
            return
        # vec_idx = cfg.plot_evec_idx
        vec_idx = 1  # for anti
        psi2d_1s3s = self.build_r1r2(vecs_2e.mtrx[:, vec_idx])
        psi2d_1s4s = self.build_r1r2(vecs_2e.mtrx[:, vec_idx+1])
        dbg([psi2d_1s3s, psi2d_1s4s])

        # fpath = cfg.current_dir_path / f'{tag}_{cfg.cfg_file_label}_psi2d_2x1e_idx{vec_idx}.npy'
        # print(tag)
        # print(fpath)
        # os.makedirs(dirname(fpath), exist_ok=True)
        # np.save(fpath, psi2d_2x1e)
        #
        # fpath = cfg.current_dir_path / f'{tag}_{cfg.cfg_file_label}_psi2d_2e_idx{vec_idx}.npy'
        # print(tag)
        # print(fpath)
        # os.makedirs(dirname(fpath), exist_ok=True)
        # np.save(fpath, psi2d_2e)

        H1d1e.plot_wf_2x1e_and_2e_PRA_v5_withPx1_onlyColors(psi2d_1s3s, psi2d_1s4s, title=tag, cfg=cfg, sign=sign)

    def build_and_plot_wf_2e_SymFromAnti_and_Symm_onlyColors(self, *, psi2d_2e_from_anti, vecs_2e, tag, sign=1):
        # sign just fixes +/-
        cfg = self.cfg
        if not cfg.plot_on:
            return
        vec_idx = cfg.plot_evec_idx
        psi2d_2e = self.build_r1r2(vecs_2e.mtrx[:, vec_idx])
        dbg([psi2d_2e_from_anti, psi2d_2e])
        # H1d1e.plot_wf_2x1e_and_2e_PRA_v5_withPx1_onlyColurs(
        #     psi2d_2e_from_anti, psi2d_2e, title=tag, cfg=cfg, sign=sign)
        H1d1e.plot_wf_2x2e_compare_PRA_v5_withPx1_onlyColurs(
            psi2d_2e_from_anti, psi2d_2e, title=tag, cfg=cfg, sign=sign,
            labels=("from reflection", "from diagonalization", "difference"))
        # H1d1e.plot_wf_2x2e_compare_PRA_v5_withPx1_onlyColurs(
        #     psi2d_2e_from_anti, psi2d_2e, title=tag, cfg=cfg, sign=sign)
        # def plot_wf_2x2e_compare_PRA_v5_withPx1_onlyColurs(
        #         psi_left, psi_right, *, cfg, sign=1, title=None,
        #         labels=(None, None, "difference")):

    def build_and_plot_wf_2e_onlyColors(self, vecs, *, tag, sign=1):
        # sign just fixes +/-
        cfg = self.cfg
        if not cfg.plot_on:
            return
        vec_idx = cfg.plot_evec_idx
        psi2d = self.build_r1r2(vecs.mtrx[:, vec_idx])
        # diag_cii = self.cfg.diag_cii
        dbg(psi2d)
        # fpath = cfg.current_dir_path / f'psi2x1e_ANTI_{cfg.cfg_file_label}_vec{vec_idx}.npy'
        fpath = cfg.current_dir_path / f'{tag}_{cfg.cfg_file_label}_vec{vec_idx}.npy'
        print(tag)
        print(fpath)
        os.makedirs(dirname(fpath), exist_ok=True)
        np.save(fpath, psi2d)
        H1d1e.plot_wf_2e_PRA_v5_withPx1_onlyColurs(psi2d, title=tag, cfg=cfg, sign=sign)
        # H1d1e.plot_wf_2e(psi2d, title=tag)

    def build_and_plot_wf_2e_forceSymm(self, vecs, *, tag, sign=1):
        cfg = self.cfg
        if not cfg.plot_on:
            return

        orth2e = cast(FuncArr1d2e, cfg.orth2e)
        B1 = cast(IWFuncArr, orth2e.basis_1d)
        B2 = cast(IWFuncArr, orth2e.basis2_1d)
        x1 = B1.getQuadr().getX().arr
        x2 = B2.getQuadr().getX().arr

        vec_idx = cfg.plot_evec_idx
        psi2d = self.build_r1r2(vecs.mtrx[:, vec_idx])
        psi2d = np.sign(x1[:, None] - x2[None, :]) * psi2d
        dbg(psi2d)
        # fpath = cfg.current_dir_path / f'psi2x1e_ANTI_{cfg.cfg_file_label}_vec{vec_idx}.npy'
        fpath = cfg.current_dir_path / f'{tag}_{cfg.cfg_file_label}_vec{vec_idx}.npy'
        print(tag)
        print(fpath)
        os.makedirs(dirname(fpath), exist_ok=True)
        np.save(fpath, psi2d)
        # H1d1e.plot_wf_2e_PRA_v5_withPx1_forceSymm(psi2d, title=tag, cfg=cfg, sign=sign)
        H1d1e.plot_wf_2e_PRA_v5_withPx1_onlyColurs(psi2d, title=tag, cfg=cfg, sign=sign)
        # def(psi2e_grid, *, cfg, sign=1, title=None):
        # H1d1e.plot_wf_2e(psi2d, title=tag)





if __name__ == "__main__":
    # from _new25.v250712_H1d1e_diag_OK.v250714c_He1d2eDiagLcr_TEST import He1d2eDiagLcr_Test
    from _new25.v250812_He1Dp.v250715_He1d_submitted250802.v250715a_He1d_LgrrOrthLcr_Test import He1d_LgrrOrthLcr_Test
    He1d_LgrrOrthLcr_Test().test_1()




