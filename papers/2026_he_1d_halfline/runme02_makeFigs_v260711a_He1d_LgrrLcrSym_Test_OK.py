# © 2026 Dmitry A. Konovalov — All rights reserved.

# todo: to run without PyCharm
import sys
from os.path import isdir
from pathlib import Path
# Automatically add src/qmbase to path
current_file = Path(__file__).resolve()
print(f"current_file: {current_file}")
# project_root = current_file.parents[4]   # adjust the number if folder depth changes
project_root = current_file.parents[2]   # adjust the number if folder depth changes
print(f"project_root: {project_root}")
src_root = str(project_root / "src" / "qmbase")

# # todo: admin: how to find all used *.py
# # pip install coverage
# # coverage run runme01_makeFigs_v260710c_He1d_LgrrLcrAnti_Test_OK.py
# # coverage report --include="**/*.py" > files.txt
# # todo admin! point to dev repo to pull all depends
# src_root = '/Users/jc138691/dev/y21m11gh_math_python/v25_qm/qm25'

print(f"src_root: {src_root}")
assert isdir(src_root), f'ERROR: missing src_root={src_root}'
sys.path.insert(0, src_root)
print(f"TEST import _new25.dbg; from src_root={src_root}")
try:
    import _new25.dbg
    print("Successfully imported _new25.dbg")
except ImportError as e:
    print(f"Failed to import _new25.dbg: {e}")
    exit(1)

# ---------------
import numpy as np

from _new25.dbg import set_dbg, dbg
from d1.d1e2.FuncArr1d2e import FuncArr1d2e
from d1.d1e1.H1d1e import H1d1e
from d1.d1e2.H_1d_2e_LgrrLcrSym import H_1d_2e_LgrrLcrSym
from javax.utilx.log.Log import Log
from javax.utilx.log.kiss.KissLog import KissLog

from project.workflow.task.test.FlowTest import FlowTest

from atom.AtomUtil import AtomUtil
from atom.wf.lcr.LcrFactory import LcrFactory
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr

from qm_math.integral.OrthFactory import OrthFactory
from qm_math.vec.Vec import Vec
from qm_math.vec.VecDbgView import VecDbgView
from qm_math.vec.grid.StepGrid import StepGrid
from qm_math.vec.grid.StepGridOpt import StepGridOpt

from scatt.jm_2008.jm.laguerre.LgrrOpt import LgrrOpt
from scatt.jm_2008.jm.laguerre.lcr.LgrrOrthLcr import LgrrOrthLcr

log = Log.getLog('He1d_LgrrLcrSym_Test')
DBG_ON = True

class cfg(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    run_self_test = False  # todo
    seed = 1
    nx_pad = 2
    atom_z = 2.  # todo <----- 2
    max_norm_err = 1e-9

class He1d_LgrrLcrSym_Test(FlowTest):
    # todo from v25_qm/qm25/atom/e_2/test/AtomHeTest.py
    LCR_FIRST = None
    LCR_N = None
    R_LAST = None
    LAMBDA = None

    def __init__(self):
        super().__init__(He1d_LgrrLcrSym_Test)  # needed by FlowTest
        FlowTest.setMaxErr(1e-12)  # ok
        FlowTest.setLog(log)
        log.setDbg(DBG_ON)
        from qm_math.vec.DbgView import DbgView
        DbgView.show_digs = 7
        KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))
        KissLog.register_formatter(np.ndarray, lambda v: str(VecDbgView(v)))

    def test_1(self):
        set_dbg(DBG_ON)

        #   # todo: use this to reproduce paper Figs!
        orth1N = 25;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        basis_lambda_ = 2.0 * cfg.atom_z  # for anti
        nr1 = 1201; r1_max = 200.; r1_min = 0.0;  x1_min = -5.
        cfg.plot_on = True;  cfg.fig_dpis = [300]
        cfg.plot_evec_idx = 0  # 0 is ground
        cfg.npy_fpath_to_plot_compare = 'psi2e_SYMM_from_ANTI_idx0_orth1N25_N25_Lamb4_nr1201_0_xmin5_vec0.npy'
        # h2x1e.getEigEngs = [-4.       -2.5      -2.222222 -2.124913 -2.073717]
        # h2x1e_anti.engs [-4.       -2.5      -2.222222 -2.124913 -2.073717]
        # err_eng_2x1e =  1.2967404927621828e-13
        # psi2x1e_ANTI_idx0_N25
        # h2e.getEigEngs = [-2.10761  -2.049071 -2.011303 -1.955866 -1.878874]
        # re-run
        # h2x1e.getEigEngs = [-4.       -2.5      -2.222222 -2.124913 -2.073717]
        # h2x1e_sym.engs [-4.       -2.5      -2.222222 -2.124913 -2.073717]
        # err_eng_2x1e =  1.2967404927621828e-13
        # h2e_sym.getEigEngs = [-2.10761  -2.049071 -2.011303 -1.955866 -1.878874]


        # todo: use this to try! faster than orth1N = 25;
        orth1N = 20;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        basis_lambda_ = 2.0 * cfg.atom_z  # for anti
        nr1 = 1201; r1_max = 200.; r1_min = 0.0;  x1_min = -5.
        cfg.plot_on = True;  cfg.fig_dpis = [300]
        cfg.plot_evec_idx = 0  # 0 is ground
        cfg.npy_fpath_to_plot_compare = 'psi2e_SYMM_from_ANTI_idx0_orth1N20_N20_Lamb4_nr1201_0_xmin5_vec0.npy'
        # h2x1e.getEigEngs = [-4.       -2.5      -2.222221 -2.123399 -2.05158 ]
        # h2x1e_sym.engs [-4.       -2.5      -2.222221 -2.123399 -2.05158 ]
        # err_eng_2x1e =  1.2922996006636822e-13
        # h2e_sym.getEigEngs = [-2.10748  -2.044719 -1.983958 -1.890802 -1.760227]


        # cfg.plot_x_min = -3.5
        cfg.plot_x_min = -5
        # set the same grids
        orth2N = orth1N  # very good check for dbg
        nr2 = nr1;  r2_min = r1_min;  r2_max = r1_max;  x2_min = x1_min
        cfg.vee_1r12_eps = 1e-10
        ASSUME_SYMM = (nr1 == nr2) & (r1_max == r2_max) & (r1_min == r2_min)
        # todo: NOTE!!! BUG in fix_sym_v2_only_orth2e for orth2N != orth1N
        ASSUME_SYMM = ASSUME_SYMM & (x1_min == x2_min) & (orth2N == orth1N)

        # Grids -------
        grid_r1 = StepGridOpt(r1_min, r1_max, nr1)  #; // R_N not used!!!
        grid_r2 = StepGridOpt(r2_min, r2_max, nr2)  #; // R_N not used!!!
        log.dbg("R step grid-1 model =", grid_r1)
        log.dbg("R step grid-2 model =", grid_r2)
        sg1 = LcrFactory.makeLcrFromR(x1_min, nr1, grid_r1)
        sg2 = LcrFactory.makeLcrFromR(x2_min, nr2, grid_r2)
        log.dbg("LCR step grid opt-1 =", sg1)
        log.dbg("LCR step grid opt-2 =", sg2)
        x1_grid = StepGrid.fromStepGridOpt(sg1)
        x2_grid = StepGrid.fromStepGridOpt(sg2)
        log.dbg("x1_grid =", x1_grid)
        log.dbg("x2_grid =", x2_grid)
        # cfg.x_grid_np = x_grid.arr  # to cfg.x_grid_np

        # Quadratures -------------
        w1 = WFQuadrLcr(x1_grid, r_min=r1_min)
        w2 = WFQuadrLcr(x2_grid, r_min=r2_min)
        cfg.wLcr1 = w1
        cfg.wLcr2 = w2
        log.dbg("WFQuadrLcr 1=", w1)
        log.dbg("WFQuadrLcr 2=", w2)
        r1_grid = w1.getR()
        r2_grid = w2.getR()
        dbg('r1_grid.arr[0]')
        dbg('r2_grid.arr[0]')
        log.dbg("r1_grid =", r1_grid)
        log.dbg("r2_grid =", r2_grid)

        # Lgrr orth
        cfg.eng_1d1e_n1 = H1d1e.calc_eng_1d1e(atom_z=cfg.atom_z, n=1)
        cfg.eng_1d1e_n2 = H1d1e.calc_eng_1d1e(atom_z=cfg.atom_z, n=2)
        cfg.eng_2x1d1e = 2 * cfg.eng_1d1e_n1
        cfg.eng_2x1d1e_symm = 2 * cfg.eng_1d1e_n1
        cfg.max_err_1d_2e_symm = 1e-3
        cfg.eng_1d2e_symm = -2.10839   # todo <-----  -2.10839 found numerically -2.10839,
        cfg.eng_1d2e_check = cfg.eng_1d2e_symm
        cfg.eng_2x1e = cfg.eng_2x1d1e
        cfg.eng_2e = -2.10839  # from N=45
        lgrrOpt1 = LgrrOpt(L=0, lambda_=basis_lambda_, N=orth1N)
        lgrrOpt2 = LgrrOpt(L=0, lambda_=basis_lambda_, N=orth2N)
        orth1 = LgrrOrthLcr(w1, lgrrOpt1)
        orth2 = LgrrOrthLcr(w2, lgrrOpt2)
        cfg.orth1 = orth1;  cfg.orth2 = orth2
        cfg.orth1N = orth1N;  cfg.orth2N = orth2N
        AtomUtil.trimTailSLOW(orth1)
        AtomUtil.trimTailSLOW(orth2)
        OrthFactory.log.setDbg(False)
        res1 = OrthFactory.calcMaxOrthErr(orth1, w1.getWithCR2())
        res2 = OrthFactory.calcMaxOrthErr(orth2, w2.getWithCR2())
        dbg('res1')
        dbg('res2')
        self.assertEquals(0, res1, 1e-10)
        self.assertEquals(0, res2, 1e-10)
        self._self_test()

        def make_wf_label(cfg):
            label = f'N{orth1N}_Lamb{int(basis_lambda_)}_nr{nr1}_{nr2-nr1}_xmin{abs(int(x1_min))}'
            dbg('label')
            return label
        current_dir = Path(__file__).resolve().parent
        file_label = make_wf_label(cfg)
        cfg.current_dir_path = current_dir / "results"
        cfg.cfg_file_label = file_label

        cfg.orth2e = FuncArr1d2e(orth1, orth2, load_symm_half=False)
        H_1d_2e_LgrrLcrSym(cfg).diag_on_2orth1(ASSUME_SYMM)


    def _self_test(self):
        if not cfg.run_self_test:
            return
        # FlowTest.setMaxErr(cfg.max_norm_err)
        # from _new25.tests.v250704_ready_common_tests import run_common_tests_part1
        # run_common_tests_part1()
        # FlowTest.setMaxErr(cfg.max_norm_err)
        # dbg('cfg.max_norm_err')
        # from scatt.jm_2008.jm.laguerre.JmLagrrOrthRTest import JmLagrrOrthRTest
        # JmLagrrOrthRTest(cfg.orth1).testNorm()
        # JmLagrrOrthRTest(cfg.orth2).testNorm()
        # from qm_station.jm.tests.JmPotEigVecRTest import JmPotEigVecRTest
        # JmPotEigVecRTest(cfg.orth1).testNorm()
        # JmPotEigVecRTest(cfg.orth2).testNorm()


if __name__ == "__main__":
    He1d_LgrrLcrSym_Test().test_1()
