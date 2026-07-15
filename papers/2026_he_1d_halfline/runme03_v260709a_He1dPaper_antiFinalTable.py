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

log = Log.getLog('He1d_LgrrLcrAnti_Test')
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

class He1d_LgrrLcrAnti_Test(FlowTest):
    # todo from v25_qm/qm25/atom/e_2/test/AtomHeTest.py
    LCR_FIRST = None
    LCR_N = None
    R_LAST = None
    LAMBDA = None

    def __init__(self):
        super().__init__(He1d_LgrrLcrAnti_Test)  # needed by FlowTest
        FlowTest.setMaxErr(1e-12)  # ok
        FlowTest.setLog(log)
        log.setDbg(DBG_ON)
        from qm_math.vec.DbgView import DbgView
        DbgView.show_digs = 7
        KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))
        KissLog.register_formatter(np.ndarray, lambda v: str(VecDbgView(v)))

    def test_1(self):
        set_dbg(DBG_ON)

        # $(Z=2,1s,2s)$              & $4$   & $-0.10839$ & $-0.108$   & diag \\

        orth1N = 45;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        basis_lambda_ = 1.0 * cfg.atom_z  # for anti
        nr1 = 1201; r1_max = 200.; r1_min = 0.0;  x1_min = -5.
        # orth_err= 2.9976021664879227e-14
        # wLcr1: nr1=1201, xmin=-5.0, xmax=5.298351055715546, rmin0.0, rmax=200.00000000000009
        # ANTI h2x1e.engs(orth1N=45) = [-2.5      -2.222222 -2.125    -2.08     -2.055556]
        # ANTI eng_2x1d1e_anti=-2.5
        # err_eng_2x1e =  8.615330671091215e-14
        # ANTI h2e_engs(orth1N=45) = [-2.108386 -2.050372 -2.02901  -2.018837 -2.01309 ]
        # eng_1d1e_n1(Z=2.0) = -2.0
        # ANTI Delta(h2e_engs - eng_1d1e_n1) = [-0.108386 -0.050372 -0.02901  -0.018837 -0.01309 ]
        # cfg.eng_2e_check= -2.10839

        # # todo: note! you need lambda = 1 * Z; to get best 1s,3s; 1s,4s
        # orth1N = 40;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        # basis_lambda_ = 1.0 * cfg.atom_z  # for anti
        # nr1 = 1201; r1_max = 200.; r1_min = 0.0;  x1_min = -5.
        # # orth_err= 2.9976021664879227e-14
        # # h2x1e.getEigEngs = [-2.5      -2.222222 -2.125    -2.08     -2.055556]
        # # h2x1e_anti.engs [-2.5      -2.222222 -2.125    -2.08     -2.055556]
        # # err_eng_2x1e =  1.3322676295501878e-13
        # # h2e.getEigEngs = [-2.108385 -2.050372 -2.02901  -2.018831 -2.012736]

        # orth1N = 35;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        # basis_lambda_ = 1.0 * cfg.atom_z  # for anti
        # nr1 = 1201; r1_max = 200.; r1_min = 0.0;  x1_min = -5.
        # # orth_err= 1.1102230246251565e-14
        # # wLcr1: nr1=1201, xmin=-5.0, xmax=5.298351055715546, rmin0.0, rmax=200.00000000000009
        # # ANTI h2x1e.engs(orth1N=35) = [-2.5      -2.222222 -2.125    -2.08     -2.055556]
        # # ANTI eng_2x1d1e_anti=-2.5
        # # err_eng_2x1e =  8.526512829121202e-14
        # # ANTI h2e_engs(orth1N=35) = [-2.108383 -2.050371 -2.02901  -2.018776 -2.01166 ]
        # # eng_1d1e_n1(Z=2.0) = -2.0
        # # ANTI Delta(h2e_engs - eng_1d1e_n1) = [-0.108383 -0.050371 -0.02901  -0.018776 -0.01166 ]
        # # cfg.eng_2e_check= -2.10839

        orth1N = 20;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        basis_lambda_ = 1.0 * cfg.atom_z  # for anti
        nr1 = 1201; r1_max = 200.; r1_min = 0.0;  x1_min = -5.
        # orth_err= 7.882583474838611e-15
        # wLcr1: nr1=1201, xmin=-5.0, xmax=5.298351055715546, rmin0.0, rmax=200.00000000000009
        # ANTI h2x1e.engs(orth1N=20) = [-2.5      -2.222222 -2.125    -2.079995 -2.054591]
        # ANTI eng_2x1d1e_anti=-2.5
        # err_eng_2x1e =  8.43769498715119e-14
        # ANTI h2e_engs(orth1N=20) = [-2.108372 -2.050366 -2.028393 -2.010477 -1.984221]
        # eng_1d1e_n1(Z=2.0) = -2.0
        # ANTI Delta(h2e_engs - eng_1d1e_n1) = [-0.108372 -0.050366 -0.028393 -0.010477  0.015779]


        cfg.plot_on = False
        cfg.plot_evec_idx = 0  # 0 is ground
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
        cfg.eng_2x1d1e_anti = cfg.eng_1d1e_n2 + cfg.eng_1d1e_n1
        cfg.eng_1d2e_anti = cfg.eng_2x1d1e_anti
        cfg.eng_2x1e = cfg.eng_2x1d1e_anti
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
        # Log.getLog('H_1d_2e_LgrrLcr').setDbg(True)
        from d1.d1e2.H_1d_2e_LgrrLcrAnti import H_1d_2e_LgrrLcrAnti
        H_1d_2e_LgrrLcrAnti(cfg).diag_on_2orth1(ASSUME_SYMM)


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
    He1d_LgrrLcrAnti_Test().test_1()
