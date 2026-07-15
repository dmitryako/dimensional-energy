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
    atom_z = 1.1  # todo <-----
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

        #  h2e_anti.getEigEngs= Vec[435] = {-0.6058876, -0.6044818, -0.6022012, -0.5989754, -0.5947491, .
        # N=45  h2e_anti.getEigEngs= Vec[990] = {-0.6059604, -0.6052570, -0.6042972, -0.6029243, -0.6011385, .
        # Delta(eng - eng_1d1e_n1) -0.00096  # from N=45

        orth1N = 45;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        basis_lambda_ = 1.0 * cfg.atom_z  # for anti
        nr1 = 1201; r1_max = 250.; r1_min = 0.0;  x1_min = -5.
        # He1d_LgrrLcrAnti_Test: WFQuadrLcr 2=
        # x = Vec[1201] = {-5.0000000, -4.9912321, -4.9824642, -4.9736963, -4.9649284, ..., 5.4864162, 5.4951841, 5.5039521, 5.5127200, 5.5214879}
        # y = Vec[1201] = {0.0027278, 0.0124699, 0.0046762, 0.0124699, 0.0054556, ..., 0.0054556, 0.0124699, 0.0046762, 0.0124699, 0.0027278}
        # r1_grid.arr[0] = 0.0
        # r2_grid.arr[0] = 0.0
        # He1d_LgrrLcrAnti_Test: r1_grid = Vec[1201] = {0.0000000, 0.0000593, 0.0001192, 0.0001796, 0.0002405, ..., 241.3838318, 243.5096275, 245.6541440, 247.8175460, 250.0000000}
        # He1d_LgrrLcrAnti_Test: r2_grid = Vec[1201] = {0.0000000, 0.0000593, 0.0001192, 0.0001796, 0.0002405, ..., 241.3838318, 243.5096275, 245.6541440, 247.8175460, 250.0000000}
        # orth_err= 3.2862601528904634e-14
        # orth_err= 3.2862601528904634e-14
        # load_mats rows: 100%|██████████| 2025/2025 [03:45<00:00,  8.99it/s]
        # wLcr1: nr1=1201, xmin=-5.0, xmax=5.52148786928705, rmin0.0, rmax=250.0
        # ANTI h2x1e.engs(orth1N=45) = [-0.75625  -0.672222 -0.642812 -0.6292   -0.621806]
        # ANTI eng_2x1d1e_anti=-0.7562500000000001
        # err_eng_2x1e =  3.3084646133829665e-14
        # ANTI h2e_engs(orth1N=45) = [-0.60596  -0.605257 -0.604297 -0.602924 -0.601139]
        # eng_1d1e_n1(Z=1.1) = -0.6050000000000001
        # ANTI Delta(h2e_engs - eng_1d1e_n1) = [-0.00096  -0.000257  0.000703  0.002076  0.003861]
        # cfg.eng_2e_check= -0.60596

        # orth1N = 35;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        # basis_lambda_ = 1.0 * cfg.atom_z  # for anti
        # nr1 = 1201; r1_max = 200.; r1_min = 0.0;  x1_min = -5.
        # # orth_err= 1.2878587085651816e-14
        # # load_mats rows: 100%|██████████| 1225/1225 [01:08<00:00, 17.80it/s]
        # # wLcr1: nr1=1201, xmin=-5.0, xmax=5.298351055715546, rmin0.0, rmax=200.00000000000009
        # # ANTI h2x1e.engs(orth1N=35) = [-0.75625  -0.672222 -0.642812 -0.6292   -0.621806]
        # # ANTI eng_2x1d1e_anti=-0.7562500000000001
        # # err_eng_2x1e =  2.5424107263916085e-14
        # # ANTI h2e_engs(orth1N=35) = [-0.605934 -0.604886 -0.603241 -0.600911 -0.597875]
        # # eng_1d1e_n1(Z=1.1) = -0.6050000000000001
        # # ANTI Delta(h2e_engs - eng_1d1e_n1) = [-0.000934  0.000114  0.001759  0.004089  0.007125]
        # # cfg.eng_2e_check= -0.60596

        orth1N = 20;  cfg.max_err_2x1e = 1e-5;  cfg.max_err_2e = 0.001
        basis_lambda_ = 1.0 * cfg.atom_z  # for anti
        nr1 = 1201; r1_max = 200.; r1_min = 0.0;  x1_min = -5.
        # He1d_LgrrLcrAnti_Test: r2_grid = Vec[1201] = {0.0000000, 0.0000581, 0.0001166, 0.0001757, 0.0002353, ..., 193.2507084, 194.9163730, 196.5963938, 198.2908946, 200.0000000}
        # orth_err= 8.104628079763643e-15
        # orth_err= 8.104628079763643e-15
        # load_mats rows: 100%|██████████| 400/400 [00:17<00:00, 23.25it/s]
        # wLcr1: nr1=1201, xmin=-5.0, xmax=5.298351055715546, rmin0.0, rmax=200.00000000000009
        # ANTI h2x1e.engs(orth1N=20) = [-0.75625  -0.672222 -0.642812 -0.629198 -0.621514]
        # ANTI eng_2x1d1e_anti=-0.7562500000000001
        # err_eng_2x1e =  2.5757174171303632e-14
        # ANTI h2e_engs(orth1N=20) = [-0.605484 -0.602241 -0.596863 -0.589122 -0.578619]
        # eng_1d1e_n1(Z=1.1) = -0.6050000000000001
        # ANTI Delta(h2e_engs - eng_1d1e_n1) = [-0.000484  0.002759  0.008137  0.015878  0.026381]
        # cfg.eng_2e_check= -0.60596

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
        cfg.eng_2e = cfg.eng_1d1e_n1 - 0.00096  # from N=45
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
