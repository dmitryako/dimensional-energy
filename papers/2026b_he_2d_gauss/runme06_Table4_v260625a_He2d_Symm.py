# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : v250715a_He1dOrthLggrLcr.py Created : 2025-07-15 at 6:08 pm by Dmitry.A.Konovalov@gmail.com
# coding: utf-8
from pathlib import Path
import numpy as np


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
# # coverage run v260625aa_Hy2d_Ln2Re_export_used_files.py
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

log = Log.getLog('He2d_LcrSymm_rerun')

DBG_ON = True
# DBG_ON = False

class cfg(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    run_self_test = False  # todo
    seed = 1
    L = -1/2
    max_norm_err = 1e-9
    atom_z = 2.  # todo <----- 2
    m_2d = 0

class He2d_LcrSymm_rerun(FlowTest):
    # todo from /v25_qm/qm25/_new25/v250715_He1d_submitted250802/v250717ab_He1dPaper_LgrrLcrAnti_FinalTable.py

    def __init__(self):
        super().__init__(He2d_LcrSymm_rerun)  # needed by FlowTest
        FlowTest.unlockMaxErr()  # ok
        FlowTest.setMaxErr(1e-12)  # ok
        FlowTest.setLog(log)
        log.setDbg(DBG_ON)
        from qm_math.vec.DbgView import DbgView
        DbgView.show_digs = 7
        KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))
        KissLog.register_formatter(np.ndarray, lambda v: str(VecDbgView(v)))
        # Log.getLog("PotHMtrx").setDbg(False)  # on/off for PotHMtrx
        # Log.getLog("PotHLcr").setDbg(False)  # on/off for PotHMtrx
        # Log.getLog("PotH").setDbg(False)  # on/off for PotHMtrx
        # Log.getLog("WFQuadrLcr").setDbg(False)  # on/off for PotHMtrx

    def test_1(self):
        set_dbg(DBG_ON)
        # todo: see Hy2D_EigVecLcr_Test
        L = cfg.L
        lambda_ = 4 * cfg.atom_z   # <-- todo note! in 2D it's  1s = 4 * Z
        # lambda_ = 2; orth1N = 10; nx1 = 1201; x1_min = -12.

        orth1N = 20; nx1 = 1201; x1_min = -12.; x1_max = 5.
        # H2d2e_LcrSymm: h2x1e_symm.getEigEngs= -16.0000000, -8.8888889, -8.3179926, -8.0859221
        # H2d2e_LcrSymm: h2e_symm.getEigEngs= -11.7521651, -8.2472358, -8.0400349, -7.7889949
        # err_eng =  -1.8286446135107326e-08
        # RERUN:
        # H2d2e_LcrSymm: h2x1e_symm.getEigEngs= -16.0000000, -8.8888889, -8.3179926, -8.0859221
        # H2d2e_LcrSymm: h2e_symm.getEigEngs= -11.7521651, -8.2472358, -8.0400349, -7.7889949
        # err_eng =  -1.8286446135107326e-08
        # ORIG:
        # H2d2e_LcrSymm: h2x1e_symm.getEigEngs= -16.0000000, -8.8888889, -8.3179926, -8.0859221
        # H2d2e_LcrSymm: h2e_symm.getEigEngs= -11.7521651, -8.2472358, -8.0400349, -7.7889949
        # err_eng =  -1.8286446135107326e-08

        # orth1N = 40; nx1 = 1601; x1_min = -12.; x1_max = 5.
        # # H2d2e_LcrSymm: h2x1e_symm.getEigEngs= -16.0000000, -8.8888889, -8.3200000, -8.1630609
        # # H2d2e_LcrSymm: h2e_symm.getEigEngs= -11.7634416, -8.2477067, -8.0849149, -8.0256340
        # # err_eng =  -1.1638253027967949e-08


        # lambda_ = 2 * cfg.atom_z   #
        # orth1N = 35; nx1 = 1601; x1_min = -12.; x1_max = 5.
        # # h2x1e_symm.getEigEngs= [-16.00000001  -8.88888889  -8.32        -8.16326531]
        # # h2e_symm.getEigEngs= [-11.76313782  -8.24769995  -8.08528108  -8.04252323]
        # # H2d2e_LcrSymm: h2x1e_symm.getEigEngs= -16.0000000, -8.8888889, -8.3200000, -8.1632653
        # # H2d2e_LcrSymm: h2e_symm.getEigEngs= -11.7631378, -8.2477000, -8.0852811, -8.0425232
        # # err_eng =  -1.0779679371353268e-08
        #
        # lambda_ = 2 * cfg.atom_z   #
        # orth1N = 40; nx1 = 1601; x1_min = -12.; x1_max = 5.
        # # H2d2e_LcrSymm: h2x1e_symm.getEigEngs= -16.0000000, -8.8888889, -8.3200000, -8.163 2653
        # # H2d2e_LcrSymm: h2e_symm.getEigEngs= -11.7632196, -8.2477018, -8.0852815, -8.0426811
        # # err_eng =  -1.0779363179835855e-08
        # # H2d2e_LcrAnti: h2x1e_anti.getEigEngs= -8.8888889, -8.3200000, -8.1632653, -8.0987651
        # # H2d2e_LcrAnti: h2e_anti.getEigEngs= -8.2952918, -8.0944737, -8.0458920, -8.0252704
        # # err_eng =  -5.41920286423192e-09


        # E(S=0,Z=2) = -11.8981, \\
        # E(S=0,Z=2) = -11.8881 \cite{Singh2020},\\
        # E(S=0,Z=2) = -11.6472 \cite{patil2008helium},\\
        # E(S=0,Z=2) = -11.56 \cite{skobelev2017energy},\\

        cfg.vee_1r12_eps = 1e-10 # same for 1e-8, 1e-10

        # cfg.plot_on = True
        cfg.plot_on = False
        # cfg.plot_evec_idx = 0  # 0 is ground
        cfg.plot_evec_idx = 0  # 0 is ground
        # TODO: NOTE!!! use diff, helps with dbging
        # nx2 = nx1 + 40
        # x2_min = x1_min - 0.1
        # x2_max = x1_max + 0.1
        # orth2N = orth1N + 1
        # todo: prod runs with the same?
        nx2 = nx1
        x2_min = x1_min
        x2_max = x1_max
        orth2N = orth1N

        # Grids -------
        x1_opt = StepGridOpt(x1_min, x1_max, nx1)  #;
        x2_opt = StepGridOpt(x2_min, x2_max, nx2)  #;
        log.dbg("x1_opt =", x1_opt)
        log.dbg("x2_opt =", x2_opt)
        x1_grid = StepGrid.fromStepGridOpt(x1_opt)
        x2_grid = StepGrid.fromStepGridOpt(x2_opt)
        log.dbg("x1_grid =", x1_grid)
        log.dbg("x2_grid =", x2_grid)

        #         # Quadratures -------------
        r1_min = LcrFactory.make_r1_min(x1_grid, use_c=0)
        r2_min = LcrFactory.make_r1_min(x2_grid, use_c=0)
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
        from d2_py.d2e1.H2d1e import H2d1e
        cfg.eng_2d1e_n0m0 = H2d1e.calc_eng_2d1e(atom_z=cfg.atom_z, n=0, m=0)
        cfg.eng_2d1e_n1m0 = H2d1e.calc_eng_2d1e(atom_z=cfg.atom_z, n=1, m=0)
        cfg.eng_2x2d1e_symm = 2 * cfg.eng_2d1e_n0m0
        cfg.eng_2x2d1e_anti = cfg.eng_2d1e_n0m0 + cfg.eng_2d1e_n1m0
        # cfg.eng_1d2e_anti = -2.10839   # todo <-----  -2.10839 found numerically -2.10839,
        cfg.eng_2d2e_symm = -11.8981   # todo <-----  ?? i have it in my latex for 2d-He
        # cfg.eng_2d1e_n0m0 = -8.0
        # cfg.eng_2d1e_n1m0 = -0.8888888888888888
        # cfg.eng_2x2d1e_anti = -8.88888888888889
        # cfg.eng_2x2d1e_symm = -16.0
        # cfg.eng_2d2e_symm = (-11.8981,)
        dbg("cfg.eng_2d1e_n0m0")
        dbg("cfg.eng_2d1e_n1m0")
        dbg("cfg.eng_2x2d1e_anti")
        dbg("cfg.eng_2x2d1e_symm")
        dbg("cfg.eng_2d2e_symm")
        lgrrOpt1 = LgrrOpt(L=L, m_2d=0, lambda_=lambda_, N=orth1N)  # see v250723a-2d2eZ-PRA.tex
        lgrrOpt2 = LgrrOpt(L=L, m_2d=0, lambda_=lambda_, N=orth2N)  # see v250723a-2d2eZ-PRA.tex
        orth1 = LgrrOrthLcr(w1, lgrrOpt1)
        orth2 = LgrrOrthLcr(w2, lgrrOpt2)
        cfg.orth1 = orth1
        cfg.orth2 = orth2
        # todo? trimTailSLOW
        # AtomUtil.trimTailSLOW(orth1)
        # AtomUtil.trimTailSLOW(orth2)
        OrthFactory.log.setDbg(False)
        res1 = OrthFactory.calcMaxOrthErr(orth1, w1)
        res2 = OrthFactory.calcMaxOrthErr(orth2, w2)
        dbg('res1')
        dbg('res2')
        self.assertEquals(0, res1, 1e-10)
        self.assertEquals(0, res2, 1e-10)
        self._self_test()

        def make_wf_label(cfg):
            label = f'N{orth1N}_Lamb{int(lambda_)}_nx{nx1}_{nx2-nx1}_xmin{abs(int(x1_min))}'
            dbg('label')
            return label
        current_dir = Path(__file__).resolve().parent
        file_label = make_wf_label(cfg)
        cfg.current_dir_path = current_dir / "results"
        cfg.cfg_file_label = file_label

        cfg.orth2e = FuncArr1d2e(orth1, orth2, load_symm_half=False)
        cfg.SYM_SIGN = 1
        # H1d2e_LgrrLcrAnti(cfg).diag_on_2orth1()
        # Log.getLog("H2d2e_LcrAnti").setDbg(False)  # on/off for PotHMtrx
        # Log.getLog('H1d2e_LgrrLcr').setDbg(False)  # H2d2e_LcrSymm
        # Log.getLog('H1d2e_LgrrLcrAnti').setDbg(False)  #
        # Log.getLog('H2d2e_LcrSymm').setDbg(False)  #
        Log.getLog('H2d2e_LcrSymm').setDbg(DBG_ON)  #
        Log.print_dbg_logs()
        from d1.d1e2.H2d2e_LcrSymm import H2d2e_LcrSymm
        H2d2e_LcrSymm(cfg).diag_on_2orth1(sign=-1)
    #     - WFQuadrLcr: dbg True
    # - OrthFactory: dbg True
    # - He2d_LcrAnti_run: dbg True
    # - H2d2e_LcrAnti: dbg True


    def _self_test(self):
        if not cfg.run_self_test:
            return
        FlowTest.setMaxErr(cfg.max_norm_err)
        from _new25.tests.v250704_ready_common_tests import run_common_tests_part1
        run_common_tests_part1()
        FlowTest.setMaxErr(cfg.max_norm_err)
        dbg('cfg.max_norm_err')
        # from scatt.jm_2008.jm.laguerre.JmLagrrOrthRTest import JmLagrrOrthRTest
        # JmLagrrOrthRTest(cfg.orth1).testNorm()
        # JmLagrrOrthRTest(cfg.orth2).testNorm()
        # from qm_station.jm.tests.JmPotEigVecRTest import JmPotEigVecRTest
        # JmPotEigVecRTest(cfg.orth1).testNorm()
        # JmPotEigVecRTest(cfg.orth2).testNorm()




# --- Add run test code block ---
if __name__ == "__main__":
    He2d_LcrSymm_rerun().test_1()
    # FlowTest.ok(He1d_LgrrOrthLcr_Test)
