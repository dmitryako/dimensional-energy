# © 2026 Dmitry A. Konovalov — All rights reserved.
import math
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


from typing import cast
import numpy as np

from _new25.dbg import dbg, set_dbg
from atom.energy.pw.lcr.PotHLcr import PotHLcr
from atom.wf.lcr.LcrFactory import LcrFactory
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
# from d2_py.tests.Hy2DLn_EigVecLcr_Test import Hy2DLn_EigVecLcr_Test
from javax.utilx.log.Log import Log
from javax.utilx.log.kiss.KissLog import KissLog
from project.workflow.task.test.FlowTest import FlowTest
from qm_math.integral.OrthFactory import OrthFactory
from qm_math.vec.Vec import Vec
from qm_math.vec.VecDbgView import VecDbgView
from qm_math.vec.grid.StepGrid import StepGrid
from qm_math.vec.grid.StepGridOpt import StepGridOpt
from scatt.jm_2008.jm.laguerre.LgrrOpt import LgrrOpt
from scatt.jm_2008.jm.laguerre.lcr.LgrrOrth2D_Lcr import LgrrOrth2D_Lcr
from d2_py.FuncAtom2D_X_crossover import FuncAtom2D_X_crossover
from qm_math.func.simple.FuncPowAbsInt import FuncPowAbsInt
from scatt.jm_2008.jm.laguerre.lcr.LgrrOrthLcr import LgrrOrthLcr


"""
Copyright dmitry.konovalov@jcu.edu.au Date: 23/06/2026, Time: 15:42:20
"""

log = Log.getLog('H_1e_D_Z_Lcr_run')

# DBG_ON = True
DBG_ON = False

def make_pot_3d(Z):
    return FuncPowAbsInt(-Z, -1)

def make_pot_P(Z):
    return FuncPowAbsInt(-Z, -1)

def make_pot_X(Z):
    return FuncAtom2D_X_crossover(z=-Z)

def make_pot_RK(Z):
    from d2_py.FuncAtom2D_RytKel import FuncAtom2D_RytKel
    # potFunc = FuncAtom2DLn(z=-cfg.atom_z)  # // f(r)=-2*Z*Ln(r)
    # potFunc = FuncAtom2D_RytKel(z=-cfg.atom_z)  # // f(r)=-2*Z*Ln(r)
    return FuncAtom2D_RytKel(z=-Z)   # use your actual RK class

pots = [ # tag, L, pot
    ("3D",  0.0,  make_pot_3d),
    ("RK", -0.5,  make_pot_RK),
    ("X",  -0.5,  make_pot_X),
    ("P",  -0.5,  make_pot_P),
]

class cfg(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    # run_self_test = True  # todo RE-RUN before starting new dev
    run_self_test = False  # todo
    seed = 1
    nx_pad = 2
    atom_z = 1.  # todo <----- 1, 1d-Hydrogen
    # L = -1/2  # todo: 2D
    L = 0  # todo 3D
    max_norm_err = 1e-10
    r0 = math.e / 2  # cross-over


class H_1e_D_Z_Lcr_run(FlowTest):
    # todo: copied here from Hy2DLn_EigVecLcr_Test so easy to find
    # Hy2DLn_EigVecLcr_Test
    # from d2_py.tests.Hy2DLn_EigVecLcr_Test import Hy2DLn_EigVecLcr_Test
    def __init__(self, basis=None):
        super().__init__(H_1e_D_Z_Lcr_run)
        self._arr = None
        self._w = None
        self._lgrrOpt = None
        if basis is not None:
            self._arr = basis
        set_dbg(DBG_ON)
        # FlowTest.setMaxErr(1e-12)  # ok
        FlowTest.unlockMaxErr()  # ok
        FlowTest.lockMaxErr(1e-12)  # ok
        FlowTest.setLog(log)
        log.setDbg(DBG_ON)
        from qm_math.vec.DbgView import DbgView
        DbgView.show_digs = 7
        KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))
        KissLog.register_formatter(np.ndarray, lambda v: str(VecDbgView(v)))
        # Log.getLog("LgrrArr").setDbg(False)  # on/off for PotHMtrx
        # Log.getLog("PotHMtrx").setDbg(False)  # on/off for PotHMtrx
        # Log.getLog("PotHLcr").setDbg(False)  # on/off for PotHMtrx
        # Log.getLog("PotH").setDbg(False)  # on/off for PotHMtrx
        # Log.getLog("WFQuadrLcr").setDbg(False)  # on/off for PotHMtrx

    def load_test_1(self):
        L = cfg.L
        np.set_printoptions(precision=6)

        orth1N = 35; lambda_basis = 2 * cfg.atom_z; nx1 = 2001;  x1_max = 5.;   x1_min = -12.
        # 1 & $1s$ & $-0.500000$ & $-0.767945$ & $-1.146608$ & $-2.000000$ \\
        # 1 & $2s$ & $-0.125000$ & $-0.155560$ & $-0.182185$ & $-0.222222$ \\
        # 1 & $3s$ & $-0.055556$ & $-0.064324$ & $-0.070868$ & $-0.080000$ \\
        # \midrule
        # 2 & $1s$ & $-2.000000$ & $-2.198163$ & $-3.410577$ & $-8.000000$ \\
        # 2 & $2s$ & $-0.500000$ & $-0.532968$ & $-0.650278$ & $-0.888889$ \\
        # 2 & $3s$ & $-0.222222$ & $-0.233431$ & $-0.264057$ & $-0.320000$ \\

        orth1N = 40; lambda_basis = 2 * cfg.atom_z; nx1 = 2001;  x1_max = 5.;   x1_min = -12.
        # 1 & $1s$ & $-0.500000$ & $-0.767945$ & $-1.146609$ & $-2.000000$ \\
        # 1 & $2s$ & $-0.125000$ & $-0.155560$ & $-0.182185$ & $-0.222222$ \\
        # 1 & $3s$ & $-0.055556$ & $-0.064324$ & $-0.070868$ & $-0.080000$ \\
        # \midrule
        # 2 & $1s$ & $-2.000000$ & $-2.198163$ & $-3.410577$ & $-8.000000$ \\
        # 2 & $2s$ & $-0.500000$ & $-0.532968$ & $-0.650278$ & $-0.888889$ \\
        # 2 & $3s$ & $-0.222222$ & $-0.233431$ & $-0.264057$ & $-0.320000$ \\

        # Z=2, P  , L=-0.5: [-8.       -0.888889 -0.32    ]
        # 1 & $1s$ & $-0.500000$ & $-0.767945$ & $-1.146609$ & $-2.000000$ \\
        # 1 & $2s$ & $-0.125000$ & $-0.155560$ & $-0.182185$ & $-0.222222$ \\
        # 1 & $3s$ & $-0.055556$ & $-0.064324$ & $-0.070868$ & $-0.080000$ \\
        # \midrule
        # 2 & $1s$ & $-2.000000$ & $-2.198163$ & $-3.410577$ & $-8.000000$ \\
        # 2 & $2s$ & $-0.500000$ & $-0.532968$ & $-0.650278$ & $-0.888889$ \\
        # 2 & $3s$ & $-0.222222$ & $-0.233431$ & $-0.264057$ & $-0.320000$ \\

        cfg.orth1N = orth1N
        dbg('lambda_basis')
        x1_opt = StepGridOpt(x1_min, x1_max, nx1)  #;
        log.dbg("x1_opt =", x1_opt)
        x1_grid = StepGrid.fromStepGridOpt(x1_opt)
        log.dbg("x1_grid =", x1_grid)

        # Quadratures -------------
        r1_min = LcrFactory.make_r1_min(x1_grid, use_c=0)
        w1 = WFQuadrLcr(x1_grid, r_min=r1_min)
        # cfg.wLcr1 = w1
        log.dbg("WFQuadrLcr 1=", w1)
        r1_grid = w1.getR()
        dbg('r1_grid.arr[0]')  # r1_grid.arr[0] = 1.3887943864964021e-11
        log.dbg("r1_grid =", r1_grid)

        # Lgrr orth
        from d2_py.d2e1.H2d1e import H2d1e
        cfg.eng_2d1e_n0m0 = H2d1e.calc_eng_2d1e(atom_z=cfg.atom_z, n=0, m=0)
        cfg.eng_2d1e_n1m0 = H2d1e.calc_eng_2d1e(atom_z=cfg.atom_z, n=1, m=0)
        # cfg.eng_2x1d1e = 2 * cfg.eng_1d1e_n1
        # cfg.eng_2x1d1e_anti = cfg.eng_1d1e_n2 + cfg.eng_1d1e_n1
        # lgrrOpt1 = LgrrOpt(m_2d=0, lambda_basis=lambda_basis, N=orth1N, L=None)  # from HyRadial2D_EigVecR_Test
        lgrrOpt1 = LgrrOpt(L=L, m_2d=0, lambda_=lambda_basis, N=orth1N)  # see v250723a-2d2eZ-PRA.tex
        self._lgrrOpt = lgrrOpt1
        # from scatt.jm_2008.jm.laguerre.lcr.LgrrOrth2D_Lcr import LgrrOrth2D_Lcr
        # orth1 = LgrrOrth2D_Lcr(w1, lgrrOpt1)
        orth1 = LgrrOrthLcr(w1, lgrrOpt1)

        OrthFactory.log.setDbg(False)
        # res1 = OrthFactory.calcMaxOrthErr(orth1, w1.getWithCR2())
        res1 = OrthFactory.calcMaxOrthErr(orth1, w1)
        dbg('res1')
        self.assertEquals(0, res1, 1e-10)

        self._arr = orth1
        log.dbg("LgrrOrthR =\n", self._arr)

    def test_1(self):
        # FlowTest.setMaxErr(1e-16)  # ok
        FlowTest.unlockMaxErr()  # ok
        FlowTest.lockMaxErr(1e-9)  # ok
        FlowTest.setLog(log)
        log.setDbg(DBG_ON)
        self.load_test_1()
        self.run_test_1()

    def run_test_1(self, potFunc):
        L = cfg.L

        # todo here M = 0  only
        arr = cast(LgrrOrth2D_Lcr, self._arr)
        wLcr = cast(WFQuadrLcr, arr.getQuadr())
        log.dbg("r1_grid=", wLcr)
        r1_grid = wLcr.getR()
        log.dbg("r1_grid=", r1_grid)

        # from d2_py.FuncAtom2D_X_crossover import FuncAtom2D_X_crossover
        # potFunc = FuncAtom2D_X_crossover(z=-cfg.atom_z)  # // f(r)=-2*Z*Ln(r)
        # from qm_math.func.simple.FuncPowAbsInt import FuncPowAbsInt
        # potFunc = FuncPowAbsInt(-cfg.atom_z, -1)  # // f(r)=-1./r

        from qm_math.func.FuncVec import FuncVec
        pot = FuncVec(r1_grid, potFunc)
        log.dbg("V2d(r)=", VecDbgView(pot))
        # pot0 = FuncVec(r1_grid, potFunc0)  # using it to calc <K> from sysH
        # log.dbg("ZEROs(r)=", VecDbgView(pot0))

        from d2_py.PotHMtrx2D_Lcr import PotHMtrx2D_Lcr
        sysH = PotHMtrx2D_Lcr(L=L, m_2d=0, basis=self._arr, pot=pot)

        eigEngs = sysH.getEigEngs()
        log.dbg("eigVal=", VecDbgView(eigEngs))
        log.dbg("eigVal=", eigEngs)
        # from d2_py.d2e1.H2d1e import H2d1e
        # eig_dimless = H2d1e.convert_au_to_dimless(E_au=eigEngs.arr, Z=cfg.atom_z)
        # log.dbg("eig_dimless=", eig_dimless)

        print(f'eigEngs(orth1N={cfg.orth1N}) =', eigEngs.arr[:4])
        # print(f'eig_dimless(orth1N={cfg.orth1N}) =', eig_dimless[:4])
        return eigEngs.arr



def main():

    results = {}
    for Z in [1.0, 2.0]:
        results[Z] = {}
        for name, L, make_pot in pots:
            cfg.atom_z = Z
            cfg.L = L
            runner = H_1e_D_Z_Lcr_run()
            runner.load_test_1()
            potFunc = make_pot(Z)
            engs = runner.run_test_1(potFunc)
            results[Z][name] = engs[:3]
            print(
                f"Z={Z:g}, {name:3s}, L={L:+.1f}: "
                f"{engs[:3]}"
            )
    #  make Table
    for Z in [1.0, 2.0]:
        for i, n in enumerate([1, 2, 3]):
            e3d = results[Z]["3D"][i]
            erk = results[Z]["RK"][i]
            ex = results[Z]["X"][i]
            ep = results[Z]["P"][i]
            print(
                f"{Z:g} & ${n}s$"
                f" & ${e3d:.6f}$"
                f" & ${erk:.6f}$"
                f" & ${ex:.6f}$"
                f" & ${ep:.6f}$"
                r" \\"
            )
        if Z == 1.0:
            print(r"\midrule")


# --- Add run test code block ---
if __name__ == "__main__":
    # Hy2d_Xover_Lcr_run().test_1()
    main()
