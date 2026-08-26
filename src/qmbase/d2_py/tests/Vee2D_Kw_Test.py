# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : JmPotEigVecRTest.py Created : 2025-06-30 at 5:35 pm by Dmitry.A.Konovalov@gmail.com
import math
from typing import cast
from scipy.special import ellipk

import numpy as np

from _new25.dbg import dbg, set_dbg
from atom.energy.pw.lcr.PotHLcr import PotHLcr
from atom.wf.WFQuadrR import WFQuadrR
from javax.utilx.log.Log import Log
from javax.utilx.log.kiss.KissLog import KissLog
from project.workflow.task.test.FlowTest import FlowTest
from qm_math.func.d2.OneOverR12_2D_delta_theta import OneOverR12_2D_delta_theta
from qm_math.vec.Vec import Vec
from qm_math.vec.VecDbgView import VecDbgView
from qm_math.vec.grid.StepGrid import StepGrid
from qm_math.vec.grid.StepGridOpt import StepGridOpt

"""
Copyright dmitry.konovalov@jcu.edu.au Date: 21/11/2008, Time: 15:42:20
"""

log = Log.getLog('Vee2D_Kw_Test')


# def compute_Kw(w):
#     """
#     Compute the complete elliptic integral of the first kind K(w)
#     using SciPy's ellipk, which expects the parameter m = w^2.
#
#     Parameters:
#     w : float or np.ndarray
#         Argument in the range [0, 1)
#
#     Returns:
#     K : float or np.ndarray
#         Value of the complete elliptic integral K(w)
#     """
#     m = w ** 2
#     return ellipk(m)

def calc_2d_1over_r12(rr1, rr2, eps=1e-10):
    rr1 = np.array([rr1])
    rr2 = np.array([rr2])
    res = calc_2d_1over_r12_np(rr1, rr2, eps=eps)
    return res[0]

def calc_2d_1over_r12_np(rr1, rr2, eps=1e-10):
    dbg([rr1, rr2])
    # a = rr1**2 + rr2**2
    # log.dbg("a =", a)
    # b = rr1 * rr2
    # log.dbg("b =", b)
    sqrt_ab = (rr1 + rr2)   # <-- sqrt(a + b)
    dbg(sqrt_ab)
    sqrt_ab = np.clip(sqrt_ab, a_min=eps, a_max=None)
    dbg(sqrt_ab)
    log.dbg("b =", sqrt_ab)
    # w = 2 * np.sqrt(rr1 * rr2) / (rr1 + rr2)
    w = 2 * np.sqrt(rr1 * rr2) / sqrt_ab  # ab was clipped
    log.dbg("w =", w)
    m = w ** 2
    dbg(m)
    m = np.clip(m, a_min=0, a_max=1 - eps)
    dbg(m)
    res1 = ellipk(m)  # todo <------- OK
    dbg(res1)
    scale = 2 / sqrt_ab
    dbg(scale)
    res1 = scale * res1
    dbg(res1)
    # eps=1e-10: (1001, 1041) float64 min= 0.019153494221022735 mean= 4303.059138683469 max= 1505989.446603124
    # res2 = scale * ellipk(w)  # NOT OK
    # res = np.array([res1, res2])
    # Vee2D_Kw_Test: res = 2.1565156, 2.4413416
    # log.dbg("res =", res)
    log.dbg("res1 =", res1)
    # return res[0]
    return res1


class cfg(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    # run_self_test = True  # todo RE-RUN before starting new dev
    run_self_test = False  # todo
    seed = 1
    atom_z = 1.  # todo <----- 1, 1d-Hydrogen
    L = -1/2
    max_norm_err = 1e-9

class Vee2D_Kw_Test(FlowTest):
    # todo: from v25_qm/qm25/d2_py/tests/Hy2D_EigVecLcr_Test.py
    def __init__(self):
        super().__init__(Vee2D_Kw_Test)
        self._arr = None
        self._w = None
        set_dbg(False)
        # FlowTest.setMaxErr(1e-12)  # ok
        FlowTest.unlockMaxErr()  # ok
        FlowTest.lockMaxErr(1e-12)  # ok
        FlowTest.setLog(log)
        log.setDbg(False)
        from qm_math.vec.DbgView import DbgView
        DbgView.show_digs = 8
        KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))
        KissLog.register_formatter(np.ndarray, lambda v: str(VecDbgView(v)))
        # Log.getLog("WFQuadrLcr").setDbg(False)  # on/off for PotHMtrx

    def test_1(self):
        FlowTest.unlockMaxErr()  # ok
        FlowTest.lockMaxErr(1e-20)  # ok
        FlowTest.setLog(log)
        log.setDbg(False)
        # todo: NOTE WFQuadrD1(QuadrPts5) Why not Pts7? try?
        x1_min = 0.
        x1_max = math.pi

        # Vee2D_Kw_Test: err = expected=0 actual=8.881784197001252e-16, err=8.881784197001252e-16, abs_tol=1e-20
        # nx1 = 101

        # Vee2D_Kw_Test: err = expected=0 actual=8.881784197001252e-16, err=8.881784197001252e-16, abs_tol=1e-20
        # nx1 = 121

        # Vee2D_Kw_Test: err = expected=0 actual=0.0, err=0.0, abs_tol=1e-20
        nx1 = 133  # !!!

        grid_opt = StepGridOpt(x1_min, x1_max, nx1)  #;
        log.dbg("x1_opt =", grid_opt)
        x1_grid = StepGrid.fromStepGridOpt(grid_opt)
        log.dbg("x1_grid =", x1_grid)

        # Quadratures -------------
        w1 = WFQuadrR(x1_grid)
        # cfg.wLcr1 = w1
        log.dbg("WFQuadrR 1=", w1)

        r1 = 0.5
        r2 = 1.5
        potFunc = OneOverR12_2D_delta_theta(r1=r1, r2=r2)  # // f(r)=-1./r
        from qm_math.func.FuncVec import FuncVec
        pot1 = FuncVec(x1_grid, potFunc)
        log.dbg("-1/r12=", VecDbgView(pot1))

        def my_func(rr1, rr2, delta_theta):
            cos_delta = np.cos(delta_theta)
            log.dbg("cos_delta =", cos_delta)
            rr12 = rr1 ** 2 + rr2 ** 2 - 2 * rr1 * rr2 * cos_delta
            log.dbg("r12 =", rr12)
            sqrt_r12 = np.sqrt(rr12)
            log.dbg("sqrt_r12 =", sqrt_r12)
            res = 1 / sqrt_r12
            log.dbg("1/sqrt_r12 =", res)
            res = Vec(res)
            log.dbg("1/sqrt_r12 =", res)
            return res

        pot2 = my_func(r1, r2, x1_grid.arr)
        log.dbg("1/r12=", pot2)

        int1 = w1.calcInt(pot1)
        int2 = w1.calcInt(pot2)
        log.dbg("1/r12=", np.array([int1, int2]))

        # Vee2D_Kw_Test: res = 2.1565156, 2.4413416
        my_ints = calc_2d_1over_r12(r1, r2)
        log.dbg("int1=", int1)
        log.dbg("my  =", my_ints)

        # ----------------
        err = abs(my_ints - int1)
        dbg('err')
        #
        self.assertEquals("err =", 0, err, True)
        assert err < 1e-20
        # self.assertEquals("wf0_kin(1s) =", +2., wf0_kin, True)
        # self.assertEquals("tot_e(0s) =", -2., tot_e, True)



# --- Add run test code block ---
if __name__ == "__main__":
    Vee2D_Kw_Test().test_1()
    FlowTest.ok(Vee2D_Kw_Test)
