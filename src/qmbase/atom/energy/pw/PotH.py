# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : PotH.py Created : 2025-06-30 at 12:02 pm by Dmitry.A.Konovalov@gmail.com
# atom/energy/pw/PotH.py

from abc import ABC, abstractmethod
from javax.utilx.log.Log import Log
from atom.wf.WFQuadrD1 import WFQuadrD1
from qm_math.func.FuncVec import FuncVec

log = Log.getLog("PotH")

class PotH(ABC):
    def __init__(self, quadr: WFQuadrD1):
        self.quadr = quadr
        # self.MAX_ERR = 1e-10
        # self.MAX_ERR = 1e-7 # somehow not 1e-10??
        # self.MAX_ERR = 1e-6 # somehow not 1e-10??
        self.MAX_ERR = 1e-4 # somehow not 1e-10?? todo !!!!

    @abstractmethod
    def calcKin(self, L: int, wf: FuncVec, wf2: FuncVec) -> float:
        pass

    def calcPot(self, pot: FuncVec, wf: FuncVec, wf2: FuncVec) -> float:
        res = self.quadr.calcInt(pot, wf, wf2)
        log.dbg("res=", res)
        return res

    def calcKin1D(self, wf: FuncVec, wf2: FuncVec) -> float:
        return self.calcKinDrv2(wf=wf, wf2=wf2)

    def calcKinDrv2(self, wf: FuncVec, wf2: FuncVec) -> float:
        # kin1 = self._calcDrv2_unsymm(wf, wf2)
        # kin2 = self._calcDrv2_unsymm(wf2, wf)
        kin1 = self._calcDrv2_unsymm(wf, wf2)
        kin2 = self._calcDrv2_unsymm(wf2, wf)
        kin = 0.5 * (kin1 + kin2)
        log.dbg("kin1 =", kin1)
        log.dbg("kin2 =", kin2)
        log.dbg("kin =", kin)
        return kin

    def _calcDrv2_unsymm(self, wf: FuncVec, wf2: FuncVec) -> float:
        # too hard via _calcDrv2_symm
        drv2 = self.quadr.calc(wf.getY(), wf2.getDrv2())
        log.dbg("drv2 =", drv2)
        res = -0.5 * drv2
        # res = 0.5 * drv2  # todo: assume 0, NO! not ok! 250727
        log.dbg("res=", res)
        return res

    def _calcDrv2_symm(self, wf: FuncVec, wf2: FuncVec) -> float:
        drv2 = self.quadr.calc(wf.getDrv(), wf2.getDrv())
        #          1 d^2
        # dir2 = - - --
        #          2 dr^2
        # note 0.5 instead of (-0.5), that is because U' * U' replaced U*U"
        # B            B               |B
        # I dr RR" = - I dr R'R' + RR' |  = - I drR'R' + RR'(B) - RR'(A)
        # A            A               |A
        wf1_0 = wf.getFirst()
        log.dbg("wf =", wf)
        log.dbg("wf1_0 =", wf1_0)
        wf2_d0 = wf2.getDrv().getFirst()
        log.dbg("wf2.getDrv() =", wf2.getDrv())
        log.dbg("wf2_d0 =", wf2_d0)
        corrA = wf1_0 * wf2_d0
        log.dbg("corrA =", corrA)

        # corrA = wf.getFirst() * wf2.getDrv().getFirst()
        corrB = wf.getLast() * wf2.getDrv().getLast()

        log.dbg("drv2 =", drv2)

        # todo: DONE! 250726
        # MAX_ERR = self.MAX_ERR
        # if abs(corrA) > MAX_ERR or abs(corrB) > MAX_ERR:
        #     log.setDbg(False)
        #     from qm_math.vec.VecDbgView import VecDbgView
        #     # log.dbg("r=self.quadr.getX() =", VecDbgView(r))
        #     log.dbg("drv2 =", drv2)
        #     log.dbg("wf =", VecDbgView(wf.getY()))
        #     log.dbg("wf2 =", VecDbgView(wf2.getY()))
        #     log.dbg("wf.getDrv =", VecDbgView(wf.getDrv()))
        #     log.dbg("wf2.getDrv =", VecDbgView(wf2.getDrv()))
        #
        #     # log.dbg("r.getFirst()=", r.getFirst())
        #     # log.dbg("r.getLast()=", r.getLast())
        #     log.dbg("wf.getFirst()=", wf.getFirst())
        #     log.dbg("wf.getLast()=", wf.getLast())
        #     log.dbg("wf2.getDrv().getFirst()=", wf2.getDrv().getFirst())
        #     log.dbg("wf2.getDrv().getLast()=", wf2.getDrv().getLast())
        # assert abs(corrA) < MAX_ERR, f'check calcKin2D: corrA={corrA} > {MAX_ERR}'
        # assert abs(corrB) < MAX_ERR, f'check calcKin2D: corrB={corrB} > {MAX_ERR}'

        # eigEngs[0] = expected=-2.0 actual=-1.9999999340369987, err=6.596300128336452e-08, abs_tol=1e-10
        # res = 0.5 * (drv2 + corrB - corrA)  # <--- wrong, see large err=6.596300128336452e-08,

        res = 0.5 * (drv2 - corrB + corrA)
        # res = 0.5 * drv2  # todo: assume 0, NO! not ok! 250727
        log.dbg("res=", res)
        return res
