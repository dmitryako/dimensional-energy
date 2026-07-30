# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : PotHMtrx.py Created : 2025-06-30 at 10:49 am by Dmitry.A.Konovalov@gmail.com
# File: atom/energy/pw/PotHMtrx.py
# Copyright Dmitry.A.Konovalov
# Created: 21/11/2008, 11:42:11 AM (original Java version)

from atom.energy.HMtrx import HMtrx
from javax.utilx.log.Log import Log
from qm_math.func.arr.FuncArr import FuncArr
from qm_math.func.arr.IFuncArr import IFuncArr
from qm_math.func.FuncVec import FuncVec
from qm_math.mtrx.api.EigenSymm import EigenSymm
from qm_math.mtrx.api.Mtrx import Mtrx
from qm_math.vec.Vec import Vec
from atom.wf.WFQuadrD1 import WFQuadrD1

from abc import ABC, abstractmethod
from typing import Optional

from scatt.jm_2008.jm.laguerre.IWFuncArr import IWFuncArr

log = Log.getLog("PotHMtrx")

class PotHMtrx(HMtrx, ABC):
    # log = Log.getLog("PotHMtrx")

    def __init__(self, L: int, basis: IWFuncArr, pot: FuncVec, *, m_2d=None):
        super().__init__(rows=basis.size(), cols=basis.size())
        self._L = L  # update 250726. L=-1/2 for 2D, no need for separate m_2d
        self._m_2d = m_2d
        if m_2d is not None:
            self._L = m_2d  # mess? todo ?
            assert m_2d == 0, 'todo'
        self._basis = basis
        self._pot = pot
        self._S = Mtrx(rows=basis.size(), cols=basis.size())
        self._eigVec: Optional[FuncArr] = None
        self._quadr: Optional[WFQuadrD1] = None
        # REMEMBER!!! call "calc()"

    @abstractmethod
    def makePotH(self):
        pass

    def getL(self):
        return self._L

    def calc(self):
        H = self.makePotH()
        w = self._basis.getQuadr()
        for r in range(self._basis.size()):
            log.dbg("row=", r)
            fr = self._basis.getFunc(r)
            for c in range(r, self._basis.size()):
                log.dbg("row=", r)
                log.dbg("col=", c)
                fc = self._basis.getFunc(c)

                sij = w.calcInt(fr, fc)  # for non-diag
                self._S.set(r, c, sij)
                self._S.set(c, r, sij)

                kinE = H.calcKin(self._L, fr, fc)
                log.dbg("kinE=", kinE)
                potE = H.calcPot(self._pot, fr, fc)
                log.dbg("potE=", potE)
                E = kinE + potE
                log.dbg("E=", E)
                self.set(r, c, E)
                self.set(c, r, E)

    def getEigEngs_withS(self):
        # gen_eig
        H_np = self.mtrx
        S_np = self._S.mtrx
        from atom.energy.HMtrx import solve_gen_eig, check_S_orthonormality
        E, C = solve_gen_eig(H_np, S_np)  # ascending; E[0] is ground state
        print("Ground state energy:", E[0])
        print("S-ortho error:", check_S_orthonormality(S_np, C[:, :6]))
        sysEngs = Vec(E)
        # return self.getEigVal(overwrite=False)
        return sysEngs

    def getEigWfs(self):
        if self._eigVec is None:
            self.loadEigVec()
        return self._eigVec

    def loadEigVec(self):
        x = self._basis.getX()
        self._eigVec = FuncArr(x, self._basis.size())

        # f_i = SUM_j C_ji * lgrrN(j)
        eig = self.eigSymm()
        C: Mtrx = eig.getV()

        # double[][] C = v.getArr2D();
        for i in range(self._basis.size()):
            f_i = FuncVec(x)
            for j in range(self._basis.size()):
                # f_i.addMultSafe(C[i][j], lgrrN.getFunc(j));  # this is WRONG!!!
                # f_i.addMultSafe(C[j][i], basis.getFunc(j));   # this is correct!!!
                f_i.addMultSafe(C.get(j, i), self._basis.getFunc(j))   # this is correct!!!
            self._eigVec.set(i, f_i)

        # fix sign
        for i in range(self._basis.size()):
            f_i = self._eigVec.get(i)
            if f_i.get(0) < 0 or f_i.get(1) < 0:  # check first and second
                f_i.mult(-1)

    def getBasis(self):
        return self._basis

    def getPot(self):
        return self._pot

    def getQuadr(self):
        return self._quadr

    def setQuadr(self, quadr: WFQuadrD1):
        self._quadr = quadr
