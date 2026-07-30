# Copyright dmitry.konovalov@jcu.edu.au Date: 21/11/2008, Time: 11:44:49
# H(r) = (-1/2) {1/r d^2(r*)/dr^2 - L*(L+1)/r^2} + U(r)
# INTL r^2dr R(r)R(r)=1
#
# After R(r) = p(r) * P(r)
# -2(H(r)-U)=p2*P"+p1*P'+p0*P
#    p2=p                          *P"
#    p1=2p'+2/r*p                  *P'
#    p0=p"+2/r*p'-L*(L+1)/r^2*p    *P
#
# After P(r)=F(x); x=ln(y), y=c+r; dx=dy/y=dr/y
#    P'=F'/y;    P"=F"/y^2-F'/y^2
#
# -2(H(r)-U)=f2*F"+f1*F'+f0*F
#    p2=p                          *(F"/y^2-F'/y^2)
#    p1=2p'+2/r*p                  *(F'/y)
#    p0=p"+2/r*p'-L*(L+1)/r^2*p    *F
#
#    f2=p/y^2
#    f1=-p/y^2 + 2p'/y + 2/r*p/y
#    f0=p0
#
# Select p(r) such that f1==0;   2p'=p/y-2p/r
#    Check y=r;   2p'=p/r-2p/r=-p/r;  solution  p(r)=1/sqrt(r)
# p(r)=sqrt(y)*g  gives g'=-g/r; g=1/r; p=sqrt(y)/r
#
# f2=1/r*1/y^3/2
# f0=-1/4*1/r*1/y^3/2 - L(L+1)*sqrt(y)/r^3
#   Check y=r; f0=-1/4*1/r5/2 - L(L+1)/r^5/2=-(L+1/2)^2 * 1/r^5/2
#
# INTL r^2dr R*HR = INTL r^2 ydx sqrt(y)/r F*HF = INTL y^3/2 dx r F*HF
# (-1/2)*INTL dx [F" -{1/4 + L(L+1)*(y/r)^2}*F - 2*U(r)*F(x)]
#    check r->0; R=P(r)/r
#    check r->oo; R=P(r)/sqrt(r); x=ln(r)
#
# w are for 'x' integral
#
# d/dr R(r) = d/dr [sqrt(y) F] = 1/2 1/sqrt(y) F + sqrt(y) dx/dr F'(x) =***
# dx/dr = 1/y
# ***=  1/sqrt(y) [F/2 + F']
from typing import cast

import numpy as np

from atom.energy.pw.PotH import PotH
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from javax.utilx.log.Log import Log
from qm_math.vec.VecDbgView import VecDbgView

log = Log.getLog("PotHLcr")


class PotHLcr(PotH):
    # log = Log.getLog("PotHLcr")

    def __init__(self, quadr):
        super().__init__(quadr)
        self.quadr = cast(WFQuadrLcr, quadr)

    def calcKin(self, L, wf, wf2):
        """
        See derivation above
        HF = -1/2F" + 1/2 * {1/4 + L(L+1)*(y/r)^2} * F + y^2 * U(r) * F
        """
        log.dbg("wf =", VecDbgView(wf))
        log.dbg("wf2=", VecDbgView(wf2))
        kin = self.quadr.calc(wf, wf2)
        log.dbg("kin=", kin)
        kinL = 0
        if L != 0:  # todo! NOTE! in 2d-case L=-1/2, and LARGE kin + kinL==0
            cr2DivR2 = self.quadr.getCR2DivR2()
            # todo: NOTE! for lcr with c=0, should be getCR2DivR2==1, ok
            log.dbg("cr2DivR2=", cr2DivR2)
            kinL = self.quadr.calcWithDivR2(wf, wf2)
            log.dbg(f"kinL=L(L+1)*(y/r)^2= {kinL}")  # L(L+1)*(y/r)^2

        potL2 = 0.125 * kin + 0.5 * L * (L + 1) * kinL
        log.dbg("potL2=", potL2)
        # todo note! calcDrv2 is actuall Kin-energy, including (-1/2). calcDrv2<-- should be better name, e.g. calcKinDrv2?
        res = self.calcKinDrv2(wf, wf2)
        log.dbg("calcDrv2=", res)
        res += potL2
        # dbg('res')
        log.dbg("calcKin=", res)

        c = self.quadr.getLcrToR().c
        if c == 0 and L == -1/2:  # L=-1/2 is the hardest
            # assert L == -1/2, 'todo generic L'
            corr_r0 = self.calc_0_to_rmin_correction(wf, wf2)
            log.dbg("calcKin=", res)
            res += corr_r0

        log.dbg("calcKin=", res)
        return res

    def calc_0_to_rmin_correction(self, wf, wf2):
        corr1 = self._calc_0_to_rmin_unsymm(wf, wf2)
        corr2 = self._calc_0_to_rmin_unsymm(wf2, wf)
        res = 0.5 * (corr1 + corr2)
        log.dbg("corr1, corr2, res =", np.array([corr1, corr2, res]))
        return res

    def _calc_0_to_rmin_unsymm(self, wf, wf2):
        # NOTE!!! drv2 is not accurate first few points
        r_grid = self.quadr.getR().arr[:6]
        r0, _, _, _, r1, r2 = r_grid
        log.dbg("r0, r1, r2 =", np.array([r0, r1, r2]))

        # df1, df2 = wf.arr[:2] * wf2.getDrv2().arr[:2]
        # log.dbg("df1, df1 =", np.array([df1, df2]))

        # todo need to 1/r???? YES !!!
        df0, _, _, _, df1, df2 = wf.arr[:6] * wf2.getDrv2().arr[:6] / r_grid
        log.dbg("df0, df1, df1 =", np.array([df0, df1, df2]))

        slope = (df2 - df1) / (r2 - r1)
        log.dbg("slope =", slope)
        intercept = df0 - slope * r0  # from r0
        log.dbg("intercept =", intercept)

        corr_lin = (r0 - 0) * (df0 + intercept) / 2.
        log.dbg("corr_lin =", corr_lin)
        kin = (-1/2) * corr_lin
        log.dbg("kin =", kin)
        return kin

    def _calc_0_to_rmin_unsymm_BAD_v3(self, wf, wf2):
        # NOTE!!! drv2 is not accurate first few points
        r_grid = self.quadr.getR().arr[:5]
        r1, _, _, _, r2 = r_grid
        r0 = r1
        log.dbg("r0, r1, r2 =", np.array([r0, r1, r2]))

        # todo need to 1/r???? YES !!!
        df1, _, _, _, df2 = wf.arr[:5] * wf2.getDrv2().arr[:5] / r_grid
        df0 = df1
        log.dbg("df0, df1, df1 =", np.array([df0, df1, df2]))

        slope = (df2 - df1) / (r2 - r1)
        log.dbg("slope =", slope)
        intercept = df0 - slope * r0  # from r0
        log.dbg("intercept =", intercept)

        corr_lin = (r0 - 0) * (df0 + intercept) / 2.
        log.dbg("corr_lin =", corr_lin)
        kin = (-1/2) * corr_lin
        log.dbg("kin =", kin)
        return kin

    def _calc_0_to_rmin_unsymm_v2_BETTER(self, wf, wf2):
        # NOTE!!! drv2 is not accurate first few points
        r_grid = self.quadr.getR().arr[:5]
        r0, _, _, r1, r2 = r_grid
        log.dbg("r0, r1, r2 =", np.array([r0, r1, r2]))

        # df1, df2 = wf.arr[:2] * wf2.getDrv2().arr[:2]
        # log.dbg("df1, df1 =", np.array([df1, df2]))

        # todo need to 1/r???? YES !!!
        df0, _, _, df1, df2 = wf.arr[:5] * wf2.getDrv2().arr[:5] / r_grid
        log.dbg("df0, df1, df1 =", np.array([df0, df1, df2]))

        slope = (df2 - df1) / (r2 - r1)
        log.dbg("slope =", slope)
        intercept = df0 - slope * r0  # from r0
        log.dbg("intercept =", intercept)

        corr_lin = (r0 - 0) * (df0 + intercept) / 2.
        log.dbg("corr_lin =", corr_lin)
        kin = (-1/2) * corr_lin
        log.dbg("kin =", kin)
        return kin

    def _calc_0_to_rmin_unsymm_GOOD_BUT_NOT_PERFECT(self, wf, wf2):
        r_grid = self.quadr.getR().arr[:2]
        r1, r2 = r_grid
        log.dbg("r1, r2 =", np.array([r1, r2]))

        # todo need to 1/r???? YES !!!
        df1, df2 = wf.arr[:2] * wf2.getDrv2().arr[:2] / r_grid
        log.dbg("df1, df1 =", np.array([df1, df2]))

        slope = (df2 - df1) / (r2 - r1)
        log.dbg("slope =", slope)
        intercept = df1 - slope * r1
        log.dbg("intercept =", intercept)

        corr_lin = (r1 - 0) * (df1 + intercept) / 2.
        log.dbg("corr_lin =", corr_lin)
        kin = (-1/2) * corr_lin
        log.dbg("kin =", kin)
        return kin

    def calc_0_to_rmin_correction_BAD_v1(self, wf, wf2):
        r_grid = self.quadr.getR().arr[:2]
        r1, r2 = r_grid
        log.dbg("r1, r2 =", np.array([r1, r2]))
        # f1, f2 = wf.arr[:2] * wf2.arr[:2]
        # log.dbg("f1, f2 =", np.array([f1, f2]))

        df1, df2 = wf.getDrv().arr[:2] * wf2.getDrv().arr[:2]
        log.dbg("df1, df1 =", np.array([df1, df2]))

        # todo need to 1/r????
        df1, df2 = wf.getDrv().arr[:2] * wf2.getDrv().arr[:2] / r_grid
        log.dbg("df1, df1 =", np.array([df1, df2]))

        slope = (df2 - df1) / (r2 - r1)
        log.dbg("slope =", slope)
        # intercept = f1 - slope * x1
        intercept = df1 - slope * r1
        log.dbg("intercept =", intercept)

        corr_lin = (r1 - 0) * (df1 + intercept) / 2.
        log.dbg("corr_lin =", corr_lin)
        kin = (+1/2) * corr_lin  # remember F_i F''_j = - F'_i * F'_j
        log.dbg("kin =", kin)
        return kin

