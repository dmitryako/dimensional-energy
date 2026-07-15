# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : OrthFactory.py Created : 2025-06-30 at 1:22 pm by Dmitry.A.Konovalov@gmail.com
# Copyright dmitry.konovalov@jcu.edu.au Date: 11/07/2008, Time: 10:37:57
# Copyright dmitry.konovalov@jcu.edu.au Date: 11/07/2008, Time: 10:37:57

import math

from javax.utilx.log.Log import Log
from qm_math.func.FuncVec import FuncVec
from qm_math.func.arr.FuncArr import FuncArr
from qm_math.func.arr.IFuncArr import IFuncArr
from qm_math.func.arr.NormFuncArr import NormFuncArr
from qm_math.integral.Quadr import Quadr
from qm_math.mtrx.api.Mtrx import Mtrx
from qm_math.vec.Vec import Vec


class OrthFactory:
    log = Log.getLog('OrthFactory')

    @staticmethod
    def keepN(wf: FuncVec, quadr, basis: IFuncArr) -> FuncVec:
        x = quadr.getX()
        res = FuncVec(x)
        for i in range(basis.size()):
            fi = basis.getFunc(i)
            # log.dbg("fi=", fi)
            dS = quadr.calcInt(wf, fi)
            # log.dbg("d=", dS)
            res.addMultSafe(dS, fi)
            # log.dbg("wf =", wf)
            # log.dbg("res=", res)
        return res

    @staticmethod
    def keepN_arr(dest: IFuncArr, quadr, basis: IFuncArr):
        for i in range(dest.size()):
            wf = dest.getFunc(i)
            OrthFactory.log.dbg("wf=", wf)
            wfN = OrthFactory.keepN(wf, quadr, basis)
            OrthFactory.log.dbg("wfN=", wfN)
            dest.setFunc(i, wfN)
        return

    @staticmethod
    def _calcMaxOrthonErr_normFuncArr(arr: NormFuncArr) -> float:  # not used! todo?
        return OrthFactory.calcMaxOrthErr(arr, arr.getNormQuadr())

    @staticmethod
    def calcMaxOrthErr(arr: FuncArr, w: Quadr) -> float:
        assert isinstance(w, Quadr), "w is not Quadr: new dk250530"
        res = 0
        for n in range(arr.size()):
            for n2 in range(n + 1):
                # norm = w.calcInt(arr.get(n), arr.get(n2))
                f1 = arr[n].getY()
                f2 = arr[n2].getY()
                norm = w.calcInt(f1, f2)
                # OrthFactory.log.dbg(f"norm error[{n}, {n2}]=" + str(float(norm)))
                if n == n2:
                    norm = abs(1.0 - norm)
                else:
                    norm = abs(norm)
                OrthFactory.log.dbg(f"norm error[{n}, {n2}]=" + str(float(norm)))
                if res < norm:
                    res = norm
                    OrthFactory.log.dbg(f"New max err =" + str(float(norm)))
        return res

    # http://en.wikipedia.org/wiki/Gram-Schmidt_process
    @staticmethod
    def makeOrthRotate(from_arr: FuncArr, w):
        # via Gram-Schmidt
        BB = Mtrx(from_arr.size(), from_arr.size())
        for r in range(from_arr.size()):
            for c in range(r + 1):
                q = w.calcInt(from_arr.get(r), from_arr.get(c))
                BB.set(r, c, q)
                BB.set(c, r, q)
        OrthFactory.log.dbg("BB=\n", BB)

        CC = [None] * from_arr.size()
        # F_0 = B_0
        i = 0
        CC[i] = Vec(i + 1)
        CC[i].set(i, 1.0)  # i:=0
        i += 1

        # F_1 = B_1 + C_0 * B_0
        # <F_1 B_0> = BB_01 + C_0 BB_00 = 0
        CC[i] = Vec(i + 1)
        CC[i].set(i, 1.0)
        CC[i].set(i - 1, -BB.get(0, 1) / BB.get(0, 0))
        i += 1

        # F_i = B_i + SUM(j<i) C_j * B_j
        # <F_i B_j'> = BB_ij' + SUM(j<i) C_j BB_jj'= 0
        # M * C = D
        # C = M^-1 * D
        while i < from_arr.size():
            OrthFactory.log.dbg("i = ", i)
            D = Vec(i)
            M = Mtrx(i, i)
            for j in range(i):
                D.set(j, -BB.get(i, j))
                for J in range(i):
                    M.set(j, J, BB.get(j, J))
            OrthFactory.log.dbg("M=\n", M)
            OrthFactory.log.dbg("D = ", D)
            M = M.inverse()
            OrthFactory.log.dbg("M.inverse=\n", M)
            # Vec C = new Vec(i)
            # C.mult(M, D);   log.dbg("C=M*D=", C)
            C = M.mult(D)
            OrthFactory.log.dbg("C=M*D=", C)
            CC[i] = Vec(i + 1)
            CC[i].set(i, 1)
            for j in range(i):
                CC[i].set(j, C.get(j))
            i += 1

        res = OrthFactory.makeOrth(from_arr, CC)
        OrthFactory.norm(res, w)
        from_arr.setArr(res)

    @staticmethod
    def makeOrth(from_arr: FuncArr, C: list[Vec]) -> FuncArr:
        x = from_arr.getX()
        res = FuncArr(x, from_arr.size())
        for ix in range(x.size()):
            for i in range(from_arr.size()):
                Cj = C[i]
                sum_val = 0
                N = Cj.size()
                for j in range(N):
                    sum_val += Cj.get(j) * from_arr.get(j).get(ix)
                res.get(i).set(ix, sum_val)
        return res

    # http://en.wikipedia.org/wiki/Gram-Schmidt_process
    @staticmethod
    def makeOrthGram(from_arr: FuncArr, w):
        res = from_arr.copyDeepY()
        OrthFactory.norm(res, w)
        for k in range(1, res.size()):  # NOTE: k=1
            vk = res.get(k)
            uk = vk.copyY()
            for j in range(k):
                uj = res.get(j)
                # OrthFactory.log.dbg("uj=", uj)
                vu = w.calcInt(vk, uj)
                # OrthFactory.log.dbg("vu=", vu)
                uu = w.calcInt(uj, uj)
                # OrthFactory.log.dbg("uu=", uu)
                uk.addMultSafe(-vu / uu, uj)
            norm_val = w.calcInt(uk, uk)
            # OrthFactory.log.dbg("norm =", norm_val)
            uk.mult(1.0 / math.sqrt(norm_val))
            # OrthFactory.log.dbg("uk =", uk)
            res.set(k, uk)

            # DEBUG
            # uk = res.get(k)
            # for j in range(k + 1):
            #     uj = res.get(j)
            #     OrthFactory.log.dbg("uj=", uj)
            #     ukj = w.calcInt(uk, uj)
            #     OrthFactory.log.dbg("ukj=", ukj)
            #     uu = w.calcInt(uj, uj)
            #     OrthFactory.log.dbg("uu=", uu)
        from_arr.copyFrom(0, res, 0, res.size())

    @staticmethod
    def norm(arr: FuncArr, w):
        for r in range(arr.size()):
            f = arr.get(r)
            s = w.calcInt(f, f)
            s = 1.0 / math.sqrt(s)
            f.mult(s)
