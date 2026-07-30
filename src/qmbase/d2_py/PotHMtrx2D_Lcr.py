# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : PotHMtrxLcr.py Created : 2025-06-30 at 12:09 pm by Dmitry.A.Konovalov@gmail.com
# Copyright dmitry.konovalov@jcu.edu.au Date: 21/11/2008, Time: 11:43:31

from atom.energy.pw.PotH import PotH
from atom.energy.pw.PotHMtrx import PotHMtrx
from atom.energy.pw.lcr.PotHLcr import PotHLcr


class PotHMtrx2D_Lcr(PotHMtrx):
    def __init__(self, *, L, m_2d, basis, pot, quadr=None):
        """
        Overloaded constructor.
        If quadr is provided, uses it; otherwise gets quadr from basis.
        """
        super().__init__(L + abs(m_2d), basis, pot)
        if quadr is not None:
            self.setQuadr(quadr)
        else:
            self.setQuadr(basis.getQuadr())
        self.calc()

    def makePotH(self):
        # todo: NOTE !!!  PotH2D_Lcr is not needed: == PotHLcr, just uses L=-1/2
        return PotHLcr(self.getQuadr())

    @staticmethod
    def get_n0_m0_Lgrr_lambda(atom_z):
        # for z=1, lambda=4
        return PotHMtrx2D_Lcr.get_n_m_Lgrr_lambda(atom_z=atom_z, n=0, m=0)

    @staticmethod
    def get_n_m_Lgrr_lambda(*, atom_z, n, m):
        return 2 * atom_z / (n + abs(m) + 0.5)
