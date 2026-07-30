# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : H2d1e.py Created : 2025-07-26 at 9:38 am by Dmitry.A.Konovalov@gmail.com
import math

import numpy as np


class H2d1e:
    def __init__(self, atom_z):
        self.atom_z = atom_z
    @staticmethod
    def calc_eng_2d1e(*, atom_z, n=0, m=0):
        # todo: H3d1e.calc_eng_3d1e
        # todo: see v250706-hydrogen-radial-2d.tex
        # \beq
        # E_{n, m} = -\frac{1}{2 \left( n + |m| + \frac{1}{2} \right)^2}, \quad n = 0, 1, 2, \ldots
        # \eeq
        res = - 0.5 * float(atom_z)**2 / (n + abs(m) + 0.5)**2
        return float(res)


    @staticmethod
    def calc_eng_2d1e_Gauss(*, atom_z, n, r0, m): # r0=math.e/2,
        # todo: see v260625b_1eHe_2dLnR_EigVecLcr_run.py
        # v260625a_Hy2dLnR_EigVecLcr_run: eigVal= Vec[40] = {-0.3332761, 1.9362086, 2.9680705, 3.6446011, 4.1489628, ..., 14.7952880, 22.7055998, 41.7289993, 103.9494488, 561.2712703}
        # v260625a_Hy2dLnR_EigVecLcr_run: eig_dimless= Vec[40] = {0.5265091, 1.6612515, 2.1771825, 2.5154478, 2.7676286, ..., 8.0907912, 12.0459471, 21.5576468, 52.6678716, 281.3287824}

        # OLD: scale au(Z=1) to au(Z)
        # # E_{n0}^{\mathrm{au}}(Z)
        # # =
        # # Z E_{n0}^{\mathrm{au}}(1)
        # # -
        # # Z\ln Z .
        # engs = [-0.3332761, 1.9362086, 2.9680705, 3.6446011, 4.1489628]
        # E_n0 = engs[n]
        # res = atom_z * E_n0 - atom_z * np.log(atom_z)

        # & E_{n0}^{\mathrm{au}}(Z)
        # =
        #   2Z E_n^{(\mathrm{dl})}
        # -
        # 2Z \ln\!\left(2\sqrt{Z}\,r_0\right).
        # Scale from dimless:
        engs = [0.5265091, 1.6612515, 2.1771825, 2.5154478, 2.7676286]
        E_n0 = engs[n]
        res = 2 * atom_z * E_n0 - 2 * atom_z * np.log(2 * np.sqrt(atom_z) * r0)
        return float(res)

    @staticmethod
    def calc_eng_2d1e_Gauss_r0e(*, atom_z, n, m):
        return H2d1e.calc_eng_2d1e_Gauss(atom_z=atom_z, n=n, r0=math.e/2, m=m)

    @staticmethod
    def calc_eng_2d1e_Gauss_r01(*, atom_z, n, m):
        return H2d1e.calc_eng_2d1e_Gauss(atom_z=atom_z, n=n, r0=1, m=m)


    def convert_au_to_dimless_Z1(*, E_au, r0=math.e/2):
        # E_{n0}^{\mathrm{au}}
        # =
        # 2\,E_n^{(\mathrm{dl})}-\ln 4,
        # \qquad
        # E_n^{(\mathrm{dl})}
        # =
        # \tfrac12\,E_{n0}^{\mathrm{au}}+\ln 2,
        # res = 0.5 * E_au + np.log(2)
        res = 0.5 * E_au + np.log(2 * r0)
        return res
    def convert_au_to_dimless(*, E_au, Z, r0=math.e/2):
        # E_{n0}^{\mathrm{au}}
        # =
        # 2\,E_n^{(\mathrm{dl})}-\ln 4,
        # \qquad
        # E_n^{(\mathrm{dl})}
        # =
        # \tfrac12\,E_{n0}^{\mathrm{au}}+\ln 2,
        res = 0.5 * E_au / Z + np.log(2 * np.sqrt(Z) * r0)
        return res


    def convert_dimless_to_aud(*, E_dimless, Z, r0=math.e/2):
        # E_{n0}^{\mathrm{au}}
        # =
        # 2\,E_n^{(\mathrm{dl})}-\ln 4,
        # \qquad
        # E_n^{(\mathrm{dl})}
        # =
        # \tfrac12\,E_{n0}^{\mathrm{au}}+\ln 2,
        res = 2 * Z * E_dimless - 2 * Z * np.log(2 * np.sqrt(Z) * r0)
        return res
