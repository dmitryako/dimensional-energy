# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncPowInt.py Created : 2025-06-26 at 5:16 pm by Dmitry.A.Konovalov@gmail.com
import math

import numpy as np
from qm_math.func.Func import Func

import math
import numpy as np

class FuncAtom2D_X_crossover(Func):
    """
    C1-matched log--Coulomb crossover potential.

    General pair form:
        V_X(z,r) =
            -2*z*log(r/r0),   r <= rc
             z/r,             r >= rc

    with
        rc = 1/2
        r0 = e/2

    Conventions:
        electron-nucleus attraction: z = -Z
            small r:  +2Z log(r/r0)
            large r:  -Z/r

        electron-electron repulsion: z = +1
            small r:  -2 log(r/r0)
            large r:  +1/r
    """

    def __init__(self, z: float, r0: float = math.e / 2.0, rc: float = 0.5):
        self._z: float = float(z)
        self._r0: float = float(r0)
        self._rc: float = float(rc)
        # The default rc=1/2, r0=e/2 gives C1 matching.
        # If rc is changed, r0=e/2 is no longer the C1-matched value.

    #     def V_X_cross(r, Z=1.0, rc=0.5, r0=math.e / 2):
    #         """
    #         C1-matched log--Coulomb crossover:
    #             V = 2 Z ln(r/r0),  r <= rc
    #             V = -Z/r,          r >= rc
    #         """
    #         r = np.asarray(r)
    #         return np.where(
    #             r <= rc,
    #             2.0 * Z * np.log(r / r0),
    #             -Z / r
    #         )
    def calc(self, x: float) -> float:
        r = float(x)
        if r <= 0.0:
            raise ValueError("Radius r must be positive.")

        if r <= self._rc:
            return (-2.0) * self._z * math.log(r / self._r0)
        else:
            return self._z / r



# class FuncAtom2D_RK_crossover(Func): # my old for 2d Log(r/r0)
#     # V(Z, r) = -2 * Z * ln(r)  // in atomic units
#     # todo!!! Note I was using r0=1 as default
#     # r0=e/2 is from matching Vc=-1/r (large r) and Vg=2*ln(r/r0) for small r.
#     # at rc = 1/2. first derivs are the same for Vc'=Vg'
#     # then Vc(rc) = Vg(rc) gives r0=e/2
#     def __init__(self, z: float, r0=math.e/2):
#         self._z: float = float(z)
#         self._r0 = r0
#     def calc(self, x: float) -> float:
#         v = (-2) * self._z * np.log(x/self._r0)  #
#         return v
#
# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Expected:
    # f(0.01) = -9.82405
    # f(0.5) = -2
    # f(1.0) = -1
    # f(2.0) = -0.5
    f = FuncAtom2D_X_crossover(z=-1)      #
    for x in (0.01, 0.5, 1.0, 2.0):
        print(f"f({x}) = {f.calc(x):.6g}")


    f = FuncAtom2D_X_crossover(z=+1)
    # f(0.01) = 9.82405
    # f(0.5) = 2
    # f(1.0) = 1
    # f(2.0) = 0.5
    for x in (0.01, 0.5, 1.0, 2.0):
        print(f"f({x}) = {f.calc(x):.6g}")

    import math
    import numpy as np
    import matplotlib.pyplot as plt


    def V_gauss(r, Z=1.0, r0=math.e / 2):
        """
        Pure intrinsic 2D Gauss electron-nucleus attraction:
            V = 2 Z ln(r/r0)
        """
        r = np.asarray(r)
        return 2.0 * Z * np.log(r / r0)


    def V_coulomb(r, Z=1.0):
        """
        Planar Coulomb electron-nucleus attraction:
            V = -Z/r
        """
        r = np.asarray(r)
        return -Z / r


    def V_X_cross(r, Z=1.0, rc=0.5, r0=math.e / 2):
        """
        C1-matched log--Coulomb crossover:
            V = 2 Z ln(r/r0),  r <= rc
            V = -Z/r,          r >= rc
        """
        r = np.asarray(r)
        return np.where(
            r <= rc,
            2.0 * Z * np.log(r / r0),
            -Z / r
        )


    # grid
    r = np.linspace(0.01, 4.0, 2000)

    Z = 1.0
    rc = 0.5
    r0 = math.e / 2.0

    plt.figure(figsize=(7, 5))

    plt.plot(r, V_gauss(r, Z=Z, r0=r0), label=r"Gauss: $2Z\ln(r/r_0)$")
    plt.plot(r, V_coulomb(r, Z=Z), label=r"Coulomb: $-Z/r$")
    plt.plot(r, V_X_cross(r, Z=Z, rc=rc, r0=r0), label=r"Crossover $X$")

    plt.axvline(rc, linestyle="--", label=r"$r_c=1/2$")
    plt.axhline(0.0, linewidth=0.8)

    plt.xlabel(r"$r$")
    plt.ylabel(r"$V(r)$")
    plt.title(r"2D log--Coulomb crossover, $Z=1$, $r_0=e/2$")
    plt.ylim(-10, 3)
    plt.legend()
    plt.tight_layout()
    plt.show()




'''

\section{Reference length from log--Coulomb matching}
\label{app:log_coulomb_matching}

In the pure intrinsic 2D Gauss-law model the logarithmic interaction
contains an arbitrary reference length \(r_0\).  Changing \(r_0\) shifts
all energies by a constant, without changing the wavefunctions.  For the
comparisons in this work it is useful to fix this otherwise arbitrary
constant by matching the short-range logarithmic interaction to a
large-range Coulomb tail.

In atomic units, the intrinsic 2D logarithmic interaction between two
charges \(q_i\) and \(q_j\) may be written
\begin{equation}
V_{ij}^{G}(r)
=
-2q_iq_j\ln\!\left(\frac{r}{r_0}\right),
\label{eq:Vij_G_log}
\end{equation}
where the electron has charge \(q_e=-1\) and the nucleus has charge
\(q_N=Z\).  The corresponding Coulomb tail is
\begin{equation}
V_{ij}^{C}(r)
=
\frac{q_iq_j}{r}.
\label{eq:Vij_C_tail}
\end{equation}
Thus Eq.~\eqref{eq:Vij_G_log} gives
\[
V_{Ne}^{G}(r)=2Z\ln\!\left(\frac{r}{r_0}\right),
\qquad
V_{ee}^{G}(r)=-2\ln\!\left(\frac{r}{r_0}\right),
\]
while Eq.~\eqref{eq:Vij_C_tail} gives the usual attractive
electron--nucleus tail \(-Z/r\) and repulsive electron--electron tail
\(1/r\).

We define a simple log--Coulomb crossover by requiring both the potential
and its first derivative to be continuous at a matching radius \(r_c\).
Derivative matching gives
\begin{equation}
\left.
\frac{d}{dr}
\left[
-2q_iq_j\ln\!\left(\frac{r}{r_0}\right)
\right]
\right|_{r=r_c}
=
\left.
\frac{d}{dr}
\left[
\frac{q_iq_j}{r}
\right]
\right|_{r=r_c}.
\end{equation}
For \(q_iq_j\ne0\), the charge factor cancels and one obtains
\begin{equation}
-\frac{2}{r_c}
=
-\frac{1}{r_c^2},
\qquad
r_c=\frac12 .
\label{eq:rc_log_coulomb}
\end{equation}
Continuity of the potential then requires
\begin{equation}
-2q_iq_j\ln\!\left(\frac{r_c}{r_0}\right)
=
\frac{q_iq_j}{r_c}.
\end{equation}
Again the charge factor cancels.  With \(r_c=1/2\),
\begin{equation}
\ln\!\left(\frac{1/2}{r_0}\right)=-1,
\qquad
r_0=\frac{\mathrm e}{2},
\label{eq:r0_log_coulomb}
\end{equation}
where \(\mathrm e\) is Euler's number.  Thus the same matching radius and
reference length work for both attractive and repulsive pair
interactions.

The resulting \(C^1\)-matched crossover interaction is
\begin{equation}
V_{ij}^{X}(r)
=
\begin{cases}
-2q_iq_j\ln\!\left(\dfrac{2r}{\mathrm e}\right),
& r\le \dfrac12,\\[8pt]
\dfrac{q_iq_j}{r},
& r\ge \dfrac12 .
\end{cases}
\label{eq:Vij_crossover}
\end{equation}
For the electron--nucleus interaction this gives
\begin{equation}
V_{Ne}^{X}(Z,r)
=
\begin{cases}
2Z\ln\!\left(\dfrac{2r}{\mathrm e}\right),
& r\le \dfrac12,\\[8pt]
-\dfrac{Z}{r},
& r\ge \dfrac12 ,
\end{cases}
\label{eq:Ven_crossover}
\end{equation}
and for the electron--electron interaction
\begin{equation}
V_{ee}^{X}(r)
=
\begin{cases}
-2\ln\!\left(\dfrac{2r}{\mathrm e}\right),
& r\le \dfrac12,\\[8pt]
\dfrac{1}{r},
& r\ge \dfrac12 .
\end{cases}
\label{eq:Vee_crossover}
\end{equation}
This model is not the full Rytova--Keldysh interaction.  It is a simple
analytic crossover model with the same limiting structure: logarithmic at
short range and Coulombic at long range.


'''