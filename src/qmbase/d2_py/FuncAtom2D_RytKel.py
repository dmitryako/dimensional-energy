# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncPowInt.py Created : 2025-06-26 at 5:16 pm by Dmitry.A.Konovalov@gmail.com
import math

import numpy as np
from qm_math.func.Func import Func

import math
import numpy as np

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import struve, y0


class FuncAtom2D_RytKel(Func):
    """
    Rytova--Keldysh potential.

    Positive like-charge kernel:
        W_RK(r) = pi/(2*kappa*rs) * [H_0(r/rs) - Y_0(r/rs)]

    Pair convention:
        V_RK(z,r) = z * W_RK(r)

    electron-nucleus attraction:
        z = -Z

    electron-electron repulsion:
        z = +1

    For kappa=1 and rs=1/2:
        large r:  V(z,r) -> z/r
        small r:  W_RK(r) ~ -2 [ln(r) + gamma]
    """

    def __init__(self, z: float, rs: float = 0.5, kappa: float = 1.0):
        self._z: float = float(z)
        self._rs: float = float(rs)
        self._kappa: float = float(kappa)

    def calc(self, x: float) -> float:
        r = float(x)
        if r <= 0.0:
            raise ValueError("Radius r must be positive.")

        u = r / self._rs
        W = (math.pi / (2.0 * self._kappa * self._rs)) * (
            struve(0, u) - y0(u)
        )
        return self._z * W






# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Expected:
    # f(0.01) = -8.0949
    # f(0.5) = -1.50922
    # f(1.0) = -0.881164
    # f(2.0) = -0.477382
    # f(0.01) = 8.0949
    # f(0.5) = 1.50922
    # f(1.0) = 0.881164
    # f(2.0) = 0.477382
    f = FuncAtom2D_RytKel(z=-1)      #
    for x in (0.01, 0.5, 1.0, 2.0):
        print(f"f({x}) = {f.calc(x):.6g}")


    f = FuncAtom2D_RytKel(z=+1)
    for x in (0.01, 0.5, 1.0, 2.0):
        print(f"f({x}) = {f.calc(x):.6g}")

    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import struve, y0


    def V_gauss(r, Z=1.0, r0=math.e / 2.0):
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


    def V_X_cross(r, Z=1.0, rc=0.5, r0=math.e / 2.0):
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


    def W_RK(r, rs=0.5, kappa=1.0):
        """
        Positive Rytova--Keldysh like-charge kernel.
        """
        r = np.asarray(r)
        u = r / rs
        return (np.pi / (2.0 * kappa * rs)) * (struve(0, u) - y0(u))


    def V_RK(r, Z=1.0, rs=0.5, kappa=1.0):
        """
        Attractive electron-nucleus RK potential:
            V = -Z W_RK(r)
        """
        return -Z * W_RK(r, rs=rs, kappa=kappa)


    # parameters
    Z = 1.0
    r0 = math.e / 2.0
    rc = 0.5
    rs = 0.5
    kappa = 1.0

    # avoid r=0
    r = np.linspace(0.01, 1.5, 3000)

    plt.figure(figsize=(7, 5))

    plt.plot(r, V_gauss(r, Z=Z, r0=r0), label=r"Gauss: $2Z\ln(r/r_0)$")
    plt.plot(r, V_coulomb(r, Z=Z), label=r"Coulomb: $-Z/r$")
    plt.plot(r, V_X_cross(r, Z=Z, rc=rc, r0=r0), label=r"Crossover $X$")
    plt.plot(r, V_RK(r, Z=Z, rs=rs, kappa=kappa), label=r"RK: $-Z W_{\rm RK}(r)$")

    plt.axvline(rc, linestyle="--", label=r"$r_c=1/2$")
    plt.axhline(0.0, linewidth=0.8)

    plt.xlabel(r"$r$")
    plt.ylabel(r"$V(r)$")
    plt.title(r"2D hydrogen potentials, $Z=1$, $r_s=1/2$, $\kappa=1$")
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