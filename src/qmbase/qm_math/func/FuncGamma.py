# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncGamma.py Created : 2025-06-29 at 3:34 pm by Dmitry.A.Konovalov@gmail.com
# © 2025 Dmitry A. Konovalov — All rights reserved.
# Port of java.math.func.FuncGamma
# Original package: math.func

from __future__ import annotations

import math
from typing import List

from javax.utilx.log.Log import Log
from qm_math.func.Func import Func


class FuncGamma(Func):
    """
    Direct, line-by-line translation of the original Java / Fortran‐style
    Gamma-function routine.  All arithmetic follows the source verbatim.
    """

    log = Log.getLog("FuncGamma")

    PI: float = 3.14159265358979324      #      DATA PI /3.14159 26535 89793 24D0/
    C1: float = 2.50662827463100050      #      DATA C1 /2.50662 82746 31000 50D0/
    C: List[float] = [                   #      DIMENSION C(0:15)
        41.624436916439068,
        -51.224241022374774,
        11.338755813488977,
        -0.747732687772388,
        +0.008782877493061,
        -0.000001899030264,
        +0.000000001946335,
        -0.000000000199345,
        +0.000000000008433,
        +0.000000000001486,
        -0.000000000000806,
        +0.000000000000293,
        -0.000000000000102,
        +0.000000000000037,
        -0.000000000000014,
        +0.000000000000006,
    ]

    # ------------------------------------------------------------------ #
    # FUNCTION WGAMMA(Z)                                                 #
    # ------------------------------------------------------------------ #
    def calc(self, Z: float) -> float:
        HF = 1.0 / 2.0                     #      PARAMETER (Z1 = 1, HF = Z1/2)

        F = 0.0                            # local variables (same names as Fortran)
        H = 0.0
        V = 0.0
        U = Z                              #      U=Z
        X = U                              #      X=U

        #      IF(GIMAG(U) .EQ. 0 .AND. -ABS(X) .EQ. INT(X)) THEN
        if -abs(X) == int(X):
            raise ValueError(
                self.log.error(f"ARGUMENT EQUALS NON-POSITIVE INTEGER = {X}")
            )

        #      ELSE ...
        if X >= 1.0:                       #       IF(X .GE. 1) THEN
            F = 1.0                        #        F=1
            V = U                          #        V=U
        elif X >= 0.0:                     #       ELSEIF(X .GE. 0) THEN
            F = 1.0 / U                    #        F=1/U
            V = 1.0 + U                    #        V=1+U
        else:                              #       ELSE
            F = 1.0                        #        F=1
            V = 1.0 - U                    #        V=1-U

        H = 1.0                            #       H=1
        S = self.C[0]                      #       S=C(0)
        for K in range(1, len(self.C)):    #       DO 1 K = 1,15
            H = ((V - K) / (V + (K - 1))) * H     #       H=((V-K)/(V+(K-1)))*H
            S += self.C[K] * H                     #   1   S=S+C(K)*H

        H = V + (4.0 + HF)                 #       H=V+(4+HF)
        V2 = (V - HF) * math.log(H)        #       (V-HF)*LOG(H)
        H = math.exp(V2 - H) * S * self.C1 #       H=C1*EXP((V-HF)*LOG(H)-H)*S

        if X < 0.0:                        #       IF(X .LT. 0) H=PI/(SIN(PI*U)*H)
            H = self.PI / (math.sin(self.PI * U) * H)

        WGAMMA = F * H                     #       WGAMMA=F*H
        return WGAMMA                      #       RETURN
