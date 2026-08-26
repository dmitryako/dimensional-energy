# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLinear.py Created : 2025-06-26 at 5:14 pm by Dmitry.A.Konovalov@gmail.com
import numpy as np

from javax.utilx.log.Log import Log
from qm_math.func.Func import Func

log = Log.getLog('R12_2D_delta_theta')

class R12_2D_delta_theta(Func):
    # def __init__(self, r1: float, r2: float, delta_theta: float):
    def __init__(self, r1: float, r2: float):
        self._r1: float = float(r1)
        self._r2: float = float(r2)
        # self._delta_theta: float = float(delta_theta)

    def calc(self, delta_theta):
        cos_delta = np.cos(delta_theta)
        log.dbg("cos_delta =", cos_delta)
        r12 = self._r1**2 + self._r2**2 - 2 * self._r1 * self._r2 *  cos_delta
        log.dbg("r12 =", r12)
        sqrt_r12 = np.sqrt(r12)
        log.dbg("sqrt_r12 =", sqrt_r12)
        return sqrt_r12


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    log.setDbg(False)
    f = R12_2D_delta_theta(r1=2.0, r2=0.5)
    for x in (-1.0, 0.0, 1.1):
        print(f"f({x}) = {f.calc(x):.6g}")

    x = np.array([-1.0, 0.0, 1.1])
    res = f.calc(x)
    print(res)
    res = f(x)
    print(res)

# R12_2D_delta_theta: cos_delta = 0.5403023058681398
# R12_2D_delta_theta: r12 = 3.1693953882637205
# R12_2D_delta_theta: sqrt_r12 = 1.7802795814881776
# f(-1.0) = 1.78028
# R12_2D_delta_theta: cos_delta = 1.0
# R12_2D_delta_theta: r12 = 2.25
# R12_2D_delta_theta: sqrt_r12 = 1.5
# f(0.0) = 1.5
# R12_2D_delta_theta: cos_delta = 0.4535961214255773
# R12_2D_delta_theta: r12 = 3.3428077571488455
# R12_2D_delta_theta: sqrt_r12 = 1.8283346950569104
# f(1.1) = 1.82833

# R12_2D_delta_theta: cos_delta = [0.54030231 1.         0.45359612]
# R12_2D_delta_theta: r12 = [3.16939539 2.25       3.34280776]
# R12_2D_delta_theta: sqrt_r12 = [1.78027958 1.5        1.8283347 ]
# [1.78027958 1.5        1.8283347 ]
