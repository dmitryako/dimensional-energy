# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FuncLinear.py Created : 2025-06-26 at 5:14 pm by Dmitry.A.Konovalov@gmail.com
import numpy as np

from javax.utilx.log.Log import Log
from qm_math.func.Func import Func
from qm_math.func.d2.R12_2D_delta_theta import R12_2D_delta_theta

log = Log.getLog('OneOverR12_2D_delta_theta')
_eps = 1e-10

class OneOverR12_2D_delta_theta(R12_2D_delta_theta):
    def __init__(self, r1: float, r2: float):
        super().__init__(r1=r1, r2=r2)

    def calc(self, delta_theta):
        r12 = super().calc(delta_theta)
        log.dbg("r12 =", r12)
        # res = 1 / r12
        # res = 1 / np.max(r12, _eps)
        res = 1 / np.maximum(r12, _eps)
        log.dbg("1 / r12 =", r12)
        return res


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    log.setDbg(False)
    f = OneOverR12_2D_delta_theta(r1=2.0, r2=0.5)
    for x in (-1.0, 0.0, 1.1):
        print(f"f({x}) = {f.calc(x):.6g}")

    x = np.array([-1.0, 0.0, 1.1])
    res = f.calc(x)
    print(res)
    res = f(x)
    print(res)

# OneOverR12_2D_delta_theta: r12 = 1.7802795814881776
# OneOverR12_2D_delta_theta: 1 / r12 = 1.7802795814881776
# f(-1.0) = 0.56171
# OneOverR12_2D_delta_theta: r12 = 1.5
# OneOverR12_2D_delta_theta: 1 / r12 = 1.5
# f(0.0) = 0.666667
# OneOverR12_2D_delta_theta: r12 = 1.8283346950569104
# OneOverR12_2D_delta_theta: 1 / r12 = 1.8283346950569104
# f(1.1) = 0.546946
# OneOverR12_2D_delta_theta: r12 = [1.78027958 1.5        1.8283347 ]
# OneOverR12_2D_delta_theta: 1 / r12 = [1.78027958 1.5        1.8283347 ]
# [0.56170953 0.66666667 0.54694581]
# OneOverR12_2D_delta_theta: r12 = [1.78027958 1.5        1.8283347 ]
# OneOverR12_2D_delta_theta: 1 / r12 = [1.78027958 1.5        1.8283347 ]
# [0.56170953 0.66666667 0.54694581]