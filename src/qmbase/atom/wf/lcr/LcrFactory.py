# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LcrFactory.py Created : 2025-06-26 at 2:40 pm by Dmitry.A.Konovalov@gmail.com

# lcr_factory.py  –– Port of atom.wf.lcr.LcrFactory
# --------------------------------------------------
# All three static methods from the Java class are reproduced
# with the same names so you can drop-in replace the original calls.

from __future__ import annotations

import numpy as np

from _new25.dbg import dbg
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from atom.wf.lcr.func.FuncLcrToCr import FuncLcrToCr
from atom.wf.lcr.func.FuncLcrToSqrtCr import FuncLcrToSqrtCr
from atom.wf.lcr.func.FuncRToLcr import FuncRToLcr
from qm_math.func.arr.FuncArr import FuncArr
from qm_math.vec.grid.StepGridOpt import StepGridOpt

class LcrFactory:
    """
    Utility functions for converting between logarithmic‐coordinate
    (x = ln(c+r)) arrays and regular r-space arrays.

    The method names are identical to the original Java code
    (`wfLcrToR`, `densLcrToR`, `makeLcrFromR`) so existing calls
    continue to work without edits.
    """

    # ---------------------------------------------------------------
    @staticmethod
    def wfLcrToR(lcrArr: FuncArr, w: WFQuadrLcr) -> FuncArr:
        """
        Multiply the LCR wave-function by √(c+r) and
        switch its x-grid to the quadrature’s r-grid.
        """
        res = lcrArr.copyDeepY()  # deep copy of Y-values
        res.mult(FuncLcrToSqrtCr())  # *sqrt(c+r)
        res.setX(w.getR())  # replace x-grid with r-grid
        return res

    # ---------------------------------------------------------------
    @staticmethod
    def densLcrToR(lcrArr: FuncArr, w: WFQuadrLcr) -> FuncArr:
        """
        Multiply LCR density by (c+r) and map to the r-grid.
        """
        res = lcrArr.copyDeepY()
        res.mult(FuncLcrToCr())  # *(c+r)
        res.setX(w.getR())
        return res

    @staticmethod
    def makeLcrFromR(firstX: float, nLcr: int,
                     fromR: StepGridOpt) -> StepGridOpt:
        r_to_x = FuncRToLcr(firstX, fromR.getFirst())
        first = r_to_x.calc(fromR.getFirst())
        last = r_to_x.calc(fromR.getLast())
        return StepGridOpt(first, last, nLcr)

    @staticmethod
    def make_r1_min(x1_grid, *, use_c):
        # x = log(c + r)
        # exp(x) = c + r
        # r_min = exp(x_min) - c
        dbg('use_c')
        x1_min = x1_grid[0]
        dbg('x1_min')
        r_min = np.exp(x1_min) - use_c
        dbg('r_min')
        return r_min
