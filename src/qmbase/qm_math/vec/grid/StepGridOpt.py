# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : StepGridOpt.py Created : 2025-06-26 at 12:46 pm by Dmitry.A.Konovalov@gmail.com
# step_grid_opt.py  –– Port of *updated* math.vec.grid.StepGridOpt
# (Java source: 12 Sep 2008, 15 : 05 : 14)
#
# • Java now has only ONE constructor  StepGridOpt(first, last, n)
#   – all fields still default to 0 before that call is made.
# • Field names get a leading “_”.  Getter / setter names stay IDENTICAL.
# -------------------------------------------------------------------------
from javax.utilx.log.Log import Log
from javax.utilx.log.kiss.KissLog import KissLog


class StepGridOpt:
    def __init__(self, first: float, last: float, n: int) -> None:
        # mimic Java’s explicit initial field values (all zeros) first
        self._first: float = 0.0
        self._last:  float = 0.0
        self._numPoints: int = 0

        # then apply constructor arguments
        self._first = first
        self._last = last
        self._numPoints = n

    # ---------- getters / setters (exact Java names) ----------------------
    def getFirst(self) -> float:
        return self._first

    def setFirst(self, first: float) -> None:
        self._first = first

    def getLast(self) -> float:
        return self._last

    def setLast(self, last: float) -> None:
        self._last = last

    def getNumPoints(self) -> int:
        return self._numPoints
    def size(self) -> int:
        return self._numPoints

    def setNumPoints(self, numPoints: int) -> None:
        self._numPoints = numPoints

    # ---------- toString identical to Java --------------------------------
    def toString(self) -> str:
        return (f"(first={float(self._first)}, "
                f"last={float(self._last)}, "
                f"nPts={self._numPoints})")

    # Python dunders
    __str__  = toString
    __repr__ = toString


# register pretty-printer with KissLog
KissLog.register_formatter(StepGridOpt, lambda e: str(e))


# -------------------- quick demo / self-test -------------------------
if __name__ == "__main__":
    from project.workflow.task.test.FlowTest import FlowTest

    s1 = StepGridOpt(0., 100., 101)
    log = Log.getLog(StepGridOpt)
    log.info("StepGridOpt =", s1)
