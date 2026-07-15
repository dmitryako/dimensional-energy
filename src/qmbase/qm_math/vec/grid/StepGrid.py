# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : StepGrid.py Created : 2025-06-26 at 12:42 pm by Dmitry.A.Konovalov@gmail.com

# step_grid.py  ––  Port of  math.vec.grid.StepGrid   (9 Jul 2008, 14:52:33)
#
# • Keeps Java method names (getStep, loadGrid, etc.).
# • Overloaded constructors are provided as **classmethods** with the suffixes
#   you specified:
#       ─ calc(Func f)               →  calc_func          (already in Vec)
#       ─ calc(Vec x, Func f)        →  calc_vec_func      (already in Vec)
#       ─ new StepGrid(StepGridOpt)  →  fromStepGridOpt
#       ─ new StepGrid(first,numSteps,stepVal)
#                                    →  from_first_numSteps_stepVal
# • Uses NumPy for numeric work; no FastLoop needed.
# • Includes tiny placeholder implementations of Step, StepGridOpt,
#   VecDbgView so the class runs immediately.
# -------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Any

from javax.utilx.log.Log import Log
from javax.utilx.log.kiss.KissLog import KissLog
from qm_math.vec.Vec import Vec
# from qm_math.vec.VecDbgView import VecDbgView
# from qm_math.vec.VecDbgView import VecDbgView
from qm_math.vec.grid.Range import Range
from qm_math.vec.grid.Step import Step
from qm_math.vec.grid.StepGridOpt import StepGridOpt

log = Log.getLog("StepGrid")


class StepGrid(Vec):
    # --- primary constructor: StepGrid(first, last, size) -----------------
    # def __init__(self, *, first: float, last: float, size: int):
    def __init__(self, first: float, last: float, size: int):
        super().__init__(size)
        log.inl() \
            .dbg("StepGrid(first=", first) \
            .dbg(", last=", last).eol() \
            .dbg(", size=", size)

        assert isinstance(first, float)
        assert isinstance(last, float)
        assert isinstance(size, int)

        rng = Range(first, last)
        self._step = Step(rng, size)
        self._loadGrid(rng)

    # --- secondary constructor: StepGrid(StepGridOpt model) ---------------
    @classmethod
    def fromStepGridOpt(cls, model: StepGridOpt) -> "StepGrid":
        # pass keyword-args to satisfy the “*” in __init__
        return cls(
            first=model.getFirst(),
            last=model.getLast(),
            size=model.getNumPoints(),
        )
        # return cls(model.getFirst(), model.getLast(), model.getNumPoints())

    # --- third constructor: StepGrid(first, numSteps, stepVal) ------------
    # @classmethod
    @staticmethod
    def from_first_numSteps_stepVal(*,
                                    first: float,
                                    numSteps: int,
                                    stepVal: float) -> "StepGrid":
        res = StepGrid(first=first, last=first + numSteps * stepVal, size=numSteps+1)
        # super().__init__(numSteps + 1)
        # obj = cls.__new__(cls)  # bypass __init__
        # Vec.__init__(obj, numSteps + 1)  # allocate array
        StepGrid.log.inl() \
            .dbg("StepGrid(first=", first) \
            .dbg("numSteps=", numSteps).eol() \
            .dbg("stepVal=", stepVal)
        # obj._step = Step(stepVal)
        # res._step = Step(stepVal)
        # range = Range(first, first + numSteps * stepVal)
        # res._loadGrid(range)
        return res

    # ---------------------------------------------------------------------
    # Java getters / setters for Step
    def getStep(self) -> Step:
        return self._step

    def setStep(self, step: Step) -> None:
        self._step = step

    # ---------------------------------------------------------------------
    # Internals ------------------------------------------------------------
    def _loadGrid(self, rng: Range) -> None:
        h = self._step.getStep()
        for i in range(self.size()):
            self.set(i, rng.getLeft() + h * i)
        # StepGrid.log.dbg("StepGrid.loadGrid()=", VecDbgView(self))
        log.dbg("StepGrid.loadGrid()=", self)

    # Java: public double getGridStep()
    def getGridStep(self) -> float:
        return self._step.getStep()


# KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))

# -------------------- quick demo ------------------------------------------
if __name__ == "__main__":
    g1 = StepGrid(first=0.0, last=10.0, size=101)  # primary ctor
    print("g1:", g1.toCSV())
    log = Log.getLog(StepGrid)
    log.info("g1", g1)

    opt = StepGridOpt(1.0, 4.0, 4)
    g2 = StepGrid.fromStepGridOpt(opt)  # secondary
    print("g2:", g2.toCSV())

    g3 = StepGrid.from_first_numSteps_stepVal(first=2.0, numSteps=5, stepVal=0.5)  # third
    print("g3:", g3.toCSV(), "step=", g3.getGridStep())
