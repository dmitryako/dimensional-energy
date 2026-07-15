# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Step.py Created : 2025-06-26 at 3:52 pm by Dmitry.A.Konovalov@gmail.com
from qm_math.vec.grid.Range import Range


# step.py  –– Port of math.vec.grid.Step
# (Java original: 09 Jul 2008, 14 : 54 : 18)
#
# • Holds a single value “step”.
# • Java had two constructors:
#       Step(Range r, int size)  – sets step = r.getRange() / (size-1)
#       Step(double step)        – sets step directly
#   Python combines these in __init__(src, size=None).
# • Method names match the Java source exactly.
# -------------------------------------------------------------------------

# from range import Range


class Step:
    def __init__(self, src, size=None):
        """
        Parameters
        ----------
        src      : Range | float
            Either a Range instance (Java’s first constructor) **or**
            a numeric step value (Java’s second constructor).
        size     : int | None
            Required only when src is a Range, representing the number
            of grid points (must be > 1).
        """
        if isinstance(src, Range):
            if size is None or size < 2:
                raise ValueError("size must be >= 2 when src is a Range")
            self._step: float = src.getRange() / (size - 1)
        else:                               # assume numeric value
            self._step = float(src)

    # ---------- getters / setters (Java names) ---------------------------
    def getStep(self) -> float:
        return self._step

    def setStep(self, step: float) -> None:
        self._step = float(step)


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # from range import Range

    r = Range(0.0, 10.0)        # length = 10
    s1 = Step(r, size=6)        # step = 10 / (6-1) = 2
    print("step from Range =", s1.getStep())

    s2 = Step(0.25)             # direct value
    print("direct step     =", s2.getStep())

    s2.setStep(0.5)
    print("after setStep   =", s2.getStep())
