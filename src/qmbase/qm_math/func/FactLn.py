# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : FactLn.py Created : 2025-06-26 at 1:33 pm by Dmitry.A.Konovalov@gmail.com


# fact_ln.py  ––  Port of math.func.FactLn
# (Java original: 11 Jul 2008, 10 : 13 : 41)
#
# • Replicates the original **singleton** pattern:
#       FactLn.getInstance()     – returns a shared calculator
#       FactLn.makeInstance(n)   – bootstrap with custom table size
# • Method names match the Java source exactly (calc, calcFactDiv, …).
# • Uses Python lists for the log-factorial table.
# -------------------------------------------------------------------------

import math
from typing import List, Optional


class FactLn:
    # -------- class-level (static) data ----------------------------------
    _logs: List[float] = []          # stores ln(n!) with index n
    _STEP_FORWARD: int = 100         # grow table in chunks
    _instance: Optional["FactLn"] = None

    # -------- singleton accessors ---------------------------------------
    @classmethod
    def getInstance(cls) -> "FactLn":
        """
        Java:  public static FactLn getInstance()
        Default table size = 100 entries (0 … 100).
        """
        if cls._instance is None:
            cls._instance = cls(100)
        return cls._instance

    @classmethod
    def makeInstance(cls, size: int) -> "FactLn":
        """
        Java:  public static FactLn makeInstance(int size)
        Creates the singleton with an initial table up to *size*.
        """
        if cls._instance is None:
            cls._instance = cls(size)
        return cls._instance

    # -------- constructor (private in Java) ------------------------------
    def __init__(self, last: int) -> None:
        # populate log-factorial table up to ‘last’
        self.loadLogs(last)

    # -------- public API --------------------------------------------------
    def calc(self, n: int) -> float:
        """
        Java:  public double calc(int n)
        Returns ln(n!).  Extends table automatically if needed.
        """
        if n >= len(FactLn._logs):
            self.loadLogs(n + FactLn._STEP_FORWARD)
        return FactLn._logs[n]

    def calcFactDiv(self, top: int, bot: int) -> float:
        """
        Java:  public double calcFactDiv(int top, int bot)
        Computes   exp( ln(top!) – ln(bot!) )  =  top! / bot!
        """
        return math.exp(self.calc(top) - self.calc(bot))

    # -------- helper to extend the table ---------------------------------
    def loadLogs(self, last: int) -> None:
        """
        Java:  protected void loadLogs(int last)
        Ensures _logs contains ln(k!) for all k ≤ last.
        """
        if not FactLn._logs:                     # initialise with 0! = 1
            FactLn._logs.append(0.0)             # ln(1) = 0
            i = 1
            logF = 0.0
        else:
            i = len(FactLn._logs)
            logF = FactLn._logs[-1]

        for k in range(i, last + 1):
            logF += math.log(k)
            FactLn._logs.append(logF)


if __name__ == "__main__":
    fl = FactLn.getInstance()
    print("ln(5!) =", fl.calc(5))               # ≈ ln(120) = 4.787…
    print("10! / 8! =", fl.calcFactDiv(10, 8))  # should be 90

