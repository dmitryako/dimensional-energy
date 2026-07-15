# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : LgrrOpt.py Created : 2025-06-30 at 11:51 am by Dmitry.A.Konovalov@gmail.com
# scatt/jm_2008/jm/laguerre/LgrrOpt.py

class LgrrOpt:
    """
    Python translation of Java LgrrOpt.
    """

    def __init__(self, L: int = 0, N: int = 0, lambda_: float = 0.0, m_2d=0) -> None:
        self.lambda_: float = lambda_
        self.N: int = N
        self.L: int = L
        # todo: NOTE  m_2d=0 is the same as L=-1/2
        self._m_2d: int = m_2d  # NOT USED!

    @classmethod
    def loadDefault(cls, model: "LgrrOpt") -> None:
        model.setLambda(1.0)
        model.setL(0)
        model.setN(10)

    def loadDefault(self) -> None:
        self.loadDefault(self)

    def getLambda(self) -> float:
        return self.lambda_

    def setLambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def getN(self) -> int:
        return self.N

    def setN(self, n: int) -> None:
        self.N = n

    def getL(self) -> int:
        return self.L

    def get_m_2d(self) -> int:
        return self._m_2d

    def setL(self, l: int) -> None:
        self.L = l

    def toString(self) -> str:
        return f"LgrrOpt(L={self.L}, n={self.N}, lambda={self.lambda_})"

    def makeLabel(self) -> str:
        return f"L{self.L}_LMBD{float(self.lambda_)}_N{self.N}"

    # Python dunders
    __str__  = toString
    __repr__ = toString

