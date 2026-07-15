# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Func.py Created : 2025-06-26 at 1:54 pm by Dmitry.A.Konovalov@gmail.com

class Func:
    def calc(self, x: float) -> float:          # Java: double calc(double x)
        raise NotImplementedError("Sub-class must implement calc(x)")
    def __call__(self, x: float) -> float:
        return self.calc(x)

