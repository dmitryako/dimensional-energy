from __future__ import annotations
# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : AtomUtil.py Created : 2025-06-29 at 2:01 pm by Dmitry.A.Konovalov@gmail.com

__pytest__ = False   # pytest should ignore this helper module

# atom/AtomUtil.py  ––  exact port of atom.AtomUtil
# -------------------------------------------------

# from math import isclose
from typing import Any

# NOTE: adjust the imports to your converted modules
from qm_math.Calc import Calc                       # same base-class as Java
from qm_math.vec.Vec import Vec
from qm_math.func.arr.FuncArr import FuncArr


class AtomUtil(Calc):                              # public class AtomUtil extends Calc
    """Utility methods originally in atom.AtomUtil (Java)."""

    # ------------------------------------------------------------
    # static void trimTailSLOW(Vec f)
    # ------------------------------------------------------------
    @staticmethod
    def trimTailSLOW(f: Any) -> None:
        """
        Remove trailing zeros from a Vec **in-place** – Java’s
        ``trimTailSLOW(Vec f)``.  If the argument is a FuncArr,
        the overload below is dispatched instead.
        """
        if isinstance(f, FuncArr):                 # delegate to the array version
            AtomUtil._trimTail_funcarr(f)
            return
        if isinstance(f, Vec):                 # delegate to the vec version
            AtomUtil._trimTail_vec(f)
            return
        if not isinstance(f, Vec):
            raise TypeError("trimTailSLOW expects Vec or FuncArr")


    @staticmethod
    def _trimTail_vec(f: Vec) -> None:
        new_size = f.size()                        # int newSize = f.size();
        for i in range(f.size() - 1, -1, -1):      # for (int i = f.size()-1; i>=0; i--)
            if Calc.isZero(f.get(i)):              #     if (Calc.isZero(f.get(i))) {
                new_size -= 1                      #         newSize--;
                f.set(i, 0.0)                      #         f.set(i,0);
            else:                                  #     } else
                break                              #         break;
        f.setSize(new_size)                        # f.setSize(newSize);

    @staticmethod
    def _trimTail_funcarr(arr: FuncArr) -> None:
        # for (int i = 0; i < arr.size(); i++) {
        for i in range(arr.size()):
            # AtomUtil.trimTailSLOW(arr.get(i))      #  trimTailSLOW(arr.get(i));
            AtomUtil._trimTail_vec(arr.get(i).getY())  # force vec

# //  public static void setTailFrom(FuncArr arr, FuncArr fromArr) {
# //    for (int i = 0; i < arr.size(); i++) {
# //      arr.get(i).setSize(fromArr.get(i).size());
# //    }
# //  }


if __name__ == "__main__":
    # from atom.AtomUtil import AtomUtil
    from qm_math.vec.Vec import Vec

    v = Vec([1.0, 0.0, 0.0, 0.0])
    print("before:", v.getArr())
    AtomUtil.trimTailSLOW(v)
    print("after :", v.getArr(), "size =", v.size())

