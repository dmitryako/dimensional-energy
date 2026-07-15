# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : DbgView.py Created : 2025-06-26 at 1:11 pm by Dmitry.A.Konovalov@gmail.com


# dbg_view.py  ––  Port of math.vec.DbgView
# (Java original: 05 Mar 2010, 11 : 57 : 52)
#
# • Static fields become class attributes with “_” prefix.
# • Java had three overloaded  append(...)  methods; Python cannot overload,
#   so we provide **one** append(buff, value) that inspects value’s type.
#   Existing call-sites such as  DbgView.append(buff, 3.14)  still work. ✔
# • `buff` is expected to be a list of strings.  After all appends you can
#   `"".join(buff)` to obtain the final string (equivalent to StringBuffer).
# • Complex numbers: accepts either Python built-in `complex` **or**
#   any object exposing  getRe() / getIm() / abs()  like the old Cmplx.
# -------------------------------------------------------------------------


class DbgView:
    show_digs: int = 5
    _numShow: int = 10
    _minVal: float = 0.0      # always stored as absolute value

    # ---------- configuration -------------------------------------------
    @staticmethod
    def getNumShow() -> int:
        return DbgView._numShow

    @staticmethod
    def setNumShow(numShow: int) -> None:
        DbgView._numShow = numShow

    @staticmethod
    def getMinVal() -> float:
        return DbgView._minVal

    @staticmethod
    def setMinVal(minVal: float) -> None:
        DbgView._minVal = abs(minVal)

    # ---------- append utility (unified) ---------------------------------
    @staticmethod
    def append(buff: list[str], value) -> None:
        """
        Java overloads:
            append(buff, double)
            append(buff, String)
            append(buff, Cmplx)
        Python version inspects `value` and formats accordingly.
        """
        # -- String -------------------------------------------------------
        if isinstance(value, str):
            buff.append(value)
            return

        # -- Real number --------------------------------------------------
        if isinstance(value, (int, float)):
            if DbgView._minVal == 0.0 or abs(value) >= DbgView._minVal:
                buff.append(str(float(value)))       # mimic Float.toString
            else:
                buff.append("0.0")
            return

        # -- Complex number (Python built-in) -----------------------------
        if isinstance(value, complex):
            mag = abs(value)
            if DbgView._minVal == 0.0 or mag >= DbgView._minVal:
                buff.append(f"({float(value.real)},{float(value.imag)})")
            else:
                buff.append("(0)")
            return

        # -- Cmplx-like object (getRe/getIm) ------------------------------
        if hasattr(value, "getRe") and hasattr(value, "getIm") and hasattr(value, "abs"):
            mag = value.abs()
            if DbgView._minVal == 0.0 or mag >= DbgView._minVal:
                buff.append(f"({float(value.getRe())},{float(value.getIm())})")
            else:
                buff.append("(0)")
            return

        # -- Fallback: just str() ----------------------------------------
        buff.append(str(value))

    @classmethod
    def toStr(cls, x):
        if isinstance(x, complex):
            return str(x)
        if isinstance(x, float):
            return f"{x:.{cls.show_digs}f}"
        return str(x)



if __name__ == "__main__":
    out = []
    DbgView.append(out, 1.2345)
    DbgView.append(out, ", ")
    DbgView.append(out, complex(0.1, -0.2))
    print("".join(out))
    # → 1.2345, (0.1,-0.2)

