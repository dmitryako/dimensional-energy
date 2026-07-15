# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : KissLog.py   (core logger, Python 3.9+)

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, Set, TextIO, Type

# ------------------------------------------------------------------------
# Real console streams – never redirected by unittest / IDEs
# ------------------------------------------------------------------------
real_out: TextIO = sys.__stdout__
real_err: TextIO = sys.__stderr__


class KissLog:
    _static_pr_streams: Set[TextIO] = set()
    _formatters: Dict[Type[Any], Any] = {}

    # ---------------- plug-in pretty-printer hook ------------------------
    @classmethod
    def register_formatter(cls, typ: Type[Any], fn) -> None:
        cls._formatters[typ] = fn

    # ---------------- construction & cfg ---------------------------------
    def __init__(self, cl: Type[Any]) -> None:
        self.logClass = cl
        self._local_pr_streams: Set[TextIO] = set()
        self._info = True
        self._dbg = False
        self._newline = True

    # ---------------- internal helpers -----------------------------------
    def _render(self, obj: Any) -> str:
        for typ, fn in self._formatters.items():
            if isinstance(obj, typ):
                return fn(obj)
        return str(obj)

    def format(self, msg: str) -> str:
        return f"{self.logClass.__name__}: {msg}"

    # core fan-out:   real_out/err  →  global streams  →  local streams
    def _fanout(self, msg: str, *, newline: bool, is_err: bool = False) -> None:
        suffix = "\n" if newline else ""
        stream: TextIO = real_err if is_err else real_out

        stream.write(msg + suffix)       # always visible
        stream.flush()

        for ps in (*self._static_pr_streams, *self._local_pr_streams):
            ps.write(msg + suffix)
            ps.flush()

    # ---------------- static/global API ----------------------------------
    @classmethod
    def add(cls, ps: TextIO) -> None:
        cls._static_pr_streams.add(ps)

    @classmethod
    def countPrintStreams(cls) -> int:
        return len(cls._static_pr_streams)

    # ---------------- instance/local streams -----------------------------
    def addTextView(self, view: TextIO) -> None:
        self._local_pr_streams.add(view)

    # ---------------- error ----------------------------------------------
    def error(self, s: str, *args: Any) -> str:   # same Java API
        if args:
            s += " " + " ".join(self._render(a) for a in args)
        return self.error2(s)

    # alias kept from Java (“err”→error)
    err = error

    def error2(self, s: str) -> str:
        s = self.format(s)
        self._fanout(s, newline=True, is_err=True)
        return s

    # ---------------- info -----------------------------------------------
    def info(self, s: str, *args: Any) -> "KissLog":
        if not self._info:
            return self
        if args:
            s += " " + " ".join(self._render(a) for a in args)
        self.info2(s)
        return self

    def info2(self, s: str) -> None:
        if not self._info:
            return
        s = self.format(s)
        self._fanout(s, newline=True)

    # ---------------- dbg ------------------------------------------------
    def dbg(self, s: str, *args: Any) -> "KissLog":
        if not self._dbg:
            return self
        if args:
            s += " " + " ".join(self._render(a) for a in args)
        self.debug2(s)
        return self

    def debug2(self, s: str) -> None:
        if not self._dbg:
            return
        s = self.format(s)
        self._fanout(s, newline=self._newline)

    # ---------------- debug switches ------------------------------------
    def getDbg(self) -> bool:
        return self._dbg

    def setDbg(self, dbg: bool = True) -> None:
        self._dbg = dbg

    def setDebugOff(self) -> None:
        self._dbg = False

    # ---------------- newline control -----------------------------------
    def setNewLine(self, newLine: bool) -> "KissLog":
        self._newline = newLine
        return self

    def eol(self) -> "KissLog":   # println mode
        return self.setNewLine(True)

    def inl(self) -> "KissLog":   # inline   mode
        return self.setNewLine(False)

    # ---------------- misc utilities ------------------------------------
    def assertZero(self, text: str, val: float, error: float) -> None:
        f = abs(val)
        self.dbg(f"{text}{f}")
        assert f <= error, f"{text}: |{val}| > {error}"

    def saveToFile(self, text: str, dirName: str,
                   dir2: str, fileName: str) -> None:
        path = Path(dirName) / dir2 / fileName
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------------
# Default: also mirror to (possibly captured) sys.stdout
# ------------------------------------------------------------------------
# KissLog.add(sys.stdout)  # todo: <-- example how to add more. e.g. to a file


# ---------------- quick demo --------------------------------------------
if __name__ == "__main__":
    log = KissLog(KissLog)

    log.info("Hello world!", 123, True)
    log.setDbg(False)

    from math import pi
    log.dbg("Pi =", pi)

    # show output on console
    # KissLog.add(sys.stdout)

    # prev version
    log = KissLog(KissLog)
    # from javax.utilx.log.kiss import KissLog
    print(hasattr(KissLog, "register_formatter"))

    # basic info
    log.info("Hello world!", 123, True, 9.81)

    # enable debugging and show various overloads
    log.setDbg(False)
    from qm_math.vec.Vec import Vec
    KissLog.register_formatter(Vec, lambda v: str(v))
    log.dbg("Vector:", Vec(10))
    # from qm_math.vec.VecDbgView import VecDbgView
    # KissLog.register_formatter(Vec, lambda v: str(VecDbgView(v)))
    KissLog.register_formatter(Vec, lambda v: str(v))

    from qm_math.vec.VecArr import VecArr
    log.dbg("Vector array:", VecArr())
    log.dbg("Float:", 3.14159).inl().dbg(" – inline continuation")
    log.eol()  # restore newline mode

    # write a file
    log.saveToFile("Saved via KissLog\n",
                   "/tmp", "kisslog_demo", "test.txt")
    log.info("File written to /tmp/kisslog_demo/test.txt")
    log.saveToFile("Saved via KissLog\n",
                   "", "", "test.txt")
    log.info("File written to test.txt")
