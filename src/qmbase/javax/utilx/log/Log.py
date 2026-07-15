# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : Log.py Created : 2025-06-26 at 10:00 am by Dmitry.A.Konovalov@gmail.com
# !/usr/bin/env python3

# log.py
from __future__ import annotations
from javax.utilx.log.kiss.KissLog import KissLog
import sys
from typing import Dict, Type
# from kiss_log import KissLog  # the decoupled, formatter-plugin version

# ---------------------------------------------------------------------------
#  Import the generic KissLog.  Provide a minimal stub if it is unavailable
#  so this file can run in isolation.
# ---------------------------------------------------------------------------

class Log(KissLog):
    """
    Python re-implementation of `javax.utilx.log.Log`, preserving every
    public method name from the original Java class.
    """

    # _globalLog: "Log" | None = None
    _map: Dict[str, "Log"] = {}

    # ---------- factory ----------------------------------------------------
    # @classmethod
    # @classmethod
    # def getLog(cls, c) -> "Log":
    #     # accept either a class/​type or a plain string
    #     name = c if isinstance(c, str) else c.__name__
    #     if name not in cls._map:
    #         cls._map[name] = Log(c if not isinstance(c, str) else Log)
    #     return cls._map[name]
    # ---------- factory ----------------------------------------------------
    @classmethod
    def print_dbg_logs(cls):
        print("Active loggers:")
        for name, logger in cls._map.items():
            dbg_on = logger.getDbg()
            if dbg_on:
                print(f"- {name}: dbg {dbg_on}")
        # if not any(getattr(logger, "_enabled_levels", None) for logger in cls._map.values()):
        #     print("  (none)")

    @classmethod
    def getLog(cls, key) -> "Log":
        """
        Return (and cache) a logger.

        Parameters
        ----------
        key : type | str
            • pass a *class*   → logger will prefix messages with that class name
            • pass a *string*  → logger will prefix messages with that string
        """
        # ‣ Determine the “name” to cache under
        if isinstance(key, str):
            name = key
            # fabricate a one-off dummy class so KissLog.format() can still use
            # logClass.__name__  →  "DerivPts3"
            dummy_cls = type(name, (), {})
            cls_obj: Type = dummy_cls
        elif isinstance(key, type):
            name = key.__name__
            cls_obj = key
        else:
            raise TypeError("getLog() expects a class object or a string")

        # ‣ Return cached logger or create a new one
        if name not in cls._map:
            cls._map[name] = Log(cls_obj)
        return cls._map[name]

    # ---------- constructor ------------------------------------------------
    def __init__(self, cl: Type[object]) -> None:
        super().__init__(cl)

    # ---------- root logger (singleton) ------------------------------------
    # @classmethod
    # def getRootLog(cls) -> "Log":
    #     if cls._globalLog is None:
    #         cls._globalLog = Log(Log)  # root identifies itself with Log class
    #     return cls._globalLog

    # ---------- convenience helpers ----------------------------------------
    # @classmethod
    # def setup(cls) -> None:
    #     root = cls.getRootLog()
    #     root.add(sys.stdout)
    #     root.add(sys.stderr)

    # @classmethod
    # def addGlobal(cls, ps) -> None:
    #     cls.getRootLog().add(ps)

    # ---------- override: setDbg -------------------------------------------
    def setDbg(self, dbg: bool = True) -> None:  # keep exact name
        # if dbg and Log.getRootLog().countPrintStreams() == 0:
        #     Log.getRootLog().add(sys.stdout)
        super().setDbg(dbg)

    # ---------- utility ----------------------------------------------------
    # @staticmethod
    # def printOpt(res: str, optName: str, optOn: bool, opt: str) -> str:
    #     if not optOn:
    #         return res
    #     if res:
    #         res += ", "
    #     res += f"{optName}={opt}"
    #     return res

# 1. Initialise global streams
# Log.setup()  # by default

# ---------------------------------------------------------------------------
#  Quick-check harness   →   $ python log.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialise global streams
    # Log.setup()

    # 2. Obtain a logger bound to an arbitrary class (here, Log itself)
    demo = Log.getLog(Log)

    # 3. Enable debug output
    demo.setDbg(False)

    # 4. Emit some messages
    demo.error("error")
    demo.info("info")
    demo.dbg("Debug message from demo logger")

    # 5. Demonstrate printOpt utility
    # opt_str = Log.printOpt("", "verbose", True, "on")
    # opt_str = Log.printOpt(opt_str, "threads", True, "4")
    # demo.info("Options:", opt_str)

