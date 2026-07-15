from __future__ import annotations
__pytest__ = False  # pytest must ignore this module

# flow_test.py  ––  1-for-1 port of project.workflow.task.test.FlowTest
# ---------------------------------------------------------------------

import inspect
import math
import sys
from typing import Callable, Optional, Type

# ---------------------------------------------------------------------
# 1)  Try the real logger; fall back to a tiny stub if not present
# ---------------------------------------------------------------------
from javax.utilx.log.Log import Log  # production logger

# moved all out of FlowTest on 250726: too many bugs in setMaxErr
MAX_WF_DIFF_ERR = 1e-9
MAX_INTGRL_ERR_E10 = 1e-10
MAX_INTGRL_ERR_E11 = 1e-11
MAX_INTGRL_ERR_E12 = 1e-12
MAX_INTGRL_ERR_E13 = 1e-13


class cfg_test(dict):
    # dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _maxErr = 1e-8
    _lockMaxErr = False
    exit_on_err = True


# ---------------------------------------------------------------------
# 2)  Java’s AssertionFailedError replacement
# ---------------------------------------------------------------------
class AssertionFailedError(AssertionError):
    """Mimics junit.framework.AssertionFailedError."""
    def __init__(self, *args, **kwargs): # real signature unknown
        print(f"CRITICAL FAILURE: {args[0]}")
        if cfg_test.exit_on_err:
            sys.exit(1)  # Exit with error code
    # pass

log = Log.getLog("FlowTest")

# ---------------------------------------------------------------------
# 3)  Pure-Python FlowTest with *all* original API methods retained
# ---------------------------------------------------------------------
class FlowTest:  # NOT a unittest.TestCase
    # -------- static (class) fields ----------------------------------
    log = Log.getLog("FlowTest")
    # MAX_WF_DIFF_ERR = 1e-9
    # MAX_INTGRL_ERR_E10 = 1e-10
    # MAX_INTGRL_ERR_E11 = 1e-11
    # MAX_INTGRL_ERR_E12 = 1e-12
    # MAX_INTGRL_ERR_E13 = 1e-13
    # _maxErr: float = 1e-8
    # _lockMaxErr: bool = False

    # -------- instance field -----------------------------------------
    def __init__(self, theClass: Optional[Type] = None) -> None:
        self.theClass = theClass

    # -----------------------------------------------------------------
    #  Java setUp()
    # -----------------------------------------------------------------
    def setUp(self) -> None:
        try:
            FlowTest.log.addGlobal(sys.__stdout__)
        except AttributeError:
            pass  # stub logger has no addGlobal

    # ---------------- helpers / utils --------------------------------
    @staticmethod
    def format(help_msg: Optional[str], expected: float, actual: float) -> str:
        return f"{help_msg or ''} expected={expected} actual={actual}"

    @staticmethod
    def failNotEquals(msg: str, expected: float, actual: float) -> None:
        raise AssertionFailedError(f"{msg} => expected {expected} but was {actual}")

    # ---------------- maxErr handling --------------------------------
    @classmethod
    def getMaxErr(cls) -> float:
        return cfg_test._maxErr

    @classmethod
    def setMaxErr(cls, d: float) -> None:
        if cfg_test._lockMaxErr:
            # raise RuntimeError("lockMaxErr==true")
            raise AssertionFailedError("lockMaxErr==true")
        cfg_test._maxErr = d
        assert not hasattr(cls, "_maxErr")  # stop accidentals

    @classmethod
    def setLog(cls, dbgLog) -> None:
        cls.log = dbgLog

    # ---------------- assertEquals overload family -------------------
    # def assertEquals(self, *args):
    #     return FlowTest.assertEquals(*args)

    #     @staticmethod
    #     def assertEquals(*args):
    @classmethod
    def assertEquals(cls, *args):
        """
        assertEquals([help], expected, actual [, showDbg2])
        """
        if len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], str):
                raise TypeError(f"assertEquals signature ({args[0]}, {args[1]}): call normal assert()")
            else:
                help_msg, expected, actual, show_dbg2 = None, args[0], args[1], False

        elif len(args) == 3:
            if isinstance(args[0], str):
                help_msg, expected, actual, show_dbg2 = args[0], args[1], args[2], False
            else:
                # orig junit: assertEquals(0, df.get(idx), Calc.EPS_18);
                # static public void assertEquals(double expected, double actual, double delta) {
                help_msg, expected, actual, show_dbg2 = None, args[0], args[1], False
                # cls.setMaxErr(args[2])
                cls._assertEquals_impl(
                    help_msg=help_msg, expected=expected, actual=actual, maxErr=args[2], show_dbg2=show_dbg2)
                return   # todo NOTE <------ !

        elif len(args) == 4:
            help_msg, expected, actual, show_dbg2 = args
        else:
            raise TypeError("assertEquals signature mismatch")
        # msg = cls.format(help_msg, expected, actual)
        # msg = msg + f", err={abs(expected - actual)}"
        # msg = msg + f", abs_tol={cfg_test._maxErr}"
        # ok = abs(expected - actual) <= abs(cfg_test._maxErr)
        # if not ok:
        #     cls.log.info(msg)
        #     raise AssertionFailedError(msg)
        # if show_dbg2:
        #     cls.log.dbg(msg)
        cls._assertEquals_impl(
            help_msg=help_msg, expected=expected, actual=actual, maxErr=cfg_test._maxErr, show_dbg2=show_dbg2)

    @classmethod
    def _assertEquals_impl(cls, *, help_msg, expected, actual, maxErr, show_dbg2):
        # todo?? what a mess!!!
        msg = cls.format(help_msg, expected, actual)
        msg = msg + f", err={abs(expected - actual)}"
        # msg = msg + f", abs_tol={cfg_test._maxErr}"
        msg = msg + f", abs_tol={maxErr}"
        # ok = abs(expected - actual) <= abs(cfg_test._maxErr)
        ok = abs(expected - actual) <= abs(maxErr)
        if not ok:
            cls.log.info(msg)
            raise AssertionFailedError(msg)
        if show_dbg2:
            cls.log.dbg(msg)

    @classmethod
    def assertEqualsRel(cls, help_msg: str, expected: float,
                        actual: float, showDbg2: bool = False) -> None:
        msg = cls.format(help_msg, expected, actual)
        msg += f" relErr={(expected - actual) / expected}"
        msg += f" maxRelErr={cfg_test._maxErr}"
        tmpMaxErr = abs(cfg_test._maxErr * expected)
        ok = abs(expected - actual) <= abs(tmpMaxErr)
        if not ok:
            cls.log.info(msg)
            raise AssertionFailedError(msg)
        if showDbg2:
            cls.log.dbg(msg)


    @classmethod
    def assertFloorRel(cls, help_msg: str, expected: float,
                       actual: float, maxRelErr: float) -> None:
        msg = cls.format(help_msg, expected, actual)
        msg += f" relErr={(expected - actual) / expected}"
        msg += f" maxRelErr={maxRelErr}"
        cls.assertFloor(msg + "; assertFloor", expected, actual,
                        abs(maxRelErr * expected))

    @classmethod
    def assertCeilRel(cls, help_msg: str, expected: float,
                      actual: float, maxRelErr: float) -> None:
        msg = cls.format(help_msg, expected, actual)
        msg += f" relErr={(expected - actual) / expected}"
        msg += f" maxRelErr={maxRelErr}"
        cls.assertCeil(msg + "; assertCeiling", expected, actual,
                       abs(maxRelErr * expected))

    @classmethod
    def assertFloor(cls, mssg: str, floor: float,
                    actual: float, delta: float) -> None:
        # if math.isclose(floor, actual):
        #     return
        if floor > actual or abs(floor - actual) > delta:
            cls.failNotEquals(mssg, floor, actual)

    @classmethod
    def assertCeil(cls, mssg: str, ceil: float,
                   actual: float, delta: float) -> None:
        # if math.isclose(ceil, actual):
        #     return
        if actual - ceil > delta:
            cls.failNotEquals(mssg, ceil, actual)

    # ---------------- lock mechanics ----------------------------------
    @classmethod
    def _setLockMaxErr(cls, value: bool) -> None:
        cfg_test._lockMaxErr = value
        assert not hasattr(cls, "_lockMaxErr")  # stop accidentals

    @classmethod
    def lockMaxErr(cls, err: float) -> None:
        cls.setMaxErr(err)
        cls._setLockMaxErr(True)

    @classmethod
    def unlockMaxErr(cls) -> None:
        cls._setLockMaxErr(False)

    # ---------------- JUnit-style runner ------------------------------
    @staticmethod
    def _run_suite(test_cls: Type) -> bool:
        """
        Executes every method starting with 'test' on *a new instance* of test_cls.
        """
        instance = test_cls()
        methods = [name for name, _ in inspect.getmembers(instance, inspect.ismethod)
                   if name.startswith("test")]
        if not methods:
            print(f"[SKIP] {test_cls.__name__}: no test* methods found.")
            return True

        print(f"\n=== Running {test_cls.__name__} ===", file=sys.__stdout__)
        failures = 0
        for name in methods:
            print(f"→ {name} ... ", end="", file=sys.__stdout__)
            try:
                getattr(instance, name)()
            except Exception as exc:  # catch all, mimic JUnit
                failures += 1
                print("FAIL", file=sys.__stdout__)
                FlowTest.log.info(f"{name} raised {exc}")
            else:
                print("ok", file=sys.__stdout__)

        if failures:
            raise AssertionFailedError(f"{failures} test(s) failed.")
        return True

    # -------------- public Ok API variants ----------------------------
    @staticmethod
    def ok(test: "Type | Callable[[], object]") -> bool:
        """
        Static convenience: FlowTest.ok(SomeTestClass) or
                            FlowTest.ok(lambda: SomeTestClass(cfg))
        """
        if isinstance(test, type):
            return FlowTest._run_suite(test)
        elif callable(test):
            return FlowTest._run_suite(test().__class__)
        else:
            raise TypeError("ok(...) expects a class or zero-arg factory")

    # original instance method
    def ok_instance(self) -> bool:
        if self.theClass is None:
            raise ValueError("theClass must be provided to run ok().")
        return FlowTest._run_suite(self.theClass)
