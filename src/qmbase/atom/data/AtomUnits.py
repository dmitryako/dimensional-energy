# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : AtomUnits.py Created : 2025-06-30 at 10:22 am by Dmitry.A.Konovalov@gmail.com
# ---------------------------------------------------------------------------
#  Python port of atom.data.AtomUnits
#  (keeps *all* original Java comments for traceability)
# ---------------------------------------------------------------------------

from __future__ import annotations
from typing import Callable

from qm_math.func.Func import Func
"""
NOTE on ENERGY_TO_EV:

- Standard infinite nuclear mass (CODATA): 27.211386245988 eV
- CCC uses: 27.2116 eV (likely includes empirical/historical adjustments)

The 27.2116 eV value used here may include effective adjustments beyond just the finite-mass correction for Hydrogen.

If precision modeling or comparison with CODATA is needed, consider switching to the infinite-mass value.

Reduced-mass correction factor (μ/m_e) for Hydrogen can also be applied manually if desired.
"""

"""
AtomUnits.py
Conversion constants between atomic units (Hartree) and electronvolts (eV).

Author: Dmitry A. Konovalov
Date: 2025-06-30

--------------------------------------------------------------
Usage Note on ENERGY_TO_EV:

By default, this module uses the **CCC / finite-mass Hydrogen value**:

    ENERGY_TO_EV = 27.2116 eV

This reflects a slight correction for the finite proton mass (commonly used
in experimental hydrogen scattering and threshold energy calculations).

If your calculations require **CODATA infinite proton mass value**, use:

    ENERGY_TO_EV = 27.21138386 eV

This is recommended for heavy elements (He, Li, Fe, etc.), or for theoretical
multi-electron atomic structure calculations.

You can control this at runtime using:

    AtomUnits.use_codata()
    AtomUnits.use_ccc()

--------------------------------------------------------------
"""

class AtomUnits:
    # Default: CCC Hydrogen (finite mass)
    #     # Accessed 25Mar2011, http://physics.nist.gov/cuu/Constants/index.html
    #     # P. J. Mohr, B. N. Taylor, and D. B. Newell,
    #     # Rev. Mod. Phys 80(2), 633-730 (2008)
    #     # public static final double ENERGY_TO_EV = 27.21138386;
    #     ENERGY_TO_EV: float = 27.2116          # CCC uses this value
    #     HARTREE_TO_EV: float = ENERGY_TO_EV
    #     ENERGY_FROM_EV: float = 1.0 / ENERGY_TO_EV
    # ENERGY_TO_EV: float = 27.2116
    ENERGY_TO_EV: float = 27.21138386
    HARTREE_TO_EV: float = ENERGY_TO_EV
    ENERGY_FROM_EV: float = 1.0 / ENERGY_TO_EV

    @classmethod
    def use_codata(cls) -> None:
        """
        Switch to CODATA 2018 infinite-mass proton Hartree-to-eV conversion.
        """
        cls.ENERGY_TO_EV = 27.21138386
        cls.HARTREE_TO_EV = cls.ENERGY_TO_EV
        cls.ENERGY_FROM_EV = 1.0 / cls.ENERGY_TO_EV

    @classmethod
    def use_ccc(cls) -> None:
        """
        Switch to CCC-style (finite-mass hydrogen) Hartree-to-eV conversion.
        """
        cls.ENERGY_TO_EV = 27.2116
        cls.HARTREE_TO_EV = cls.ENERGY_TO_EV
        cls.ENERGY_FROM_EV = 1.0 / cls.ENERGY_TO_EV

    @classmethod
    def from_ev(cls, ev: float) -> float:
        return ev * cls.ENERGY_FROM_EV

    @classmethod
    def to_ev(cls, au: float) -> float:
        return au * cls.ENERGY_TO_EV

    # Optional: Provide a functional interface for functors
    @classmethod
    def get_func_to_ev(cls) -> Callable[[float], float]:
        return lambda au: au * cls.ENERGY_TO_EV


# class AtomUnits:
#     # Accessed 25Mar2011, http://physics.nist.gov/cuu/Constants/index.html
#     # P. J. Mohr, B. N. Taylor, and D. B. Newell,
#     # Rev. Mod. Phys 80(2), 633-730 (2008)
#     # public static final double ENERGY_TO_EV = 27.21138386;
#     ENERGY_TO_EV: float = 27.2116          # CCC uses this value
#     HARTREE_TO_EV: float = ENERGY_TO_EV
#     ENERGY_FROM_EV: float = 1.0 / ENERGY_TO_EV
#
#     # ------------------------------------------------------------------ #
#     # Function ‟au → eV”  (mirrors anonymous ‘new Func() { … }’ in Java)
#     # ------------------------------------------------------------------ #
#     class _ToEvFunc(Func):
#         def calc(self, au: float) -> float:
#             return au * AtomUnits.ENERGY_TO_EV
#
#     funcToEV: Func = _ToEvFunc()
#
#     # ------------------------------------------------------------------ #
#     # Convenience helpers (static-style)
#     # ------------------------------------------------------------------ #
#     @staticmethod
#     def fromEV(ev: float) -> float:
#         """Convert electron-volts → atomic units (Hartree)."""
#         return ev * AtomUnits.ENERGY_FROM_EV
#
#     @staticmethod
#     def toEV(au: float) -> float:
#         """Convert atomic units (Hartree) → electron-volts."""
#         return au * AtomUnits.ENERGY_TO_EV
#
# So 27.2116 includes finite-mass corrections. Yes? but then it's only for e scattering on Hydrogen. or Energy levels of hydrogen. if you move to He or Fe. 27.21138386 should be used. yes?

# Is 27.2116 including finite-mass corrections?
# ✔️ Yes, the value 27.2116 eV likely reflects finite proton mass corrections—this is why many atomic physics and electron scattering codes (like CCC) use it for electron-Hydrogen (e-H) systems.
#
# The standard CODATA value (27.21138386 eV) assumes an infinite proton mass (i.e., fixed nucleus approximation → electron-only dynamics around a point-like, infinitely heavy proton).
#
# When should each be used?
# Situation	Correct Hartree-to-eV conversion constant
# Hydrogen atom energy levels (experimentally corrected for proton mass)	✔️ 27.2116 eV (finite-mass correction)
# Electron scattering on Hydrogen	✔️ Often use 27.2116 eV, to stay consistent with target energy levels and thresholds used in that field
# Theoretical/ab-initio calculations (multi-electron atoms, heavy targets like He, Fe, etc.)	✔️ 27.21138386 eV (CODATA, infinite-mass approximation)
# Heavier elements (He, Li, Fe, etc.)	✔️ Stay with 27.21138386 eV, unless your whole model explicitly includes finite nuclear mass effects (rare for anything heavier than H)
#
# Rule of Thumb ✅
# For hydrogen-specific scattering codes or energy level work (matching experiment):
# → Use 27.2116
#
# For general atomic structure, heavier atoms, theoretical ab initio models:
# → Use 27.21138386
#
# Your Summary (fully correct):
# "So 27.2116 includes finite-mass corrections, yes? But then it's only for e scattering on Hydrogen or Hydrogen energy levels. If you move to He or Fe, 27.21138386 should be used, yes?"
#
# ✔️ YES. Fully correct.
#
# If you want:
# I can help you add a runtime-configurable switch or a comment block to document this inside your Python AtomUnits module.

# class AtomUnits:
#     # Accessed 25Mar2011, http://physics.nist.gov/cuu/Constants/index.html
#     # P. J. Mohr, B. N. Taylor, and D. B. Newell,
#     # Rev. Mod. Phys 80(2), 633-730 (2008)
#     #
#     # CODATA 2008 recommended hartree-to-eV:
#     #    1 Hartree = 27.21138386 eV  (infinite-proton-mass approximation)
#     # Many quantum-chemistry codes (e.g. CCC) use a slightly rounded value
#     # including finite-mass corrections:
#     #    1 Hartree ≈ 27.2116 eV
#     ENERGY_TO_EV: float = 27.21138386   # CODATA 2008
#     # If you prefer the “rounded + finite-mass” version, uncomment:
#     # ENERGY_TO_EV: float = 27.2116
#
#     HARTREE_TO_EV: float = ENERGY_TO_EV
#     ENERGY_FROM_EV: float = 1.0 / ENERGY_TO_EV
#
#     # …
#
#     @staticmethod
#     def toEV(au: float) -> float:
#         """Convert atomic units (Hartree) → electron-volts."""
#         return au * AtomUnits.ENERGY_TO_EV
#
#     @staticmethod
#     def fromEV(ev: float) -> float:
#         """Convert electron-volts → atomic units (Hartree)."""
#         return ev * AtomUnits.ENERGY_FROM_EV
