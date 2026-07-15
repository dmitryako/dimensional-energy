# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : AtomUnits.py Created : 2025-06-30 at 10:22 am by Dmitry.A.Konovalov@gmail.com
# ---------------------------------------------------------------------------
#  Python port of atom.data.AtomUnits
#  (keeps *all* original Java comments for traceability)
# ---------------------------------------------------------------------------

from __future__ import annotations
from typing import Callable

from qm_math.func.Func import Func



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


