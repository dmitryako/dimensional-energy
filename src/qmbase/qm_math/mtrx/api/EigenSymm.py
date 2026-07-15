# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : EigenSymm.py Created : 2025-06-30 at 10:18 am by Dmitry.A.Konovalov@gmail.com
# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : EigenSymm.py   Converted : 2025-06-29

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from qm_math.mtrx.api.Mtrx import Mtrx
from javax.utilx.log.Log import Log      # re-use your unified logger


class EigenSymm:
    """
    NumPy implementation of the old `EigenSymm extends EigenEjml`.

    Parameters
    ----------
    mtrx : Mtrx
        Real symmetric matrix to diagonalise.
    overwrite : bool, default ``True``
        If *True* the underlying NumPy array may be reused (NumPy’s LAPACK
        routines honour this flag), otherwise a copy is made first.
    """

    log = Log.getLog("EigenSymm")

    # ------------------------------------------------------------------ #
    def __init__(self, mtrx: Mtrx, overwrite: bool = True) -> None:
        if mtrx.getNumRows() != mtrx.getNumCols():
            raise ValueError("EigenSymm requires a square matrix.")

        # store a *view* or a copy according to overwrite flag
        # self._A: np.ndarray = (
        #     mtrx._mtrx if overwrite else mtrx._mtrx.copy()
        # )
        if overwrite:
            self._A = mtrx.mtrx
        else:
            self._A = mtrx.mtrx.copy()
        # lazily computed results
        self._vals: Optional[np.ndarray] = None
        self._vec: Optional[Mtrx] = None
        self._D: Optional[Mtrx] = None

    # ------------------------------------------------------------------ #
    # Public API — mirrors Java version
    # ------------------------------------------------------------------ #
    def getRealEVals(self) -> np.ndarray:
        """Return the eigen-values (real).  Sorted in *ascending* order."""
        if self._vals is None:
            self._compute()
        assert self._vals is not None
        return self._vals

    def getV(self) -> Mtrx:
        """Return eigen-vectors column-wise, sorted like the eigen-values."""
        if self._vec is None:
            self._compute()
        assert self._vec is not None
        return self._vec

    def getD(self) -> Mtrx:
        """Return diagonal matrix of eigen-values (matching `getV()`)."""
        if self._D is None:
            self._compute()
        assert self._D is not None
        return self._D

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _compute(self) -> None:
        """
        Perform the eigen-decomposition exactly once, cache results.
        """

        # -------- LAPACK call -------- #
        # eigh is specialised for (Hermitian) symmetric matrices
        vals, vecs = np.linalg.eigh(self._A, UPLO="L")
        # vals: (n,)  ascending by default, vecs: (n, n) columns = eigen-vectors

        # -------- Wrap into Mtrx -------- #
        self._vals = vals
        self._vec = Mtrx(data=vecs)
        self._D = Mtrx(data=np.diag(vals))

        # Debug log (optional)
        self.log.dbg("Eigenvalues:", self._vals)
