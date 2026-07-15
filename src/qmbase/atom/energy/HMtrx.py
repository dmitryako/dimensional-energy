# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : HMtrx.py Created : 2025-06-30 at 10:45 am by Dmitry.A.Konovalov@gmail.com
# File: atom/energy/HMtrx.py
# Copyright Dmitry.A.Konovalov
# Created: 16/02/2010, 2:16:07 PM (original Java version)
import numpy as np

from _new25.dbg import dbg
from qm_math.mtrx.api.Mtrx import Mtrx
from qm_math.mtrx.api.EigenSymm import EigenSymm
from qm_math.vec.Vec import Vec
from atom.data.AtomUnits import AtomUnits

class HMtrx(Mtrx):
    def __init__(self, *, rows=None, cols=None, mh=None):
        """
        Constructor overload:
        - HMtrx(m, n) → zero matrix
        - HMtrx(mh) → copy from another Mtrx
        """
        if mh is not None:
            super().__init__(data=mh)
        elif rows is not None and cols is not None:
            super().__init__(rows=rows, cols=cols)
            assert mh is None
        else:
            raise ValueError("HMtrx requires either (m, n) or (mh) argument")

        self._eig = None  # Lazy-loaded EigenSymm decomposition
        self._S = None

    # public EigenSymm eigSymm()
    def eigSymm(self):
        return self._eig_decomp(overwrite=False)

    # private EigenSymm eig(boolean overwrite)
    def _eig_decomp(self, overwrite=False):
        if self._eig is None:
            self._eig = EigenSymm(self, overwrite)  # NOTE true for isSymm
        return self._eig

    # final public Vec getEigEngs()
    def getEigEngs(self):
        return self.getEigVal(overwrite=False)

    def getEigEngs_withS(self):
        # gen_eig
        H_np = self.mtrx
        S_np = self._S.mtrx
        # from atom.energy.HMtrx import solve_gen_eig, check_S_orthonormality
        E, C = solve_gen_eig(H_np, S_np)  # ascending; E[0] is ground state
        print("Ground state energy:", E[0])
        print("S-ortho error:", check_S_orthonormality(S_np, C[:, :6]))
        sysEngs = Vec(E)
        # return self.getEigVal(overwrite=False)
        return sysEngs

    # public Vec getEigVal(boolean overwrite)
    def getEigVal(self, overwrite=False):
        eig = self._eig_decomp(overwrite)
        res = eig.getRealEVals()
        return Vec(res)

    # public Vec getEngEv(int fromIdx)
    def getEngEv(self, fromIdx):
        engs = self.getEigEngs().getArr()
        res = Vec(len(engs))
        for i in range(res.size()):
            e = AtomUnits.toEV(engs[i] - engs[fromIdx])
            res.set(i, e)
        return res

    # final public Mtrx getEigVec()
    def getEigVec(self):
        return self.getEigVecOverwrite(overwrite=False)

    # public Mtrx getEigVec(boolean overwrite)
    def getEigVecOverwrite(self, overwrite=False):
        eig = self._eig_decomp(overwrite)
        return eig.getV()

    # @staticmethod
def hermitian_error(H):
    if isinstance(H, HMtrx):
        H = H.mtrx
    # Return relative symmetry error  ||H - H.T|| / ||H||
    # using the Frobenius norm.
    num = np.linalg.norm(H - H.T.conj())        # antisymmetric part
    den = np.linalg.norm(H)                     # full matrix size
    herm_err = num / den
    from _new25.dbg import dbg
    dbg('herm_err')
    return herm_err


# ---------------- 250816
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh, cholesky, solve_triangular
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse

def symmetrize(A):
    return 0.5 * (A + A.T.conj())

def solve_gen_eig(H, S, k=None, sigma=None, which='SA', use_sparse=False):
    """
    Solve H c = E S c for lowest k eigenpairs (Hermitian H, SPD S).

    Args:
        H, S: numpy arrays (dense) or scipy sparse matrices (Hermitian/Symmetric).
        k: number of smallest eigenpairs to return. If None, return all (dense only).
        sigma: shift-invert target (for eigsh); finds eigenvalues near sigma.
        which: 'SA' (smallest algebraic) for ground state; with sigma use 'LM' (largest magnitude near sigma).
        use_sparse: if True, use eigsh; else use dense eigh.

    Returns:
        evals (ascending), evecs (columns), S-orthonormal: V^T S V = I.
    """
    # Light symmetrization to remove tiny quadrature asymmetries
    dbg([H, S])
    H = symmetrize(H)
    S = symmetrize(S)

    # Ensure S is positive definite; small diagonal jitter can help if needed
    # S += 1e-14 * np.eye(S.shape[0])

    if use_sparse or issparse(H) or issparse(S):
        if k is None:
            raise ValueError("Sparse mode requires k (number of eigenpairs).")
        # For generalized problems, eigsh supports SPD mass matrix M=S
        # - Use which='SA' for smallest eigenvalues
        # - With sigma (shift-invert), set which='LM' to get those closest to sigma
        if sigma is None:
            evals, evecs = eigsh(H, k=k, M=S, which=which)
        else:
            evals, evecs = eigsh(H, k=k, M=S, sigma=sigma, which='LM')
        # Columns of evecs are already S-orthonormal
        order = np.argsort(evals)
        return evals[order], evecs[:, order]

    # Dense: use eigh for generalized Hermitian eigenproblem
    if k is None:
        evals, evecs = eigh(H, S)  # returns all eigenpairs sorted ascending
        return evals, evecs

    # If you only want k lowest in dense mode, do a Cholesky reduction then slice
    L = cholesky(S, lower=True, check_finite=False)
    # Reduce to standard problem: (L^-1 H L^-T) y = E y
    Linv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)
    Hred = Linv.T @ H @ Linv
    Hred = symmetrize(Hred)
    evals, Y = eigh(Hred)  # all; cheap for N~few hundred
    # Back-transform: C = L^-T Y
    C = Linv @ Y
    # Take k smallest
    return evals[:k], C[:, :k]

def check_S_orthonormality(S, V):
    """Returns max |V^T S V - I| (should be ~1e-12 or so)."""
    G = V.T.conj() @ (S @ V)
    return norm(G - np.eye(G.shape[0]), ord=np.inf)

# Dense, all eigenpairs (small
# 𝑁
# N
# E, C = solve_gen_eig(H, S)  # ascending; E[0] is ground state
# print("Ground state energy:", E[0])
# print("S-ortho error:", check_S_orthonormality(S, C[:, :6]))

# Sparse, only k lowest (recommended when
# 𝑁
# N is large):
# k = 6
# E, C = solve_gen_eig(H, S, k=k, which='SA', use_sparse=True)
# print("Lowest energies:", E)
# print("S-ortho error:", check_S_orthonormality(S, C))

# Target near a value (shift-invert), e.g. look near
# −
# 0.5
# −0.5:
# k = 4
# E, C = solve_gen_eig(H, S, k=k, sigma=-0.5, which='LM', use_sparse=True)
