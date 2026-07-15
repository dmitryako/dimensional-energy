# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : H1d1e.py Created : 2025-07-13 at 7:46 am by Dmitry.A.Konovalov@gmail.com
from typing import cast

import numpy as np
import scipy.sparse
import scipy.sparse as sp
from matplotlib.ticker import ScalarFormatter

from _new25.dbg import dbg
from atom.wf.lcr.WFQuadrLcr import WFQuadrLcr
from d1.d1e2.FuncArr1d2e import FuncArr1d2e
from scatt.jm_2008.jm.laguerre.IWFuncArr import IWFuncArr


class H1d1e:
    def __init__(self, atom_z):
        self.atom_z = atom_z
    @staticmethod
    def calc_eng_1d1e(*, atom_z, n=1):
        # the same as 3d1e
        # todo: H2d1e.calc_eng_2d1e
        # todo: H3d1e.calc_eng_3d1e
        res = -float(atom_z)**2 / (2. * n**2)
        return float(res)

    @staticmethod
    def generate_one_hot_basis_dense(x, buffer=None):
        # Generates a one-hot encoded basis (delta-like) on grid x,
        # with 'buffer' points zeroed on each side.
        # Parameters:
        #     x (np.ndarray): 1D grid array (length N)
        #     buffer (int): Number of grid points to leave as zero on each side
        # Returns:
        #     orth1 (np.ndarray): (len(x), nx) matrix with each column a basis vector
        #                         normalized to have 1 at one internal grid point
        N = len(x)
        nx = N - 2 * buffer  # number of usable basis functions
        if nx <= 0:
            raise ValueError(f"Grid too small for buffer size {buffer}")
        basis = np.zeros((N, nx), dtype=np.float64)
        for i in range(nx):
            basis[i + buffer, i] = 1.0  # delta at inner point with buffer
        return basis  # already in (N, nx) form, vectors in columns

    # @staticmethod
    # def generate_one_hot_basis_dense_skip1(x, buffer=None):
    #     N = len(x)
    #     nx = N - 2 * buffer  # number of usable basis functions
    #     nx  = nx // 2
    #     if nx <= 0:
    #         raise ValueError(f"Grid too small for buffer size {buffer}")
    #     basis = np.zeros((N, nx), dtype=np.float64)
    #     for i in range(0, nx, 2):
    #         basis[i + buffer, i] = 1.0  # delta at inner point with buffer
    #     return basis  # already in (N, nx) form, vectors in columns

    @staticmethod
    def generate_one_hot_basis_dense_skip1(x, buffer=2):
        """
        One-hot basis that uses every second usable grid point.
        Result has shape (N, n_basis); columns are basis vectors.

        Parameters
        ----------
        x : (N,) ndarray
            Uniform log-grid points.
        buffer : int, optional
            Number of grid points to skip at each boundary (default 2).

        Returns
        -------
        basis : (N, n_basis) ndarray
            One-hot basis with points  buffer, buffer+2, buffer+4, ...
        """
        N = len(x)
        # index of first usable point     = buffer
        # index of last  usable point     = N-buffer-1
        usable = N - 2 * buffer
        if usable <= 0:
            raise ValueError("buffer too large for grid")

        # choose every 2nd usable point
        grid_indices = np.arange(buffer, N - buffer, 2)  # buffer, buffer+2, ...
        n_basis = len(grid_indices)

        basis = np.zeros((N, n_basis), dtype=float)
        for j, idx in enumerate(grid_indices):
            basis[idx, j] = 1.0

        return basis



    @staticmethod
    def remove_2e_diag_basis2e_sp(orth2_sp, *, n_funcs):
        # Removes columns from orth2_sp corresponding to diagonal entries where x1_i == x2_i.
        # Assumes orth2_sp = kron(orth1_sp, orth1_sp), and n_funcs is the number of basis functions.
        assert scipy.sparse.isspmatrix_csr(orth2_sp), "orth2_sp must be CSR sparse matrix"
        dbg(orth2_sp)
        total_cols = orth2_sp.shape[1]
        dbg('total_cols')
        diag_col_indices = np.array([i * n_funcs + i for i in range(n_funcs) if i * n_funcs + i < total_cols])
        dbg(diag_col_indices)
        dbg('diag_col_indices[:10]')
        keep_cols_mask = np.ones(total_cols, dtype=bool)
        dbg(keep_cols_mask)
        keep_cols_mask[diag_col_indices] = False
        dbg(keep_cols_mask)
        res = orth2_sp[:, keep_cols_mask]
        dbg(res)
        return res

    @staticmethod
    def remove_2e_diag_from_2e_mat(mat_full, *, n_basis):
        # Remove rows & columns where the 2-electron basis index corresponds to (i==j).
        # mat_full : scipy.sparse.spmatrix
        #     Square matrix on the full 2-electron product basis.
        #     Expected shape = (n_basis**2, n_basis**2)
        # n_basis : int
        #     Number of 1-electron basis functions (columns in orth1_sp).
        # Returns
        # mat_reduced : csr_matrix
        #     Matrix in the filtered 2-electron basis (no i==j terms).
        # keep_idx : np.ndarray
        #     Indices kept, so that   mat_reduced = mat_full[keep_idx, :][:, keep_idx]
        # --- make sure in CSR for efficient slicing ---------------------------
        if not sp.isspmatrix_csr(mat_full):
            mat_full = mat_full.tocsr()
        dbg(mat_full)
        # --- build index list  (i != j) ---------------------------------------
        keep_idx = [i * n_basis + j
                    for i in range(n_basis)
                    for j in range(n_basis)
                    if i != j]
        keep_idx = np.array(keep_idx, dtype=np.int64)
        dbg(keep_idx)  # length should be n_basis*(n_basis-1)
        # --- slice rows then columns ------------------------------------------
        mat_reduced = mat_full[keep_idx, :][:, keep_idx].tocsr()
        dbg(mat_reduced)
        return mat_reduced, keep_idx

    @staticmethod
    def max_diff_sparse(A_sp, B_sp):
        import scipy.sparse as sp
        assert sp.isspmatrix(A_sp) and sp.isspmatrix(B_sp), "Both must be sparse matrices"
        diff = A_sp - B_sp
        if diff.nnz == 0:
            # print("Matrices are exactly equal (no non-zero differences).")
            dbg('diff.nnz')
            return 0.0
        max_abs_diff = np.max(np.abs(diff.data))
        # print(f"Max absolute difference: {max_abs_diff:.3e}")
        dbg('max_abs_diff')
        return max_abs_diff

    @staticmethod
    def remove_diags_from_orth2_sp(orth1_sp):
        from scipy.sparse import kron, csr_matrix
        """
        Constructs orth2_sp from orth1_sp and removes diagonal basis columns where x1 == x2.
        Args:
            orth1_sp (csr_matrix): Sparse orthonormal 1-electron basis of shape (nx, nb).
        Returns:
            orth2_sp (csr_matrix): Sparse 2-electron orthonormal basis (excluding diagonal).
            filtered_indices (List[Tuple[int, int]]): List of (i, j) basis indices kept.
        """
        nb = orth1_sp.shape[1]  # number of 1e basis functions
        dbg('nb')
        # Build list of (i, j) for which i != j
        filtered_indices = [(i, j) for i in range(nb) for j in range(nb) if i != j]
        dbg('filtered_indices[:10]')
        # Map (i, j) to linear basis index: i * nb + j
        col_indices = [i * nb + j for i, j in filtered_indices]
        dbg('col_indices[:10]')
        # Build full orth2_sp
        orth2_full = kron(orth1_sp, orth1_sp).tocsr()
        dbg(orth2_full)
        # Filter out diagonal basis vectors
        orth2_sp = orth2_full[:, col_indices]
        dbg(orth2_sp)
        return orth2_sp, filtered_indices


    @staticmethod
    def calc_norm_max_err(orth1, S):
        pred_norm = orth1.T @ S @ orth1
        assert pred_norm.shape[0] == pred_norm.shape[1]
        true_norm = np.eye(pred_norm.shape[0], dtype=np.float64)
        max_err = np.max(np.abs(pred_norm - true_norm))
        dbg('max_err')
        return max_err

    @staticmethod
    def calc_norm_max_err_sparse(orth1_sp, *, S_sp):
        assert scipy.sparse.isspmatrix_csr(orth1_sp), "orth1_sparse must be a CSR matrix"
        assert scipy.sparse.isspmatrix_csr(S_sp), "S_sp must be a CSR matrix"

        pred_norm_sp = orth1_sp.T @ (S_sp @ orth1_sp)        # Compute the Gram matrix: orth1.T @ S @ orth1
        pred_norm = pred_norm_sp.toarray()
        true_norm = np.eye(pred_norm.shape[0], dtype=np.float64)
        max_err = np.max(np.abs(pred_norm - true_norm))
        dbg('max_err')
        return max_err

    @staticmethod
    def calc_norm_max_err_2e_sparse(orth2_sp, *, S_sp):
        """
        Verifies orthonormality of the 2-electron sparse basis `orth2_sp`.
        Assumes `orth2_sp = kron(orth1_sp, orth1_sp)` and `S_sp` is 1-electron quadrature matrix (e.g., diag(h)).
        Parameters:
            orth2_sp (scipy.sparse.csr_matrix): Two-electron basis matrix (n^2, nb^2 or filtered nb2)
            S_sp (scipy.sparse.csr_matrix): One-electron quadrature weight matrix (n, n)
        Returns:
            float: Maximum absolute error from identity in the Gram matrix.
        """
        assert scipy.sparse.isspmatrix_csr(orth2_sp), "orth2_sp must be CSR sparse matrix"
        assert scipy.sparse.isspmatrix_csr(S_sp), "S_sp must be CSR sparse matrix"
        # Kronecker quadrature weight matrix for two-particle space: S ⊗ S
        S2_sp = scipy.sparse.kron(S_sp, S_sp, format='csr')
        dbg(S2_sp)
        # Compute Gram matrix: orth2.T @ S2 @ orth2
        pred_norm_sp = orth2_sp.T @ (S2_sp @ orth2_sp)
        dbg(pred_norm_sp)
        # Diagonal error
        identity_diag = np.ones(pred_norm_sp.shape[0], dtype=np.float64)
        dbg(identity_diag)
        pred_diag = pred_norm_sp.diagonal()
        dbg(pred_diag)
        diag_errs = np.abs(pred_diag - identity_diag)
        dbg(diag_errs)
        max_diag_error = np.max(diag_errs)
        dbg('max_diag_error')
        # Off-diagonal error
        pred_norm_sp_no_diag = pred_norm_sp - scipy.sparse.diags(pred_diag, format='csr')
        dbg(pred_norm_sp_no_diag)
        if pred_norm_sp_no_diag.nnz > 0:
            max_off_diag_error = np.max(np.abs(pred_norm_sp_no_diag.data))
        else:
            max_off_diag_error = 0.0
        # Final maximum error
        max_err = max(max_diag_error, max_off_diag_error)
        dbg('max_err')
        return max_err


    @staticmethod
    def calc_norm_max_err_2e_sparse_BAD(orth2_sp, *, S_sp):
        """
        Verifies orthonormality of the 2-electron sparse basis `orth2_sp`.
        Assumes `orth2_sp = kron(orth1_sp, orth1_sp)` and `S_sp` is 1-electron quadrature matrix (e.g., diag(h)).
        Parameters:
            orth2_sp (scipy.sparse.csr_matrix): Two-electron basis matrix (n^2, nb^2 or filtered nb2)
            S_sp (scipy.sparse.csr_matrix): One-electron quadrature weight matrix (n, n)
        Returns:
            float: Maximum absolute error from identity in the Gram matrix.
        """
        assert scipy.sparse.isspmatrix_csr(orth2_sp), "orth2_sp must be CSR sparse matrix"
        assert scipy.sparse.isspmatrix_csr(S_sp), "S_sp must be CSR sparse matrix"
        # Kronecker quadrature weight matrix for two-particle space: S ⊗ S
        dbg(orth2_sp)
        dbg(S_sp)
        S2_sp = scipy.sparse.kron(S_sp, S_sp, format='csr')
        dbg(S2_sp)
        # Compute Gram matrix: orth2.T @ S2 @ orth2
        pred_norm_sp = orth2_sp.T @ (S2_sp @ orth2_sp)
        dbg(pred_norm_sp)
        pred_norm = pred_norm_sp.toarray()
        dbg(pred_norm)
        # Compare with identity
        true_norm = np.eye(pred_norm.shape[0], dtype=np.float64)
        dbg(true_norm)
        max_err = np.max(np.abs(pred_norm - true_norm))
        dbg(max_err)
        return max_err

    @staticmethod
    def calc_norm_max_err_sparse_approx(orth_sp, *, S_sp, topk=5):
        """
        Efficiently estimates the max orthonormality error from Gram matrix of a sparse basis:
            max|orth_sp.T @ S_sp @ orth_sp - I|
        without converting to dense.

        Args:
            orth_sp (csr_matrix): basis matrix (N, n_basis)
            S_sp (csr_matrix): diagonal quadrature matrix (N, N)
            topk (int): number of top off-diagonal errors to print (optional)

        Returns:
            float: max absolute error from identity
        """
        import scipy.sparse as sp
        assert sp.isspmatrix_csr(orth_sp), "orth_sp must be CSR matrix"
        assert sp.isspmatrix_csr(S_sp), "S_sp must be CSR matrix"

        Gram_sp = orth_sp.T @ (S_sp @ orth_sp)

        # Check diagonal entries (should be 1)
        diag = Gram_sp.diagonal()
        diag_err = np.abs(diag - 1.0)
        max_diag_err = np.max(diag_err)

        # Check off-diagonal entries (should be 0)
        off_diag_mask = Gram_sp.copy()
        off_diag_mask.setdiag(0)
        off_diag_abs = np.abs(off_diag_mask.data)
        max_off_diag_err = np.max(off_diag_abs) if off_diag_abs.size > 0 else 0.0

        # Optionally print largest off-diagonal errors
        if topk > 0 and off_diag_abs.size > 0:
            top_errors = np.sort(off_diag_abs)[-topk:]
            print(f"Top {topk} off-diagonal errors: {top_errors}")

        max_err = max(max_diag_err, max_off_diag_err)
        print(f"Max diag error: {max_diag_err:.3e}, max off-diag error: {max_off_diag_err:.3e}")
        return max_err

    @staticmethod
    def plot_wf(x_grid, *, pred_wf, true_wf):
        import matplotlib.pyplot as plt
        # analytic_wf = self.wf0_np
        plt.figure(figsize=(8, 4))
        plt.plot(x_grid, pred_wf, label='pred_wf(x)')
        plt.plot(x_grid, true_wf, '--', label='true_wf(x)')
        plt.xlabel('x')
        plt.ylabel('ψ(x)')
        plt.title('pred_wf vs true_wf')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_wf_many(x_grid, *, wfs_in_cols, true_wf):
        import matplotlib.pyplot as plt
        n = 5
        plt.figure(figsize=(8, 6))
        for i in range(n):
            plt.plot(x_grid, wfs_in_cols[:, i], label=f"$\psi_{i}(r)$")
        # plt.figure(figsize=(8, 4))
        # plt.plot(x_grid, pred_wf, label='pred_wf(x)')
        plt.plot(x_grid, true_wf, '--', label='true_wf(x)')
        plt.xlabel('x')
        plt.ylabel('ψ(x)')
        plt.title('pred_wf vs true_wf')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_wf_many(x_grid, *, wfs_in_cols, true_wf, zoom=None):
        import matplotlib.pyplot as plt
        if zoom is not None:
            i1, i2 = zoom
            x_grid = x_grid[i1:i2]
            true_wf = true_wf[i1:i2]
            wfs_in_cols = wfs_in_cols[i1:i2, :]

        n = 5
        plt.figure(figsize=(8, 6))
        for i in range(n):
            plt.plot(x_grid, wfs_in_cols[:, i], label=f"$\psi_{i}(r)$")
        # plt.figure(figsize=(8, 4))
        # plt.plot(x_grid, pred_wf, label='pred_wf(x)')
        plt.plot(x_grid, true_wf, '--', label='true_wf(x)')
        plt.xlabel('x')
        plt.ylabel('ψ(x)')
        plt.title('pred_wf vs true_wf')
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    # 4.  Quick plot  (|Ψ| or Ψ) ---------------------------------------
    import matplotlib.pyplot as plt

    @staticmethod
    def plot_wf_2e(psi2e_grid, *, zoom=None, title=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(psi2e_grid, origin='lower',
                   # extent=[r.min(), r.max(), r.min(), r.max()],
                   aspect='auto', cmap='RdBu')
        plt.colorbar(label=r'$\Psi(r_1,r_2)$')
        plt.xlabel(r'$r_1$')
        plt.ylabel(r'$r_2$')
        if title is not None:
            plt.title(title)
        else:
            plt.title("2-electron ground-state wave-function")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_wf_2e_PRA_v1(psi2e_grid, *, cfg, zoom=None, title=None):
        import matplotlib.pyplot as plt
        orth2e = cast(FuncArr1d2e, cfg.orth2e)
        B1 = cast(IWFuncArr, orth2e.basis_1d)
        B2 = cast(IWFuncArr, orth2e.basis2_1d)
        x1 = B1.getQuadr().getX().arr
        x2 = B2.getQuadr().getX().arr

        fig, ax = plt.subplots(figsize=(4, 4))  # small enough to fit PRA column width

        cmap = plt.get_cmap("RdBu_r")  # Red–blue color map, diverging at zero
        im = ax.imshow(
            psi2e_grid,
            origin="lower",
            extent=[x1[0], x1[-1], x2[0], x2[-1]],
            cmap=cmap,
            aspect="equal"
        )

        # Axis labels
        ax.set_xlabel(r"$r_1$")
        ax.set_ylabel(r"$r_2$")
        ax.set_title(r"$\psi_{2\times1e}^{\mathrm{ANTI}}$")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"$\Psi(r_1, r_2)$")

        # Save at 300 dpi (minimum for journals)
        plt.tight_layout()
        plt.savefig("psi2x1e_ANTI_PRA.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def plot_wf_2e_PRA(psi2d, title, cfg, sign=1):
        # H1d1e.plot_wf_2e_PRA_v1(psi2e_grid, *, cfg, zoom=None, title=None)
        # H1d1e.plot_wf_2e_PRA_v2_OK(psi2d, cfg=cfg, title=title)
        # H1d1e.plot_wf_2e_PRA_v3_withPx1(psi2d, cfg=cfg, title=title)
        H1d1e.plot_wf_2e_PRA_v4_withPx1(psi2d, cfg=cfg, title=title)


    @staticmethod
    def plot_wf_2e_PRA_v3_withPx1(psi2e_grid, *, cfg, zoom=None, title=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        orth2e = cast(FuncArr1d2e, cfg.orth2e)
        B1 = cast(IWFuncArr, orth2e.basis_1d)
        B2 = cast(IWFuncArr, orth2e.basis2_1d)
        x1 = B1.getQuadr().getX().arr
        x2 = B2.getQuadr().getX().arr
        dbg([psi2e_grid, x1, x2])

        # fix sign
        i = len(x1) // 4
        j = 2 * len(x2) // 4
        if psi2e_grid[i, j] < 0.:
            psi2e_grid *= -1

        X_MIN = -2  # for plot
        X_MAX = 4
        # Compute marginal P(x1) = ∫ |ψ(x1,x2)|² dx2
        B2.getQuadr()
        wLcr1 = cast(WFQuadrLcr, cfg.wLcr1)
        wLcr2 = cast(WFQuadrLcr, cfg.wLcr2)
        sqrtCr1 = wLcr1.getSqrtCR().arr
        sqrtCr2 = wLcr2.getSqrtCR().arr
        cr2_1 = wLcr1.getCR2().arr
        cr2_2 = wLcr2.getCR2().arr
        cr2w1 = wLcr1.getWithCR2().arr  # for integral
        cr2w2 = wLcr2.getWithCR2().arr  # for integral
        dbg([cr2w2, cr2_1, sqrtCr1, sqrtCr2])
        P_x1 = psi2e_grid**2 @  cr2w2
        dbg([psi2e_grid, P_x1])
        # P_x1 = np.sum(psi2e_grid ** 2 * cr2w2[None, :], axis=1)  # the same !
        # dbg([psi2e_grid, P_x1])
        # TEST norm
        norm = np.sum(cr2w1 * P_x1)
        dbg('norm')
        assert abs(norm - 1) < 1e-8, 'erro norm'

        # convert to r1, r1
        # psi2e_grid_r1r2 = sqrtCr1[:, None] * psi2e_grid * sqrtCr2[None, :]  # missing dr1 dr2
        # psi2e_grid_r1r2 = cr2_1[:, None] * psi2e_grid * cr2_2[None, :]  # todo <---- YES??
        # psi2e_grid_r1r2 = sqrtCr1[:, None] * psi2e_grid * sqrtCr2[None, :]  # sqrtCr to func_in r
        psi2e_grid_r1r2 = sqrtCr1[:, None]**2 * psi2e_grid * sqrtCr2[None, :]**2  # sqrtCr to func_in r
        # if now sum psi2e_grid_r1r2**2 = 1, since it will have CR2
        # need to add dr but in dx space?
        P_r1 = cr2_1 * P_x1  # todo!!! remember it's psi2e_grid**2 AND dr so it's cr^2

        fig, axs = plt.subplots(
            2, 1,
            figsize=(3.3, 5.2),
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}  # reduce vertical gap
        )

        # im = axs[0].imshow(
        #     # psi2e_grid.T,  # your 2D data todo <--- NOTE .T orig is (1001, 1021)
        #     psi2e_grid_r1r2.T,
        #     origin="lower",
        #     extent=[x1[0], x1[-1], x2[0], x2[-1]],
        #     cmap="RdBu_r",
        #     aspect="equal"
        # )
        # Colormap	Description
        # "coolwarm"	Smooth, perceptually symmetric
        # "seismic"	High-contrast red-blue
        # "twilight"	Very modern, soft transitions
        # "RdBu_r"	Your current one; fine, but harsh
        vmax = np.max(np.abs(psi2e_grid_r1r2))
        vmin = -vmax
        im = axs[0].imshow(
            psi2e_grid_r1r2.T,  # transpose if (x1,x2) axes need swapping
            origin="lower",
            extent=[x1[0], x1[-1], x2[0], x2[-1]],
            cmap="coolwarm",  # or choose from suggestions below
            # vmin=-np.max(np.abs(psi2e_grid_r1r2)),
            # vmax=+np.max(np.abs(psi2e_grid_r1r2)),
            vmin=vmin,
            vmax=vmax,
            aspect="equal"
        )

        cax = inset_axes(
            axs[0],
            width="4%",        # thinner
            height="40%",      # relative height
            loc='upper right',
            #  [left, bottom, width, height],
            # bbox_to_anchor=(-0.05, 0.05, 1, 1),  # tweak x and y shift here
            # bbox_to_anchor=(-0.15, -0.05, 1, 1),  # tweak x and y shift here
            bbox_to_anchor=(-0.2, -0.05, 1, 1),  # tweak x and y shift here
            bbox_transform=axs[0].transAxes,
            borderpad=0        # minimal padding
        )
        # fig.colorbar(im, cax=cax)
        cbar = fig.colorbar(im, cax=cax)
        # fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
        # fig.colorbar(im, ax=axs[0])
        # cbar.set_label(r"$\Psi(x_1, x_2)$")

        # for gray print
        # axs[0].imshow(
        #     psi2e_grid_r1r2.T,
        #     origin="lower",
        #     extent=[x1[0], x1[-1], x2[0], x2[-1]],
        #     cmap="gray",
        #     vmin=-vmax,
        #     vmax=vmax,
        #     aspect="equal"
        # )
        axs[0].imshow(psi2e_grid_r1r2.T, ..., cmap='gray', vmin=-vmax, vmax=vmax)
        thresh = 0
        mask_pos = psi2e_grid_r1r2.T > thresh
        axs[0].contourf(x1, x2, mask_pos.T, levels=[0.5, 1],
                        hatches=['///'], colors='none',
                        origin='lower', extent=[x1[0], x1[-1], x2[0], x2[-1]])

        # ax.set_xlabel(r"$x_1$")
        # ax.set_ylabel(r"$x_2$")
        # axs[0].set_xlabel(r"$x_1$")
        axs[0].set_ylabel(r"$x_2$")
        # ax.set_title(r"$\psi_{2 \times 1e}$")
        axs[0].tick_params(axis='x', labelbottom=False)  # remove x1 ticks from top
        axs[0].set_xlim(left=X_MIN)  # ensure y starts from 0
        axs[0].set_xlim(right=X_MAX)  # ensure y starts from 0
        axs[0].set_ylim(bottom=X_MIN)  # ensure y starts from 0
        axs[0].set_ylim(top=X_MAX)  # ensure y starts from 0

        # --- Bottom: Marginal distribution ---
        # axs[1].plot(x1, P_x1, lw=1.2)
        axs[1].plot(x1, P_r1, lw=1.2)
        axs[1].set_xlabel(r'$x_1$')
        axs[1].set_ylabel(r'$P(x_1)$')
        # axs[1].set_title(r'Marginal density $P(x_1)$')
        axs[1].set_xlim(left=X_MIN)  # ensure y starts from 0
        axs[1].set_xlim(right=X_MAX)  # ensure y starts from 0
        axs[1].set_ylim(bottom=0)  # ensure y starts from 0

        plt.tight_layout()
        # plt.savefig("psi2x1e_inset_colorbar.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{title}_low_res.png", bbox_inches='tight')
        plt.savefig(f"{title}_300dpi.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{title}_600dpi.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def plot_wf_2e_PRA_v4_withPx1(psi2e_grid, *, cfg, sign=1, title=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.patches as mpatches
        import matplotlib as mpl
        mpl.rcParams['hatch.linewidth'] = 1.0  # or larger

        orth2e = cast(FuncArr1d2e, cfg.orth2e)
        B1 = cast(IWFuncArr, orth2e.basis_1d)
        B2 = cast(IWFuncArr, orth2e.basis2_1d)
        x1 = B1.getQuadr().getX().arr
        x2 = B2.getQuadr().getX().arr

        # Fix sign
        i = len(x1) // 4
        j = 2 * len(x2) // 4
        if psi2e_grid[i, j] < 0.:
            psi2e_grid = -psi2e_grid * sign

        # Marginal P(x1)
        wL1, wL2 = cast(WFQuadrLcr, cfg.wLcr1), cast(WFQuadrLcr, cfg.wLcr2)
        sqrtCr1, sqrtCr2 = wL1.getSqrtCR().arr, wL2.getSqrtCR().arr
        cr2w2 = wL2.getWithCR2().arr
        cr2w1 = wL1.getWithCR2().arr
        P_x1 = psi2e_grid ** 2 @ cr2w2
        norm = np.sum(wL1.getWithCR2().arr * P_x1)
        assert abs(norm - 1) < 1e-8, f"norm error: {norm}"

        psi2e_grid_r1r2 = sqrtCr1[:, None] ** 2 * psi2e_grid * sqrtCr2[None, :] ** 2
        P_r1 = wL1.getCR2().arr * P_x1

        vmax = np.max(np.abs(psi2e_grid_r1r2))
        extent = [x1[0], x1[-1], x2[0], x2[-1]]

        fig, axs = plt.subplots(2, 1,
                                # figsize=(3.3, 5.2),
                                figsize=(6.6, 10.4),
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

        # Top panel: color heatmap
        im = axs[0].imshow(
            psi2e_grid_r1r2.T,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal"
        )
        # Inset colorbar
        cax = inset_axes(axs[0], width="4%", height="40%", loc='upper right',
                         bbox_to_anchor=(-0.2, -0.05, 1, 1),
                         bbox_transform=axs[0].transAxes, borderpad=0)
        fig.colorbar(im, cax=cax)

        # Overlay hatch for positive region (for grayscale print)
        thresh = 0.1
        mask_pos = psi2e_grid_r1r2.T > thresh
        axs[0].contourf(
            x1, x2, mask_pos.T,
            levels=[0.5, 1],
            # hatches=['///'],
            hatches=['xxxx'],
            colors='none',
            origin='lower',
            extent=extent
        )
        # Legend proxy
        hatch_patch = mpatches.Patch(facecolor='white', edgecolor='black',
                                     hatch='xxxx', label=f'< -{thresh:.1f}')
        # hatch = '///', label = f'< {thresh:.2f}')
        axs[0].legend(handles=[hatch_patch], loc='upper left')

        axs[0].set_ylabel(r"$x_2$")
        axs[0].tick_params(axis='x', labelbottom=False)
        axs[0].set_xlim(X_MIN := -2, X_MAX := 4)
        axs[0].set_ylim(X_MIN, X_MAX)

        # Bottom: Marginal
        axs[1].plot(x1, P_r1, lw=1.2)
        axs[1].set_xlabel(r"$x_1$")
        axs[1].set_ylabel(r"$P(x_1)$")
        axs[1].set_xlim(X_MIN, X_MAX)
        axs[1].set_ylim(bottom=0)

        plt.tight_layout()
        # for dpi in [300, 600]:
        for dpi in [600]:
            plt.savefig(f"{title}_{dpi}dpi.png", dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close()


    @staticmethod
    def plot_wf_2x2e_compare_PRA_v5_withPx1_onlyColurs(
            psi_left, psi_right, *, cfg, sign=1, title=None,
            labels=(None, None, "difference")):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        orth2e = cast(FuncArr1d2e, cfg.orth2e)
        B1 = cast(IWFuncArr, orth2e.basis_1d)
        B2 = cast(IWFuncArr, orth2e.basis2_1d)
        x1 = B1.getQuadr().getX().arr
        x2 = B2.getQuadr().getX().arr

        wL1, wL2 = cast(WFQuadrLcr, cfg.wLcr1), cast(WFQuadrLcr, cfg.wLcr2)
        sqrtCr1, sqrtCr2 = wL1.getSqrtCR().arr, wL2.getSqrtCR().arr
        cr2w1, cr2w2 = wL1.getWithCR2().arr, wL2.getWithCR2().arr
        cr2_1 = wL1.getCR2().arr
        extent = [x1[0], x1[-1], x2[0], x2[-1]]
        i, j = len(x1) // 4, 2 * len(x2) // 4  # sign-fix sample point (as v5)

        def to_disp(psi):
            return sqrtCr1[:, None] ** 2 * psi * sqrtCr2[None, :] ** 2

        def prep(psi):
            if psi[i, j] < 0.:  # canonical sign at shared point
                psi = -psi * sign  # -> both states sign-aligned
            P_x1 = psi ** 2 @ cr2w2
            norm = np.sum(cr2w1 * P_x1)
            assert abs(norm - 1) < 1e-8, f"norm error: {norm}"
            return psi, to_disp(psi), cr2_1 * P_x1

        psiL, dispL, P_L = prep(psi_left)
        psiR, dispR, P_R = prep(psi_right)

        disp_diff = to_disp(psiL - psiR)  # diff of SIGN-FIXED states
        dP = P_L - P_R
        max_wf = float(np.max(np.abs(psiL - psiR)))
        max_dP = float(np.max(np.abs(dP)))
        print(f"[compare] max|psi_L-psi_R| = {max_wf:.3e},  max|P_L-P_R| = {max_dP:.3e}")

        vmax = max(np.max(np.abs(dispL)), np.max(np.abs(dispR)))  # shared cols 0,1
        vmax_d = max(np.max(np.abs(disp_diff)), 1e-30)  # own, diff col
        pmax = max(np.max(P_L), np.max(P_R))  # shared cols 0,1

        disps = [dispL, dispR, disp_diff]
        Ps = [P_L, P_R, dP]
        vmaxs = [vmax, vmax, vmax_d]

        fig, axs = plt.subplots(2, 3, figsize=(19.8, 10.4),
                                gridspec_kw={'height_ratios': [3, 1],
                                             'hspace': 0.05, 'wspace': 0.12})
        X_MIN, X_MAX = cfg.plot_x_min, 4  # xmin was -2
        for col in range(3):
            is_diff = (col == 2)
            ax = axs[0, col]
            im = ax.imshow(disps[col].T, origin="lower", extent=extent,
                           cmap="coolwarm", vmin=-vmaxs[col], vmax=vmaxs[col],
                           aspect="equal")
            # cax = inset_axes(ax, width="4%", height="40%", loc='upper right',
            #                  bbox_to_anchor=(-0.2, -0.05, 1, 1),
            #                  bbox_transform=ax.transAxes, borderpad=0)
            cax = inset_axes(
                ax,
                width="4%",
                height="40%",
                loc="lower left",
                bbox_to_anchor=(0.03, 0.05, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0
            )
            fig.colorbar(im, cax=cax)
            ax.set_xlim(X_MIN, X_MAX);
            ax.set_ylim(X_MIN, X_MAX)
            ax.tick_params(axis='x', labelbottom=False)
            if labels[col]:
                ax.set_title(labels[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(r"$x_2$")
            else:
                ax.tick_params(axis='y', labelleft=False)  # shared spatial axis

            axb = axs[1, col]
            axb.plot(x1, Ps[col], lw=1.2)
            axb.set_xlabel(r"$x_1$")
            axb.set_xlim(X_MIN, X_MAX)

            cb = fig.colorbar(im, cax=cax)
            if is_diff:
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))  # force ×10^n factored out
                cb.ax.yaxis.set_major_formatter(fmt)
                cb.ax.yaxis.set_offset_position('left')
                cb.ax.tick_params(labelsize=9)

            if is_diff:
                axb.axhline(0.0, color='0.7', lw=0.8)
                d = max_dP * 1.05 if max_dP > 0 else 1e-30
                axb.set_ylim(-d, d)
                axb.yaxis.set_label_position('right')
                axb.yaxis.tick_right()
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))
                axb.yaxis.set_major_formatter(fmt)
                axb.set_ylabel(r"$\Delta P(x_1)$")
            # if is_diff:  # old
            #     axb.axhline(0.0, color='0.7', lw=0.8)
            #     d = max_dP * 1.05 if max_dP > 0 else 1e-30
            #     axb.set_ylim(-d, d)
            #     axb.set_ylabel(r"$\Delta P(x_1)$")  # own scale, ticks visible
            else:
                axb.set_ylim(0, pmax * 1.05)
                if col == 0:
                    axb.set_ylabel(r"$P(x_1)$")
                else:
                    axb.tick_params(axis='y', labelleft=False)

        plt.tight_layout()
        for dpi in cfg.fig_dpis:
            plt.savefig(f"{title}_{dpi}dpi.png", dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def plot_wf_2x1e_and_2e_PRA_v5_withPx1_onlyColors(
            psi_left, psi_right, *, cfg, sign=1, title=None, labels=(None, None)):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        orth2e = cast(FuncArr1d2e, cfg.orth2e)
        B1 = cast(IWFuncArr, orth2e.basis_1d)
        B2 = cast(IWFuncArr, orth2e.basis2_1d)
        x1 = B1.getQuadr().getX().arr
        x2 = B2.getQuadr().getX().arr

        wL1, wL2 = cast(WFQuadrLcr, cfg.wLcr1), cast(WFQuadrLcr, cfg.wLcr2)
        sqrtCr1, sqrtCr2 = wL1.getSqrtCR().arr, wL2.getSqrtCR().arr
        cr2w1, cr2w2 = wL1.getWithCR2().arr, wL2.getWithCR2().arr
        cr2_1 = wL1.getCR2().arr
        extent = [x1[0], x1[-1], x2[0], x2[-1]]
        i, j = len(x1) // 4, 2 * len(x2) // 4  # sign-fix sample point (as v5)

        def prep(psi):
            if psi[i, j] < 0.:
                psi = -psi * sign
            P_x1 = psi ** 2 @ cr2w2
            norm = np.sum(cr2w1 * P_x1)
            assert abs(norm - 1) < 1e-8, f"norm error: {norm}"
            psi_r1r2 = sqrtCr1[:, None] ** 2 * psi * sqrtCr2[None, :] ** 2
            P_r1 = cr2_1 * P_x1
            return psi_r1r2, P_r1

        cols = [prep(psi_left), prep(psi_right)]
        vmax = max(np.max(np.abs(g)) for g, _ in cols)  # shared colour scale
        pmax = max(np.max(P) for _, P in cols)  # shared P(x1) scale

        fig, axs = plt.subplots(2, 2, figsize=(13.2, 10.4),
                                gridspec_kw={'height_ratios': [3, 1],
                                             'hspace': 0.05, 'wspace': 0.10})

        X_MIN, X_MAX = cfg.plot_x_min, 4
        for col, (psi_r1r2, P_r1) in enumerate(cols):
            ax = axs[0, col]
            im = ax.imshow(psi_r1r2.T, origin="lower", extent=extent,
                           cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="equal")
            # cax = inset_axes(ax, width="4%", height="40%", loc='upper right',
            #                  bbox_to_anchor=(-0.2, -0.05, 1, 1),
            #                  bbox_transform=ax.transAxes, borderpad=0)
            cax = inset_axes(
                ax,
                width="4%",
                height="40%",
                loc="lower left",
                bbox_to_anchor=(0.03, 0.05, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0
            )

            fig.colorbar(im, cax=cax)
            ax.set_xlim(X_MIN, X_MAX);
            ax.set_ylim(X_MIN, X_MAX)
            ax.tick_params(axis='x', labelbottom=False)
            if labels[col]:
                ax.set_title(labels[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(r"$x_2$")
            else:
                ax.tick_params(axis='y', labelleft=False)

            axb = axs[1, col]
            axb.plot(x1, P_r1, lw=1.2)
            axb.set_xlabel(r"$x_1$")
            axb.set_xlim(X_MIN, X_MAX);
            axb.set_ylim(0, pmax * 1.05)
            if col == 0:
                axb.set_ylabel(r"$P(x_1)$")
            else:
                axb.tick_params(axis='y', labelleft=False)

        plt.tight_layout()
        for dpi in cfg.fig_dpis:
            plt.savefig(f"{title}_{dpi}dpi.png", dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close()


    @staticmethod
    def plot_wf_2e_PRA_v5_withPx1_onlyColurs(psi2e_grid, *, cfg, sign=1, title=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.patches as mpatches
        import matplotlib as mpl
        mpl.rcParams['hatch.linewidth'] = 1.0  # or larger

        orth2e = cast(FuncArr1d2e, cfg.orth2e)
        B1 = cast(IWFuncArr, orth2e.basis_1d)
        B2 = cast(IWFuncArr, orth2e.basis2_1d)
        x1 = B1.getQuadr().getX().arr
        x2 = B2.getQuadr().getX().arr

        # Fix sign
        i = len(x1) // 4
        j = 2 * len(x2) // 4
        if psi2e_grid[i, j] < 0.:
            psi2e_grid = -psi2e_grid * sign

        # Marginal P(x1)
        wL1, wL2 = cast(WFQuadrLcr, cfg.wLcr1), cast(WFQuadrLcr, cfg.wLcr2)
        sqrtCr1, sqrtCr2 = wL1.getSqrtCR().arr, wL2.getSqrtCR().arr
        cr2w2 = wL2.getWithCR2().arr
        cr2w1 = wL1.getWithCR2().arr
        P_x1 = psi2e_grid ** 2 @ cr2w2
        norm = np.sum(wL1.getWithCR2().arr * P_x1)
        assert abs(norm - 1) < 1e-8, f"norm error: {norm}"

        psi2e_grid_r1r2 = sqrtCr1[:, None] ** 2 * psi2e_grid * sqrtCr2[None, :] ** 2
        P_r1 = wL1.getCR2().arr * P_x1

        vmax = np.max(np.abs(psi2e_grid_r1r2))
        extent = [x1[0], x1[-1], x2[0], x2[-1]]

        fig, axs = plt.subplots(2, 1,
                                # figsize=(3.3, 5.2),
                                figsize=(6.6, 10.4),
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

        # Top panel: color heatmap
        im = axs[0].imshow(
            psi2e_grid_r1r2.T,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal"
        )
        # Inset colorbar
        # cax = inset_axes(axs[0], width="4%", height="40%", loc='upper right',
        #                  bbox_to_anchor=(-0.2, -0.05, 1, 1),
        #                  bbox_transform=axs[0].transAxes, borderpad=0)
        cax = inset_axes(
            axs[0],
            width="4%",
            height="40%",
            loc="lower left",
            bbox_to_anchor=(0.03, 0.05, 1, 1),
            bbox_transform=axs[0].transAxes,
            borderpad=0
        )
        fig.colorbar(im, cax=cax)

        axs[0].set_ylabel(r"$x_2$")
        axs[0].tick_params(axis='x', labelbottom=False)
        # axs[0].set_xlim(X_MIN := -2, X_MAX := 4)
        axs[0].set_xlim(X_MIN := cfg.plot_x_min, X_MAX := 4)
        axs[0].set_ylim(X_MIN, X_MAX)

        # Bottom: Marginal
        axs[1].plot(x1, P_r1, lw=1.2)
        axs[1].set_xlabel(r"$x_1$")
        axs[1].set_ylabel(r"$P(x_1)$")
        axs[1].set_xlim(X_MIN, X_MAX)
        axs[1].set_ylim(bottom=0)

        plt.tight_layout()
        # for dpi in [300, 600]:
        for dpi in cfg.fig_dpis:
            plt.savefig(f"{title}_{dpi}dpi.png", dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close()



    @staticmethod
    def plot_wf_2e_PRA_v2_OK(psi2e_grid, *, cfg, zoom=None, title=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        orth2e = cast(FuncArr1d2e, cfg.orth2e)
        B1 = cast(IWFuncArr, orth2e.basis_1d)
        B2 = cast(IWFuncArr, orth2e.basis2_1d)
        x1 = B1.getQuadr().getX().arr
        x2 = B2.getQuadr().getX().arr
        dbg([psi2e_grid, x1, x2])
        # Example: plotting the wavefunction
        fig, ax = plt.subplots(figsize=(4, 4))  # adjust for PRA column

        im = ax.imshow(
            psi2e_grid,  # your 2D data
            origin="lower",
            extent=[x1[0], x1[-1], x2[0], x2[-1]],
            cmap="RdBu_r",
            aspect="equal"
        )
        # # Create inset colorbar inside the axes
        # cax = inset_axes(ax,
        #                  width="5%",  # width relative to parent axes
        #                  height="40%",  # height relative to parent axes
        #                  loc='upper right',
        #                  borderpad=1)
        cax = inset_axes(
            ax,
            width="4%",  # thinner
            height="40%",  # relative height
            loc='upper right',
            #  [left, bottom, width, height],
            # bbox_to_anchor=(-0.05, 0.05, 1, 1),  # tweak x and y shift here
            bbox_to_anchor=(-0.15, -0.05, 1, 1),  # tweak x and y shift here
            bbox_transform=ax.transAxes,
            borderpad=0  # minimal padding
        )
        cbar = fig.colorbar(im, cax=cax)
        # cbar.set_label(r"$\Psi(x_1, x_2)$")

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        # ax.set_title(r"$\psi_{2 \times 1e}$")

        plt.tight_layout()
        # plt.savefig("psi2x1e_inset_colorbar.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{title}_low_res.png", bbox_inches='tight')
        plt.savefig(f"{title}_300dpi.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


