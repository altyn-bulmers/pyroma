import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

def spdot(A, B):
    """Dot that works for sparse/dense combos."""
    if sp.issparse(A) and sp.issparse(B):
        return A @ B
    elif sp.issparse(A):
        return (A @ B).view(type=B.__class__)
    elif sp.issparse(B):
        return (B.T @ A.T).T.view(type=A.__class__)
    else:
        return A @ B

class DoubleCentered(la.LinearOperator):
    """
    A LinearOperator for X with both row‐ and column‐means removed:
       out_ij = x_ij - r_i - c_j + g
    where r_i = row mean of row i,
          c_j = column mean of column j,
          g   = grand mean.
    """
    def __init__(self, X):
        # X must be a scipy.sparse matrix
        if not sp.issparse(X):
            raise ValueError("DoubleCentered only accepts sparse matrices")
        self.X = X.tocsr()  # ensure CSR for fast row operations

        # 1) compute row means (shape: n_rows,)
        self.row_means = np.asarray(self.X.mean(axis=1)).flatten()
        # 2) compute column means (shape: n_cols,)
        self.col_means = np.asarray(self.X.mean(axis=0)).flatten()
        # 3) compute grand mean
        self.grand_mean = self.row_means.mean()

        # set up LinearOperator: signature is __init__(dtype, shape)
        super().__init__(dtype=X.dtype, shape=X.shape)

    def matvec(self, v):
        """
        Compute (X - r - c + g) @ v  in O(nnz) + O(n).
        v may be 1-d or a column vector.
        """
        # ensure column-vector
        if v.ndim == 1:
            v = v.reshape(-1, 1)

        # 1) X @ v
        Xv = spdot(self.X, v)             # shape (n_rows,1)

        # 2) subtract row_means * sum(v)
        sv = v.sum()                      # scalar
        out = Xv - self.row_means.reshape(-1,1) * sv

        # 3) subtract cᵀ·v (a scalar) from every row
        cv = float(self.col_means.dot(v.flatten()))
        out = out - cv

        # 4) add grand_mean * sum(v) back
        out = out + self.grand_mean * sv

        # flatten and return
        return out.ravel()

    def rmatvec(self, v):
        # allow using eigen/svd routines that call rmatvec
        # note: (A^T v) = (DoubleCentered(X)).T @ v
        return (self.T @ v)

    def transpose(self):
        # the transpose of (X - r - c + g) is
        # (X^T - c - r + g), so just swap row/col means
        op = DoubleCentered(self.X)
        op.row_means, op.col_means = self.col_means, self.row_means
        # grand mean stays the same
        return op

def sparse_centering_operator(X):
    """
    Return a memory-efficient LinearOperator which is X
    double-centered (rows then columns):
        C = DoubleCentered(X)
    Use like:   y = C @ some_vector
    """
    return DoubleCentered(X)
