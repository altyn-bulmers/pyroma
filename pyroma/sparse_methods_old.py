import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

def spdot(A, B):
    """Safe dot product for combinations of sparse/dense matrices."""
    if sp.issparse(A) and sp.issparse(B):
        return A * B
    elif sp.issparse(A) and not sp.issparse(B):
        return (A * B).view(type=B.__class__)
    elif not sp.issparse(A) and sp.issparse(B):
        return (B.T * A.T).T.view(type=A.__class__)
    else:
        return np.dot(A, B)

def spadd(a, b):
    """Add b to a in-place. Either can be sparse."""
    if sp.issparse(b):
        # Only works if a is ndarray and b is sparse matrix (column vector)
        a[b.nonzero()[0]] += b.data.reshape(-1, 1)
    elif sp.issparse(a):
        a = a + b
    else:
        np.add(a, b, a)
    return a

class SparseLinearOperator(la.LinearOperator, sp.spmatrix):
    """A LinearOperator that pretends to be a scipy.sparse matrix."""
    def __init__(self, shape, dtype=None):
        sp.spmatrix.__init__(self)
        la.LinearOperator.__init__(self, shape=shape, dtype=dtype)
    def matvec(self, x):
        raise NotImplementedError()
    def matmat(self, X):
        if X.ndim != 2:
            raise ValueError('expected rank-2 ndarray or matrix')
        M, N = self.shape
        if X.shape[0] != N:
            raise ValueError('dimension mismatch')
        Y = self._matmat(X)
        if isinstance(Y, np.matrix):
            Y = np.asmatrix(Y)
        return Y
    def _matmat(self, X):
        # Fallback: apply matvec to each column
        if sp.issparse(X):
            return np.hstack([self.matvec(X[:, i]) for i in range(X.shape[1])])
        else:
            return np.hstack([self.matvec(col.reshape(-1, 1)) for col in X.T])
    def __mul__(self, x):
        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            return self.matvec(x)
        elif x.ndim == 2:
            return self.matmat(x)
        else:
            raise ValueError('expected rank-1 or rank-2 array or matrix')
    def todense(self):
        print("Warning: Converting to dense! This may use a lot of memory.")
        return self * np.eye(self.shape[1], dtype=float)
    def __getattr__(self, attr):
        # Convenience properties
        if attr == 'A':
            return self.toarray()
        elif attr == 'T':
            return self.transpose()
        elif attr == 'H':
            return self.getH()
        elif attr == 'real':
            return self._real()
        elif attr == 'imag':
            return self._imag()
        elif attr == 'size':
            return self.getnnz()
        else:
            try:
                return self.__dict__[attr]
            except KeyError:
                raise AttributeError(attr)

class Centered(SparseLinearOperator):
    """
    A linear operator for a sparse matrix X that applies centering over a given axis.
    (axis=0 for column-centering, axis=1 for row-centering)
    """
    def __init__(self, X, axis=0, Xt=None, center=None):
        self.X = X
        self.axis = axis
        self.Xt = Xt
        if center is None:
            self.center = np.asarray(self.X.mean(axis=axis)).reshape(-1)
        else:
            self.center = center
        def matvec(v):
            if v.ndim == 1:
                v = v.reshape(-1, 1)
            if self.axis == 0:
                thisV = spdot(self.X, v)
                cv = spdot(self.center, v)
                return (thisV - cv).A1 if sp.issparse(thisV) else (thisV - cv).reshape(-1)
            elif self.axis == 1:
                Xt = self._get_Xt()
                thisV = spdot(Xt, v)
                cv = v.sum() * self.center
                return (thisV - cv.reshape(-1, 1)).A1 if sp.issparse(thisV) else (thisV - cv.reshape(-1, 1)).reshape(-1)
            else:
                raise ValueError("axis must be 0 or 1")
        # Determine output shape
        if axis == 0:
            shape = X.shape
        elif axis == 1:
            shape = (X.shape[1], X.shape[0])
        else:
            raise ValueError("axis must be 0 or 1")
        super().__init__(shape=shape, matvec=matvec, dtype=X.dtype)

    def _get_Xt(self):
        if self.Xt is None:
            self.Xt = self.X.T.tocsr()
        return self.Xt
    def transpose(self):
        return Centered(self.X, axis=abs(-1 + self.axis), Xt=self.Xt)

def sparse_centering_operator(X):
    """
    Returns a memory-efficient linear operator that is X,
    centered by rows then by columns.
    Use as: sparse_centering_operator(X) @ v
    """
    return Centered(Centered(X, axis=1), axis=0)

# --------- Usage Example ------------
if __name__ == "__main__":
    # Example sparse matrix
    X = sp.random(4, 5, density=0.4, format='csr', random_state=42)
    print("Original sparse matrix X:\n", X.toarray())

    # Build double-centered operator
    X_centered = sparse_centering_operator(X)

    # Multiply by a vector (for example)
    v = np.random.rand(X.shape[1])
    result = X_centered @ v
    print("Result of (X double-centered) @ v:\n", result)

    # If you must materialize the double-centered dense matrix (use only if memory is not a concern):
    # WARNING: this may consume a lot of memory!
    # dense_X_centered = X_centered @ np.eye(X.shape[1])
    # print("Dense double-centered matrix:\n", dense_X_centered)
