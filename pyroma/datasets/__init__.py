"""Dataset loading utilities for pyroma."""

from pathlib import Path
import scanpy as sc
import anndata as ad


def _get_data_path(filename: str) -> str:
    """Get path to a dataset file with backward compatibility."""
    try:
        # Python 3.9+ style
        from importlib.resources import files
        path = files(__package__).joinpath(filename)
        return str(path)
    except (ImportError, AttributeError, TypeError):
        # Python 3.7-3.8 fallback
        try:
            import pkg_resources
            return pkg_resources.resource_filename(__package__, filename)
        except Exception:
            # Last resort: relative to this file
            return str(Path(__file__).parent / filename)


def pbmc3k() -> ad.AnnData:
    """
    10X peripheral blood mononuclear cells (PBMCs).

    Loads single-cell RNA-seq data of peripheral blood mononuclear
    cells (PBMCs) from a healthy donor.

    Returns
    -------
    adata : AnnData
        Annotated data matrix.

    Example
    -------
    >>> import pyroma
    >>> adata = pyroma.datasets.pbmc3k()
    >>> print(adata)
    """
    data_path = _get_data_path("rna_10xpmbc3k.h5ad")
    adata = sc.read_h5ad(data_path)
    return adata


def pbmc_ifnb() -> ad.AnnData:
    """
    10X peripheral blood mononuclear cells (PBMCs) + IFNb stimulation.
    
    Data from Kang et al. 2018.

    Loads single-cell RNA-seq data of peripheral blood mononuclear
    cells (PBMCs) from healthy and stimulated samples.

    Returns
    -------
    adata : AnnData
        Annotated data matrix.

    Example
    -------
    >>> import pyroma
    >>> adata = pyroma.datasets.pbmc_ifnb()
    >>> print(adata)
    """
    data_path = _get_data_path("kang_tutorial.h5ad")
    adata = sc.read_h5ad(data_path)
    return adata


__all__ = ['pbmc3k', 'pbmc_ifnb']