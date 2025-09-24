# pyroma/datasets/_load_datasets.py
from importlib.resources import files
import scanpy as sc
import anndata as ad


def pbmc3k(

) -> ad.AnnData:
    """
    10X peripheral blood mononuclear cells (PBMCs).

    Loads single-cell RNA-seq data of peripheral blood mononuclear
    cells (PBMCs) from a healthy donor. 

    Returns
    -------
    AnnData object.

    Example
    -------
    .. code-block:: python

        import pyroma

        adata = pyroma.datasets.pbmc3k()
        adata
    """
    data_path = files(__package__).joinpath("rna_10xpmbc3k.h5ad")
    adata = sc.read_h5ad(str(data_path))

    return adata


def pbmc_ifnb(

) -> ad.AnnData:
    """
    10X peripheral blood mononuclear cells (PBMCs) + IFNb stimulation, Kang et al 2018.

    Loads single-cell RNA-seq data of peripheral blood mononuclear
    cells (PBMCs) from a healthy and stimulated samples. 

    Returns
    -------
    AnnData object.

    Example
    -------
    .. code-block:: python

        import pyroma

        adata = pyroma.datasets.pbmc_ifnb()
        adata
    """
    data_path = files(__package__).joinpath("kang_tutorial.h5ad")
    adata = sc.read_h5ad(str(data_path))

    return adata