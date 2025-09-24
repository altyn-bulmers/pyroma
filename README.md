# pyROMA

**Representation and Quantification of Module Activity in single-cell and bulk transcriptomics**

---

![Methods Workflow](Methods.png)

---

## Features

* Compute module activity scores for single-cell and bulk RNA-seq data
* Seamless integration with AnnData objects (`scanpy`)
* Support for GMT pathway files (e.g., MSigDB hallmark gene sets)
* Lightweight and easy to extend

---

## Installation

### Create Conda environment

```bash
conda env create -f environment.yml
conda activate pyroma
```

### Install using Pip

```bash
pip install roma-analysis
```

### Or install directly from source

```bash
git clone https://github.com/altyn-bulmers/pyroma.git
cd pyroma
pip install -e .
```

---

## Quick Start

```python
import pyroma

# Initialize ROMA
roma = pyroma.ROMA()

# Assign your AnnData object and GMT file
roma.adata = adata  # AnnData from scanpy

hallmarks_gmt_path = pyroma.genesets.use_hallmarks()
roma.gmt   = hallmarks_gmt_path

# Compute module activity scores
roma.compute()

# Inspect results
roma.adata.uns['ROMA_active_modules']
```

---

## Tutorials

Comprehensive tutorials are available on Read the Docs:

- [**Getting started**](https://pyroma.readthedocs.io/en/latest/tutorials/getting_started.html)
- [**Advanced usage**](https://pyroma.readthedocs.io/en/latest/tutorials/advanced_usage.html)
- [**Advanced usage (cell-type setting)**](https://pyroma.readthedocs.io/en/latest/tutorials/advanced_usage_ct_setting.html)
---

## Clone with Submodules

If you need submodule content (e.g., additional scripts or data):

```bash
git clone --recurse-submodules git@github.com:altyn-bulmers/pyroma.git
```

---

## Reproducibility

Companion notebooks and detailed workflows are maintained in a dedicated repository (will be available soon):

* [pyroma\_reproducibility](https://github.com/altyn-bulmers/pyroma_reproducibility)

Core datasets are sourced from [rRoma\_comp](https://github.com/sysbio-curie/rRoma_comp) and included as TSV files in the `datasets/` directory.

---

## References

1. Martignetti L, Calzone L, Bonnet E, Barillot E, Zinovyev A (2016). [ROMA: Representation and Quantification of Module Activity from Target Expression Data](https://doi.org/10.3389/fgene.2016.00018). *Front. Genet.* 7:18.
2. Najm M, Cornet M, Albergante L, et al. (2024). [Representation and quantification of module activity from omics data with rROMA](https://doi.org/10.1038/s41540-024-00331-x). *npj Syst Biol Appl.* 10:8.

