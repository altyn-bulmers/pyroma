# pyROMA

[![PyPI version](https://badge.fury.io/py/roma-analysis.svg)](https://badge.fury.io/py/roma-analysis)

###  Representation and Quantification of Module Activity in single cell and bulk transcriptomics[1, 2].

## Create the conda environment

```sh
conda env create -f environment.yml
conda activate pyroma
```


## Clone main repo with submodules
```sh
git clone git@github.com:altyn-bulmers/pyroma.git pyroma-tutorial
cd pyroma-tutorial
```

## Run roma on your data

```py
import pyroma
roma = pyroma.ROMA()
roma.adata = adata
roma.gmt = 'h.all.v2023.2.Hs.symbols.gmt'
roma.compute()
```


## Reproducibility
Jupyter notebooks demonstrating the usage and reproducibility of results can be found in our companion repository: [pyroma_reproducibility](https://github.com/altyn-bulmers/pyroma_reproducibility)

Datasets are downloaded from [here](https://github.com/sysbio-curie/rRoma_comp) and exported as .tsv files to `datasets` folder

## References

Martignetti L, Calzone L, Bonnet E, Barillot E and Zinovyev A (2016) ROMA: Representation and Quantification of Module Activity from Target Expression Data. Front. Genet. 7:18. doi: 10.3389/fgene.2016.00018

Najm, M., Cornet, M., Albergante, L. et al. Representation and quantification of module activity from omics data with rROMA. npj Syst Biol Appl 10, 8 (2024). https://doi.org/10.1038/s41540-024-00331-x


