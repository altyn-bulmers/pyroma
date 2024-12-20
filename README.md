# pyROMA

Representation and Quantification of Module Activity in single cell and bulk transcriptomics

### Create the conda environment

`conda env create -f environment.yml`

### Install using pip

```sh
pip install roma-analysis
```

### Run roma on your data

```py
import pyroma`
roma = pyroma.ROMA()
roma.adata = adata
roma.gmt = 'h.all.v2023.2.Hs.symbols.gmt'
roma.compute()
```

# Clone main repo with submodules
`git clone --recurse-submodules git@github.com:yourusername/pyroma.git`


## Reproducibility
Jupyter notebooks demonstrating the usage and reproducibility of results can be found in our companion repository: [pyroma_reproducibility](https://github.com/altyn-bulmers/pyroma_reproducibility)

Datasets are downloaded from [here](https://github.com/sysbio-curie/rRoma_comp) and exported as .tsv files to `datasets` folder


