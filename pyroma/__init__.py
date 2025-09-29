# pyroma/__init__.py
"""
PyROMA: Python implementation of Representation Of Module Activity for single-cell RNA-seq data.
"""

from .roma import ROMA, GeneSetResult, color
from .utils import integrate_projection_results

# submodules 
from . import datasets
from . import genesets
from . import plotting
from . import utils
from . import sparse_methods

# submodule functions 
from .datasets import pbmc3k, pbmc_ifnb
from .genesets import use_hallmarks, use_reactome, use_progeny

__version__ = '0.2.3'

__all__ = [
    # Core classes
    'ROMA', 
    'GeneSetResult', 
    'color',
    
    # Submodules
    'datasets',
    'genesets',
    'plotting',
    'utils',
    'sparse_methods',
    
    # Functions
    'integrate_projection_results',
    'pbmc3k',
    'pbmc_ifnb',
    'use_hallmarks',
    'use_reactome',
    'use_progeny',
]