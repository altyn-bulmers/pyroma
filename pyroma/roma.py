"""
PyROMA: Python implementation of Representation Of Module Activity for single-cell RNA-seq data.

This module implements the ROMA algorithm for quantifying gene set activity
in single-cell transcriptomics data using principal component analysis.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import time
import os
import warnings
import pickle
from scipy import stats, sparse
from scipy.stats import wilcoxon, fisher_exact
from statsmodels.stats.multitest import multipletests
from scipy.stats import false_discovery_control as benj_hoch
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA
from sklearn.model_selection import LeaveOneOut
from joblib import Parallel, delayed
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Any
from types import SimpleNamespace
import multiprocessing
from .sparse_methods import *

# Suppress warnings
warnings.filterwarnings("ignore")

# Color codes for terminal output
class color:
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'


class GeneSetResult:
    """
    Container for gene set analysis results.
    
    Attributes
    ----------
    subset : sc.AnnData
        Subset of data for this gene set
    subsetlist : np.ndarray
        Array of gene names in the gene set
    outliers : List[int]
        Indices of outlier genes
    nullgenesetsize : int
        Size of null gene set for comparison
    svd : Union[TruncatedSVD, PCA, IncrementalPCA]
        Fitted decomposition object
    X : np.ndarray
        Processed data matrix
    raw_X_subset : np.ndarray
        Raw data subset for PC sign correction
    projections_1 : np.ndarray
        Projections onto first principal component
    projections_2 : np.ndarray
        Projections onto second principal component
    nulll1 : np.ndarray
        Null distribution of L1 values
    null_median_exp : np.ndarray
        Null distribution of median expression
    null_projections : np.ndarray
        Null projections for both components
    q_value : float
        Adjusted p-value for L1 statistic
    med_exp_q_value : float
        Adjusted p-value for median expression
    non_adj_p : float
        Raw p-value for L1 statistic
    non_adj_med_exp_p : float
        Raw p-value for median expression
    test_l1 : float
        L1 test statistic
    test_median_exp : float
        Median expression test statistic
    """
    
    def __init__(self, 
                 subset: sc.AnnData, 
                 subsetlist: np.ndarray, 
                 outliers: List[int], 
                 nullgenesetsize: int, 
                 svd: Union[TruncatedSVD, PCA, IncrementalPCA]) -> None:
        """Initialize GeneSetResult with basic attributes."""
        self.subset = subset
        self.subsetlist = subsetlist
        self.outliers = outliers
        self.nullgenesetsize = nullgenesetsize
        self.svd = svd
        self.X: Optional[np.ndarray] = None
        self.raw_X_subset: Optional[np.ndarray] = None
        self.projections_1: Optional[np.ndarray] = None
        self.projections_2: Optional[np.ndarray] = None
        self.nulll1: Optional[np.ndarray] = None
        self.null_median_exp: Optional[np.ndarray] = None
        self.null_projections: Optional[np.ndarray] = None
        self.q_value: Optional[float] = None
        self.med_exp_q_value: Optional[float] = None
        self.non_adj_p: Optional[float] = None
        self.non_adj_med_exp_p: Optional[float] = None
        self.test_l1: Optional[float] = None
        self.test_median_exp: Optional[float] = None
        self.custom_name: str = ""
    
    def __repr__(self) -> str:
        return self.custom_name

    def __str__(self) -> str:
        return self.custom_name


class ROMA:
    """
    ROMA (Representation Of Module Activity) implementation for single-cell RNA-seq data.
    
    This class implements the ROMA algorithm for quantifying gene set activity
    in single-cell transcriptomics data using principal component analysis.
    
    Attributes
    ----------
    adata : Optional[sc.AnnData]
        Annotated data matrix
    gmt : Optional[str]
        Path to GMT file containing gene sets
    genesets : Dict[str, np.ndarray]
        Dictionary mapping gene set names to gene arrays
    results : Dict[str, GeneSetResult]
        Results for each analyzed gene set
    null_distributions : Dict[int, Tuple[np.ndarray, np.ndarray]]
        Cached null distributions by gene set size
    approx_int : int
        Granularity of null gene set size approximation
    min_n_genes : int
        Minimum number of genes required in a gene set
    q_L1_threshold : float
        Q-value threshold for L1 significance
    q_Med_Exp_threshold : float
        Q-value threshold for median expression significance
    """
    
    def __init__(self) -> None:
        """Initialize ROMA analyzer with default parameters."""
        self.adata: Optional[sc.AnnData] = None
        self.gmt: Optional[str] = None
        self.genesets: Dict[str, np.ndarray] = {}
        self.idx: List[str] = []
        self.approx_int: int = 20
        self.min_n_genes: int = 10
        self.nullgenesetsize: Optional[int] = None
        self.subset: Optional[sc.AnnData] = None
        self.subsetlist: Optional[np.ndarray] = None
        self.outliers: List[int] = []
        self.loocv_scores: Dict[str, float] = {}
        self.global_gene_counts: Dict[str, int] = {}
        self.global_outlier_counts: Dict[str, int] = {}
        self.svd: Optional[Union[TruncatedSVD, PCA, IncrementalPCA]] = None
        self.X: Optional[np.ndarray] = None
        self.raw_X_subset: Optional[np.ndarray] = None
        self.nulll1: List[float] = []
        self.test_l1: Optional[float] = None
        self.p_value: Optional[float] = None
        self.test_median_exp: Optional[float] = None
        self.med_exp_p_value: Optional[float] = None
        self.projections_1: Optional[np.ndarray] = None
        self.projections_2: Optional[np.ndarray] = None
        self.results: Dict[str, GeneSetResult] = {}
        self.null_distributions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.unprocessed_genesets: List[str] = []
        
        # Multiprocessing manager
        self.parallel_results = None
        self._manager = None
        
        # Display name
        self.custom_name: str = f"{color.BOLD}{color.GREEN}scROMA{color.END}"
        
        # Thresholds
        self.q_L1_threshold: float = 0.05
        self.q_Med_Exp_threshold: float = 0.05
        
        # PC sign correction parameters
        self.gene_weights: Dict[str, np.ndarray] = {}
        self.pc_sign_mode: str = 'PreferActivation'
        self.pc_sign_thr: float = 0.90
        self.def_wei: float = 1.0
        self.cor_method: str = 'pearson'
        self.correct_pc_sign: int = 1
        self.gene_signs: Dict[str, Dict[str, int]] = {}
        self.extreme_percent: float = 0.1
        
        # Plotting module
        from .plotting import plotting as pl
        self.pl = pl()

    def __repr__(self) -> str:
        return self.custom_name
    
    def __str__(self) -> str:
        return self.custom_name
    
    def _get_manager(self):
        """Initialization of multiprocessing manager."""
        if self._manager is None:
            self._manager = multiprocessing.Manager()
            self.parallel_results = self._manager.dict()
        return self.parallel_results

    def read_gmt_to_dict(self, gmt_path: str) -> Dict[str, np.ndarray]:
        """
        Read GMT file and convert to dictionary of gene sets.
        
        Parameters
        ----------
        gmt_path : str
            Path to GMT format file
        
        Returns
        -------
        genesets : Dict[str, np.ndarray]
            Dictionary mapping gene set names to gene arrays
        
        Raises
        ------
        FileNotFoundError
            If GMT file doesn't exist
        ValueError
            If GMT file format is invalid
        """
        if not os.path.exists(gmt_path):
            raise FileNotFoundError(f"GMT file not found: {gmt_path}")
        
        genesets = {}
        
        try:
            with open(gmt_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    parts = line.rstrip('\n').split('\t')
                    
                    if len(parts) < 3:
                        raise ValueError(f"Invalid GMT format at line {line_num}: "
                                       f"expected at least 3 tab-separated fields")
                    
                    name = parts[0]
                    # Skip description field (parts[1])
                    genes = [g for g in parts[2:] if g]
                    
                    if name in genesets:
                        warnings.warn(f"Duplicate gene set name '{name}' found, "
                                    f"overwriting previous definition")
                    
                    genesets[name] = np.array(genes)
        
        except Exception as e:
            raise ValueError(f"Error reading GMT file: {e}")
        
        self.genesets = genesets
        self.gmt = gmt_path
        return genesets

    def indexing(self, adata: sc.AnnData) -> None:
        """
        Create index of unique gene names from AnnData object.
        
        Parameters
        ----------
        adata : sc.AnnData
            Annotated data matrix
        """
        self.idx = list(set(adata.var.index.tolist()))

    def subsetting(self, adata: sc.AnnData, geneset: np.ndarray, verbose: int = 0) -> Tuple[sc.AnnData, np.ndarray]:
        """
        Create subset of data for specific gene set.
        
        Parameters
        ----------
        adata : sc.AnnData
            Full annotated data matrix
        geneset : np.ndarray
            Array of gene names to subset
        verbose : int, default=0
            Verbosity level
        
        Returns
        -------
        subset : sc.AnnData
            Subset of data containing only genes in gene set
        subsetlist : np.ndarray
            Array of gene names that were found in the data
        """
        if verbose:
            print(' '.join(geneset))
        
        if not self.idx:
            raise ValueError('No adata index detected. Run indexing() first.')
        
        # Efficient intersection using numpy
        subsetlist = geneset[np.isin(geneset, self.idx)]
        subset = adata[:, subsetlist]
        
        self.subset = subset
        self.subsetlist = subsetlist
        
        return subset, subsetlist

    def double_mean_center_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Double-center a matrix by removing row and column means.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix to center
        
        Returns
        -------
        centered_matrix : np.ndarray
            Double-centered matrix
        """
        overall_mean = np.mean(matrix)
        row_means = np.mean(matrix, axis=1, keepdims=True)
        col_means = np.mean(matrix, axis=0, keepdims=True)
        
        centered_matrix = matrix - row_means - col_means + overall_mean
        
        return centered_matrix

    @lru_cache(maxsize=128)
    def _get_null_distribution(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get or compute null distribution for given size with caching.
        
        Parameters
        ----------
        size : int
            Size of gene set for null distribution
        
        Returns
        -------
        null_l1 : np.ndarray
            Null distribution of L1 values
        null_median : np.ndarray
            Null distribution of median expression values
        """
        if size in self.null_distributions:
            return self.null_distributions[size]
        
        # If not cached, will be computed elsewhere
        return None, None

    def loocv(self, subset: sc.AnnData, verbose: int = 0, for_randomset: bool = False) -> List[int]:
        """
        Perform leave-one-out cross-validation for outlier detection.
        
        Uses vectorized operations to efficiently compute L1 scores
        with each gene removed.
        
        Parameters
        ----------
        subset : sc.AnnData
            Subset of data for current gene set
        verbose : int, default=0
            Verbosity level
        for_randomset : bool, default=False
            Whether this is for random set computation
        
        Returns
        -------
        outliers : List[int]
            Indices of detected outlier genes
        """
        #X = self._prepare_data(subset.X.T)
        X = self.subset.X.T
        n_features, n_samples = X.shape
        X = np.asarray(X)
        if n_samples < 2:
            if verbose:
                print(f"Cannot perform LOOCV with {n_samples} samples.")
            return []

        # Vectorized computation of L1 scores
        l1scores = np.zeros(n_features)
        svd = TruncatedSVD(n_components=1, algorithm='randomized', n_oversamples=2, random_state=42)

        # Use efficient leave-one-out
        loo = LeaveOneOut()
        for i, (train_index, _) in enumerate(loo.split(X)):
            svd.fit(X[train_index])
            l1scores[i] = svd.explained_variance_ratio_[0]
        
        # Store scores for each gene
        loocv_scores = {gene: score for gene, score in zip(subset.var_names, l1scores)}
        self.loocv_scores = loocv_scores

        # Detect outliers using z-scores
        if len(l1scores) > 1:
            mean_score = np.mean(l1scores)
            std_score = np.std(l1scores)
            if std_score > 0:
                z_scores = np.abs((l1scores - mean_score) / std_score)
                outliers = np.where(z_scores > 3.0)[0].tolist()
            else:
                outliers = []
        else:
            outliers = []

        if verbose:
            print(f"Number of samples: {n_samples}, Number of features: {n_features}")
            print(f"Number of outliers detected: {len(outliers)}")

        return outliers

    def fisher_outlier_filter(self, 
                            gene_outlier_counts: Dict[str, int], 
                            gene_pathway_counts: Dict[str, int],
                            outlier_fisher_thr: float = 0.05, 
                            min_gene_sets: int = 3) -> Dict[str, bool]:
        """
        Determine outlier genes using Fisher's exact test.
        
        Tests whether each gene's proportion of outlier calls is significantly
        higher than the global average.
        
        Parameters
        ----------
        gene_outlier_counts : Dict[str, int]
            Number of gene sets where each gene was flagged as outlier
        gene_pathway_counts : Dict[str, int]
            Total number of gene sets containing each gene
        outlier_fisher_thr : float, default=0.05
            P-value threshold for Fisher's test
        min_gene_sets : int, default=3
            Minimum gene sets required for outlier consideration
        
        Returns
        -------
        gene_outlier_flags : Dict[str, bool]
            Whether each gene is considered an outlier
        """
        total_outliers = sum(gene_outlier_counts.values())
        total_pathways = sum(gene_pathway_counts.values())
        
        gene_outlier_flags = {}
        
        for gene, pathways_count in gene_pathway_counts.items():
            # Skip genes in too few gene sets
            if pathways_count < min_gene_sets:
                gene_outlier_flags[gene] = False
                continue
            
            gene_outlier = gene_outlier_counts.get(gene, 0)
            
            # Build contingency table
            table = [
                [total_outliers - gene_outlier, gene_outlier],
                [total_pathways - pathways_count, pathways_count]
            ]
            
            # Perform Fisher's exact test
            _, p_value = fisher_exact(table, alternative='greater')
            
            gene_outlier_flags[gene] = (p_value < outlier_fisher_thr)
            
        return gene_outlier_flags

    def limit_outliers_per_geneset(self, 
                                 gene_set: List[str], 
                                 gene_flags: Dict[str, bool], 
                                 loocv_scores: Dict[str, float], 
                                 max_outlier_prop: float = 0.5) -> Dict[str, bool]:
        """
        Limit the proportion of genes flagged as outliers in a gene set.
        
        Parameters
        ----------
        gene_set : List[str]
            Gene names in the gene set
        gene_flags : Dict[str, bool]
            Current outlier flags for genes
        loocv_scores : Dict[str, float]
            LOOCV scores for prioritizing outliers
        max_outlier_prop : float, default=0.5
            Maximum proportion of outliers allowed
        
        Returns
        -------
        gene_flags : Dict[str, bool]
            Updated outlier flags
        """
        flagged_genes = [g for g in gene_set if gene_flags.get(g, False)]
        allowed_num = int(max_outlier_prop * len(gene_set))
        
        if len(flagged_genes) > allowed_num:
            # Sort by LOOCV score extremity
            flagged_sorted = sorted(flagged_genes, 
                                  key=lambda g: abs(loocv_scores.get(g, 0)), 
                                  reverse=True)
            keep_flagged = set(flagged_sorted[:allowed_num])
            
            # Update flags
            for g in flagged_genes:
                gene_flags[g] = (g in keep_flagged)
        
        return gene_flags

    def _prepare_data(self, X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Prepare data matrix for computation, handling sparse matrices efficiently.
        If the data resembles bulk transcriptomics, i.e. big proportion of non-zero values, 
        then the data will used as dense matrix.

        Parameters
        ----------
        X : Union[np.ndarray, sparse.spmatrix]
            Input data matrix
        
        Returns
        -------
        X_prepared : np.ndarray
            Prepared array (dense or sparse as appropriate)
        """
        # TODO: if the matrix is dense, and single cell -> to sparse
        if sparse.issparse(X):
            # TODO: consider the size of the dense matrix relative to available memory,
            # convert if sparse is more efficient 
            # Only convert to dense if necessary
            density = X.nnz / (X.shape[0] * X.shape[1])
            if density > 0.5:  # More than 50% non-zero
                return X.toarray()
            else:
                # Keep sparse for efficiency in downstream operations
                # Note: sklearn TruncatedSVD handles sparse matrices
                return X
        return np.asarray(X)
    
    def robustTruncatedSVD(self, 
                          adata: sc.AnnData, 
                          subsetlist: np.ndarray, 
                          outliers: List[int], 
                          for_randomset: bool = False, 
                          algorithm: str = 'randomized') -> Tuple[TruncatedSVD, np.ndarray]:
        # Filter outliers to only include valid indices for current gene_subset
        outliers_array = np.array(outliers)
        valid_outliers = outliers_array[outliers_array < len(subsetlist)]
        
        # Create boolean mask for genes (excluding outliers)
        mask = np.ones(len(subsetlist), dtype=bool)
        if len(valid_outliers) > 0:
            mask[valid_outliers] = False
        
        # Get the subset of genes excluding outliers
        robust_gene_subset = subsetlist[mask]
        
        if len(robust_gene_subset) < 2:
            # Return zeros if too few genes remain
            return None, None
            
        # Extract data for robust gene subset
        X = adata[:, robust_gene_subset].X#.copy()
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        X = X.T  # Transpose to have genes as rows, samples as columns
        
        # Center the data
        #X = X - X.mean(axis=1, keepdims=True)
        
        # Perform SVD
        try:
            if algorithm == 'arpack' and min(X.shape) > 2:
                svd_ = TruncatedSVD(n_components=2, algorithm='arpack')
            else:
                svd_ = TruncatedSVD(n_components=2, algorithm='randomized')
            svd_.fit(X) 
        except Exception as e:
            print(f"SVD failed: {e}")
            return None, None
            
        if not for_randomset:
            self.svd = svd_
            self.X = X

        return svd_, X


    def robustPCA(self, 
                  adata: sc.AnnData, 
                  subsetlist: np.ndarray, 
                  outliers: List[int], 
                  for_randomset: bool = False, 
                  algorithm: str = 'auto') -> Tuple[PCA, np.ndarray]:
        """
        Perform robust PCA excluding outlier genes.
        
        Parameters
        ----------
        adata : sc.AnnData
            Annotated data matrix
        subsetlist : np.ndarray
            Gene names to include
        outliers : List[int]
            Indices of outlier genes to exclude
        for_randomset : bool, default=False
            Whether this is for random set computation
        algorithm : str, default='auto'
            PCA algorithm
        
        Returns
        -------
        pca : PCA
            Fitted PCA object
        X : np.ndarray
            Processed data matrix
        """
        mask = np.ones(len(subsetlist), dtype=bool)
        mask[outliers] = False
        subset_genes = subsetlist[mask]
        
        subset_adata = adata[:, subset_genes]
        X = self._prepare_data(subset_adata.X.T)
        
        pca = PCA(n_components=2, svd_solver=algorithm, random_state=42)
        pca.fit(X)
        
        if not for_randomset:
            self.svd = pca
            self.X = X
        
        return pca, X

    def robustIncrementalPCA(self, 
                           adata: sc.AnnData, 
                           subsetlist: np.ndarray, 
                           outliers: List[int], 
                           for_randomset: bool = False, 
                           partial_fit: bool = False,
                           batch_size: int = 1000) -> Tuple[IncrementalPCA, np.ndarray]:
        """
        Perform robust incremental PCA excluding outlier genes.
        
        Parameters
        ----------
        adata : sc.AnnData
            Annotated data matrix
        subsetlist : np.ndarray
            Gene names to include
        outliers : List[int]
            Indices of outlier genes to exclude
        for_randomset : bool, default=False
            Whether this is for random set computation
        partial_fit : bool, default=False
            Whether to use partial_fit
        batch_size : int, default=1000
            Batch size for incremental PCA
        
        Returns
        -------
        ipca : IncrementalPCA
            Fitted IncrementalPCA object
        X : np.ndarray
            Processed data matrix
        """
        mask = np.ones(len(subsetlist), dtype=bool)
        mask[outliers] = False
        subset_genes = subsetlist[mask]
        
        subset_adata = adata[:, subset_genes]
        X = self._prepare_data(subset_adata.X.T)
        
        ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
        
        if partial_fit:
            ipca.partial_fit(X)
        else:
            ipca.fit(X)
        
        if not for_randomset:
            self.svd = ipca
            self.X = X
        
        return ipca, X

    def fix_pc_sign(self, 
                   GeneScore: np.ndarray, 
                   SampleScore: np.ndarray, 
                   Wei: Optional[np.ndarray] = None, 
                   Mode: str = 'PreferActivation', 
                   DefWei: float = 1.0,
                   Thr: Optional[float] = None, 
                   Grouping: Optional[Dict[str, str]] = None, 
                   ExpMat: Optional[np.ndarray] = None, 
                   CorMethod: str = "pearson",
                   gene_set_name: Optional[str] = None) -> int:
        """
        Determine the correct orientation of principal component.
        
        Parameters
        ----------
        GeneScore : np.ndarray
            Gene loadings on PC
        SampleScore : np.ndarray
            Sample projections on PC
        Wei : Optional[np.ndarray], default=None
            Gene weights
        Mode : str, default='PreferActivation'
            Method for PC orientation
        DefWei : float, default=1.0
            Default weight for missing values
        Thr : Optional[float], default=None
            Threshold for various methods
        Grouping : Optional[Dict[str, str]], default=None
            Sample grouping information
        ExpMat : Optional[np.ndarray], default=None
            Expression matrix
        CorMethod : str, default='pearson'
            Correlation method
        gene_set_name : Optional[str], default=None
            Name of current gene set
        
        Returns
        -------
        sign : int
            +1 or -1 indicating PC orientation
        """
        from scipy.stats import pearsonr, spearmanr, kendalltau
        
        # Helper functions
        def apply_threshold_to_genescore(gscore: np.ndarray, thr: float) -> np.ndarray:
            q_val = np.quantile(np.abs(gscore), thr)
            return np.abs(gscore) >= q_val

        def correlation_function(method: str):
            if method == 'pearson':
                return pearsonr
            elif method == 'spearman':
                return spearmanr
            elif method == 'kendall':
                return kendalltau
            else:
                raise ValueError(f"Invalid CorMethod: {method}")

        def cor_test(x: np.ndarray, y: np.ndarray, method: str, thr: Optional[float]):
            func = correlation_function(method)
            corr, pval = func(x, y)
            if thr is None:
                return (np.nan, corr)
            else:
                return (pval, corr)

        # Ensure Wei is numpy array
        if Wei is not None:
            Wei = np.asarray(Wei, dtype=float)
        else:
            Wei = np.full_like(GeneScore, np.nan)

        # Handle different modes
        if Mode == 'none':
            return 1

        elif Mode == 'PreferActivation':
            ToUse = np.full(len(GeneScore), True, dtype=bool)
            if Thr is not None:
                ToUse = apply_threshold_to_genescore(GeneScore, Thr)
            
            return -1 if np.sum(GeneScore[ToUse]) < 0 else 1

        elif Mode == 'UseAllWeights':
            Wei = np.where(np.isnan(Wei), DefWei, Wei)
            Mode = 'UseKnownWeights'

        if Mode == 'UseKnownWeights':
            Wei = np.where(np.isnan(Wei), 0, Wei)
            
            ToUse = np.full(len(GeneScore), True)
            if Thr is not None:
                ToUse = apply_threshold_to_genescore(GeneScore, Thr)
            
            mask = (~np.isnan(Wei)) & ToUse
            if np.sum(mask) < 1:
                return 1
            
            return -1 if np.sum(Wei[mask] * GeneScore[mask]) < 0 else 1

        elif Mode == 'UseMeanExpressionAllWeights':
            Wei = np.where(np.isnan(Wei), DefWei, Wei)
            Mode = 'UseMeanExpressionKnownWeights'

        if Mode == 'UseMeanExpressionKnownWeights':
            if np.sum(~np.isnan(Wei)) < 1 or ExpMat is None:
                return 1
            
            ToUse = np.full(len(GeneScore), True)
            if Thr is not None:
                q_thr_high = np.quantile(GeneScore, Thr)
                q_thr_low = np.quantile(GeneScore, 1 - Thr)
                ToUse = (GeneScore >= max(q_thr_high, 0)) | (GeneScore <= min(0, q_thr_low))
            
            nbUsed = np.sum(ToUse)
            if nbUsed < 2:
                if nbUsed == 1:
                    median_per_gene = np.median(ExpMat, axis=1)
                    centered_median = median_per_gene - np.mean(median_per_gene)
                    val = np.sum(GeneScore[ToUse] * Wei[ToUse] * centered_median[ToUse])
                    return -1 if val <= 0 else 1
                else:
                    return 1
            
            subset_mat = ExpMat[ToUse, :]
            row_medians = np.median(subset_mat, axis=1)
            centered_medians = row_medians - np.mean(row_medians)
            val = np.sum(GeneScore[ToUse] * Wei[ToUse] * centered_medians)
            
            return -1 if val <= 0 else 1

        elif Mode == 'UseExtremeWeights':
            if ExpMat is None:
                return 1
            
            if Thr is None:
                ToUse = np.full(len(SampleScore), True, dtype=bool)
            else:
                cutoff = np.quantile(np.abs(SampleScore), 1 - Thr)
                ToUse = (np.abs(SampleScore) >= cutoff)
            
            gene_means = np.mean(ExpMat, axis=1)
            sum_val = np.sum(SampleScore[ToUse] * gene_means[ToUse])
            
            return -1 if sum_val <= 0 else 1

        # Default
        return 1

    def orient_pc1(self, 
                   pc1: np.ndarray, 
                   X: np.ndarray, 
                   raw_X_subset: np.ndarray, 
                   gene_set_name: Optional[str] = None) -> int:
        """
        Orient first principal component according to configured method.
        
        Parameters
        ----------
        pc1 : np.ndarray
            First principal component
        X : np.ndarray
            Processed data matrix
        raw_X_subset : np.ndarray
            Raw data subset for orientation
        gene_set_name : Optional[str], default=None
            Name of current gene set
        
        Returns
        -------
        correct_sign : int
            +1 or -1 for PC orientation
        """
        sample_score = pc1
        gene_score = X @ pc1
        
        # Get gene weights if available
        wei = self.gene_weights.get(gene_set_name, None)
        
        # Ensure raw_X_subset is dense array
        if sparse.issparse(raw_X_subset):
            raw_X_subset = raw_X_subset.toarray()
        
        correct_sign = self.fix_pc_sign(
            GeneScore=gene_score,
            SampleScore=sample_score,
            Wei=wei,
            DefWei=self.def_wei,
            Mode=self.pc_sign_mode,
            Thr=self.pc_sign_thr,
            Grouping=None,
            ExpMat=raw_X_subset,
            CorMethod=self.cor_method,
            gene_set_name=gene_set_name
        )
        
        return correct_sign

    def compute_median_exp(self, 
                            svd_: Union[TruncatedSVD, PCA, IncrementalPCA], 
                            X: np.ndarray, 
                            raw_X_subset: np.ndarray, 
                            gene_set_name: Optional[str] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute median expression for shifted pathway detection.
        
        Parameters
        ----------
        svd_ : Union[TruncatedSVD, PCA, IncrementalPCA]
            Fitted decomposition object
        X : np.ndarray
            Processed data matrix
        raw_X_subset : np.ndarray
            Raw data subset
        gene_set_name : Optional[str], default=None
            Name of current gene set
        
        Returns
        -------
        median_exp : float
            Median of gene projections on PC1
        projections_1 : np.ndarray
            Gene projections on PC1
        projections_2 : np.ndarray
            Gene projections on PC2
        """
        pc1, pc2 = svd_.components_
        
        # Orient PC1
        correct_sign = self.orient_pc1(pc1, X, raw_X_subset, gene_set_name)
        self.correct_pc_sign = correct_sign
        pc1 = correct_sign * pc1
        
        # Compute projections
        projections_1 = X @ pc1
        projections_2 = X @ pc2
        
        # Compute median
        median_exp = np.median(projections_1)
        
        return median_exp, projections_1, projections_2

    def process_iteration(self, 
                            sequence: np.ndarray, 
                            idx: np.ndarray, 
                            iteration: int, 
                            incremental: bool, 
                            partial_fit: bool, 
                            algorithm: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Process single iteration for null distribution computation.
        
        Parameters
        ----------
        sequence : np.ndarray
            Sequence of indices to sample from
        idx : np.ndarray
            Gene indices
        iteration : int
            Current iteration number
        incremental : bool
            Whether to use incremental PCA
        partial_fit : bool
            Whether to use partial fit
        algorithm : str
            SVD algorithm
        
        Returns
        -------
        l1 : float
            L1 value for this iteration
        median_exp : float
            Median expression for this iteration
        null_projections_1 : np.ndarray
            Projections on PC1
        null_projections_2 : np.ndarray
            Projections on PC2
        """
        # Set random seed for reproducibility within parallel processes
        np.random.seed(iteration)
        
        subset = np.random.choice(sequence, self.nullgenesetsize, replace=False)
        gene_subset = np.array([x for i, x in enumerate(idx) if i in subset])
        
        outliers = self.loocv(self.adata[:, gene_subset], for_randomset=True)
        
        if incremental:
            svd_, X = self.robustIncrementalPCA(self.adata, gene_subset, outliers, 
                                                for_randomset=True, partial_fit=partial_fit)
        else:
            svd_, X = self.robustTruncatedSVD(self.adata, gene_subset, outliers, 
                                                for_randomset=True, algorithm=algorithm)
        
        l1 = svd_.explained_variance_ratio_[0]
        
        # Get raw subset for median computation
        subsetlist = [x for i, x in enumerate(gene_subset) if i not in outliers]
        raw_X_subset = self.adata.raw[:, subsetlist].X.T
        
        median_exp, null_projections_1, null_projections_2 = self.compute_median_exp(
            svd_, X, raw_X_subset
        )
        
        return l1, median_exp, null_projections_1, null_projections_2

    def randomset_parallel(self, 
                            subsetlist: np.ndarray, 
                            outliers: List[int], 
                            verbose: int = 0, 
                            prefer_type: str = 'processes', 
                            incremental: bool = False, 
                            iters: int = 100, 
                            partial_fit: bool = False, 
                            algorithm: str = 'randomized') -> None:
        """
        Compute null distributions using parallel processing.
        
        Parameters
        ----------
        subsetlist : np.ndarray
            Array of gene names in current set
        outliers : List[int]
            Indices of outlier genes
        verbose : int, default=0
            Verbosity level
        prefer_type : str, default='processes'
            Joblib backend preference
        incremental : bool, default=False
            Whether to use incremental PCA
        iters : int, default=100
            Number of iterations
        partial_fit : bool, default=False
            Whether to use partial fit
        algorithm : str, default='randomized'
            SVD algorithm
        """
        start = time.time()
        
        candidate_nullgeneset_size = self.nullgenesetsize
        
        # Check cache
        if candidate_nullgeneset_size in self.null_distributions:
            self.nulll1, self.null_median_exp = self.null_distributions[candidate_nullgeneset_size]
            if verbose:
                print('Using existing null distribution')
            return
        
        # Setup for parallel processing
        sequence = np.arange(self.adata.shape[1])
        idx = self.adata.var.index.to_numpy()
        
        # Determine optimal number of jobs
        n_jobs = min(multiprocessing.cpu_count(), iters)
        
        # Run parallel iterations with batch processing for large iteration counts
        if iters > 1000:
            # Process in batches to avoid memory issues
            batch_size = 100
            n_batches = iters // batch_size
            all_results = []
            
            for batch in range(n_batches):
                batch_start = batch * batch_size
                batch_end = min((batch + 1) * batch_size, iters)
                
                batch_results = Parallel(n_jobs=n_jobs, prefer=prefer_type)(
                    delayed(self.process_iteration)(
                        sequence, idx, iteration, incremental, partial_fit, algorithm
                    ) for iteration in range(batch_start, batch_end)
                )
                all_results.extend(batch_results)
                
                # Clear memory periodically
                if batch % 10 == 0:
                    import gc
                    gc.collect()
            
            results = all_results
        else:
            # Standard parallel processing for smaller iteration counts
            results = Parallel(n_jobs=n_jobs, prefer=prefer_type)(
                delayed(self.process_iteration)(
                    sequence, idx, iteration, incremental, partial_fit, algorithm
                ) for iteration in range(iters)
            )
        
        # Unpack and convert results
        nulll1, null_median_exp, null_proj1, null_proj2 = zip(*results)
        
        nulll1_array = np.array(nulll1)
        null_median_exp_array = np.array(null_median_exp)
        
        # Store in cache
        self.null_distributions[candidate_nullgeneset_size] = [
            nulll1_array.reshape(-1, 1),  # Ensure 2D array
            null_median_exp_array
        ]
        
        self.nulll1 = nulll1_array.reshape(-1, 1)
        self.null_median_exp = null_median_exp_array
        
        if verbose:
            elapsed = time.time() - start
            minutes, seconds = divmod(elapsed, 60)
            print(f"Running time: {int(minutes):02}:{seconds:05.2f}")

    def get_raw_p_values(self, gene_set_name: Optional[str] = None) -> None:
        """
        Calculate raw p-values from null distributions.
        
        Parameters
        ----------
        gene_set_name : Optional[str], default=None
            Name of current gene set
        """
        # When parallel=False, compute a simple p-value
        if not hasattr(self, 'nulll1') or self.nulll1 is None or len(self.nulll1) == 0:
            # No null distribution available, use default values
            test_l1 = self.svd.explained_variance_ratio_[0]
            self.test_l1 = test_l1
            self.p_value = 0.5  # Default p-value when no null distribution
            
            # Compute median expression
            test_median_exp, projections_1, projections_2 = self.compute_median_exp(
                self.svd, self.X, self.raw_X_subset, gene_set_name
            )
            
            self.test_median_exp = test_median_exp
            self.med_exp_p_value = 0.5  # Default p-value
            self.projections_1 = projections_1
            self.projections_2 = projections_2
            
            return
        
        # Normal case with null distribution
        null_distribution = np.array(self.nulll1)
        if null_distribution.ndim == 2:
            null_distribution = null_distribution[:, 0]

        null_median_distribution = np.array(self.null_median_exp)
        
        # L1 statistics
        test_l1 = self.svd.explained_variance_ratio_[0]
        p_value = np.mean(null_distribution >= test_l1)
        
        self.test_l1 = test_l1
        self.p_value = p_value
        
        # Median expression statistics
        test_median_exp, projections_1, projections_2 = self.compute_median_exp(
            self.svd, self.X, self.raw_X_subset, gene_set_name
        )
        
        # Empirical p-value
        med_exp_p_value = (np.sum(np.abs(null_median_distribution) >= np.abs(test_median_exp)) + 1) / (len(null_median_distribution) + 1)
        
        self.test_median_exp = test_median_exp
        self.med_exp_p_value = med_exp_p_value
        self.projections_1 = projections_1
        self.projections_2 = projections_2
        
        # Clear temporary data
        self.nulll1 = None
        self.null_median_exp = None
        self.raw_X_subset = None

    def assess_significance(self, results: Dict[str, GeneSetResult]) -> Dict[str, GeneSetResult]:
        """
        Compute adjusted p-values using Benjamini-Hochberg procedure.
        
        Parameters
        ----------
        results : Dict[str, GeneSetResult]
            Results for each gene set
        
        Returns
        -------
        results : Dict[str, GeneSetResult]
            Results with adjusted p-values
        """
        n_tests = len(results)
        ps = np.zeros(n_tests)
        med_ps = np.zeros(n_tests)
        
        # Collect p-values
        for i, (_, gene_set_result) in enumerate(results.items()):
            ps[i] = gene_set_result.non_adj_p
            med_ps[i] = gene_set_result.non_adj_med_exp_p
        
        # Apply Benjamini-Hochberg correction
        qs = benj_hoch(ps)
        med_exp_qs = benj_hoch(med_ps)
        
        # Update results
        for i, (_, gene_set_result) in enumerate(results.items()):
            gene_set_result.q_value = qs[i]
            gene_set_result.med_exp_q_value = med_exp_qs[i]
        
        return results

    def approx_size(self, flag: bool) -> None:
        """
        Determine appropriate null gene set size with approximation.
        
        Parameters
        ----------
        flag : bool
            Whether this is the first gene set
        """
        candidate_size = sum(1 for i in range(len(self.subsetlist)) if i not in self.outliers)
        
        if flag:
            self.nullgenesetsize = candidate_size
        else:
            # Check if we can reuse existing null distribution
            for cached_size in self.null_distributions:
                if abs(cached_size - candidate_size) <= self.approx_int:
                    self.nullgenesetsize = cached_size
                    return
            
            self.nullgenesetsize = candidate_size

    def select_and_sort_gene_sets(self, selected_geneset_names: List[str]) -> List[str]:
        """
        Sort gene sets by size for efficient processing.
        
        Parameters
        ----------
        selected_geneset_names : List[str]
            Names of gene sets to process
        
        Returns
        -------
        sorted_names : List[str]
            Gene set names sorted by size
        """
        selected_gene_sets = {
            name: genes for name, genes in self.genesets.items() 
            if name in selected_geneset_names
        }
        
        sorted_gene_sets = sorted(selected_gene_sets.items(), key=lambda x: len(x[1]))
        
        return [name for name, _ in sorted_gene_sets]

    def p_values_in_frame(self, assessed_results: Dict[str, GeneSetResult]) -> pd.DataFrame:
        """
        Create DataFrame with all p-values and statistics.
        
        Parameters
        ----------
        assessed_results : Dict[str, GeneSetResult]
            Results with computed p-values
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with all statistics
        """
        data = {
            'L1': {},
            'ppv L1': {},
            'Median Exp': {},
            'ppv Med Exp': {},
            'q L1': {},
            'q Med Exp': {}
        }
        
        for name, result in assessed_results.items():
            data['L1'][name] = result.test_l1
            data['ppv L1'][name] = result.non_adj_p
            data['Median Exp'][name] = result.test_median_exp
            data['ppv Med Exp'][name] = result.non_adj_med_exp_p
            data['q L1'][name] = result.q_value
            data['q Med Exp'][name] = result.med_exp_q_value
        
        return pd.DataFrame(data)

    def compute(self, 
                selected_gene_sets: Union[List[str], str] = 'all', 
                parallel: bool = True, 
                incremental: bool = False, 
                iters: int = 100, 
                partial_fit: bool = False, 
                algorithm: str = 'randomized', 
                loocv_on: bool = True, 
                double_mean_centering: bool = False, 
                outlier_fisher_thr: float = 0.05, 
                min_gene_sets: int = 3, 
                max_outlier_prop: float = 0.5,
                verbose: int = 0) -> None:
        """
        Compute ROMA module activity scores for selected gene sets.
        
        Parameters
        ----------
        selected_gene_sets : Union[List[str], str], default='all'
            List of gene set names to analyze, or 'all' for all gene sets
        parallel : bool, default=False
            Whether to use parallel processing for null distribution computation
        incremental : bool, default=False
            Whether to use incremental PCA instead of truncated SVD
        iters : int, default=100
            Number of iterations for null distribution generation
        partial_fit : bool, default=False
            Whether to use partial_fit for incremental PCA
        algorithm : str, default='randomized'
            Algorithm for SVD computation ('randomized' or 'arpack')
        loocv_on : bool, default=True
            Whether to perform leave-one-out cross-validation for outlier detection
        double_mean_centering : bool, default=False
            Whether to center data across both samples and genes
        outlier_fisher_thr : float, default=0.05
            P-value threshold for Fisher's exact test in outlier detection
        min_gene_sets : int, default=3
            Minimum number of gene sets a gene must appear in to be considered for outlier removal
        max_outlier_prop : float, default=0.5
            Maximum proportion of genes that can be removed as outliers per gene set
        verbose : int, default=0
            Verbosity level
        
        Raises
        ------
        ValueError
            If no gene sets are loaded or if data is not properly initialized
        """
        if self.adata is None:
            raise ValueError("No data loaded. Set adata attribute first.")
        
        # Load gene sets from gmt
        self.read_gmt_to_dict(self.gmt)
        
        if not self.genesets:
            raise ValueError("No gene sets loaded. Run read_gmt_to_dict() first.")
        
        results = {}
        
        # Store raw data efficiently
        if not hasattr(self.adata, 'raw') or self.adata.raw is None:
            self.adata.raw = self.adata
        
        # Prepare data with minimal copying
        X = self.adata.X.T if not self.adata.is_view else self.adata.X.T#.copy()
        
        # Convert sparse to dense if needed
        #X = self._prepare_data(X)
        #if sparse.issparse(X):
        #    X = X.toarray()
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        # centering the sparse matrix would give a dense one

        # Center data
        if double_mean_centering:
            X_centered = self.double_mean_center_matrix(X)
        else:
            #if sparse.issparse(X):
                # replicates the centering operation below for sparse matrices 
                #X_centered = sparse_centering_operator(X)
            #else:
            # Center over samples (genes have 0 mean)
            X_centered = X - np.mean(X, axis=1, keepdims=True)
            X_centered = X_centered - np.mean(X_centered, axis=0, keepdims=True)
        


        self.adata.X = X_centered.T
        
        # Index genes
        self.indexing(self.adata)
        
        # Handle gene set selection
        if selected_gene_sets == 'all':
            selected_gene_sets = list(self.genesets.keys())
        
        # Sort gene sets by size for efficiency
        sorted_gene_sets = self.select_and_sort_gene_sets(selected_gene_sets)
        
        # Process gene sets
        flag = True
        for gene_set_name in sorted_gene_sets:
            print(f'Processing gene set: {color.BOLD}{color.DARKCYAN}{gene_set_name}{color.END}', end=' | ')
            
            # Subset data
            self.subsetting(self.adata, self.genesets[gene_set_name])
            print(f'len of subsetlist: {color.BOLD}{len(self.subsetlist)}{color.END}', end=' ')
            
            if len(self.subsetlist) < self.min_n_genes:
                self.unprocessed_genesets.append(gene_set_name)
                print(" | smaller than min n genes for geneset |")
                continue
            else:
                print()
            
            # Outlier detection
            if loocv_on:
                self.loocv(self.subset, verbose=verbose)
            
            if len(self.outliers) > 0:
                print(f"Outliers: {self.outliers}, Gene: {self.subsetlist[self.outliers[0]]}")
            
            # Update global counts for Fisher test
            for gene in self.subsetlist:
                self.global_gene_counts[gene] = self.global_gene_counts.get(gene, 0) + 1
            
            for outlier_idx in self.outliers:
                gene_name = self.subsetlist[outlier_idx]
                self.global_outlier_counts[gene_name] = self.global_outlier_counts.get(gene_name, 0) + 1
            
            # Apply Fisher test
            gene_flags = self.fisher_outlier_filter(
                self.global_outlier_counts, 
                self.global_gene_counts,
                outlier_fisher_thr=outlier_fisher_thr, 
                min_gene_sets=min_gene_sets
            )
            
            # Limit outliers per gene set
            gene_flags = self.limit_outliers_per_geneset(
                self.subsetlist, 
                gene_flags, 
                self.loocv_scores, 
                max_outlier_prop=max_outlier_prop
            )
            
            # Update outliers list
            self.outliers = [i for i, gene in enumerate(self.subsetlist) if gene_flags.get(gene, False)]
            
            # Determine null gene set size
            self.approx_size(flag)
            flag = False
            
            # Perform decomposition
            if incremental:
                self.robustIncrementalPCA(self.adata, self.subsetlist, self.outliers)
            else:
                self.robustTruncatedSVD(self.adata, self.subsetlist, self.outliers, algorithm=algorithm)
            
            # Get raw data for PC sign correction
            subsetlist_no_out = [x for i, x in enumerate(self.subsetlist) if i not in self.outliers]
            self.raw_X_subset = self.adata.raw[:, subsetlist_no_out].X.T#.copy()
            
            # Compute null distributions if parallel
            if parallel:
                self.randomset_parallel(
                    self.subsetlist, 
                    self.outliers, 
                    verbose=verbose,
                    prefer_type='processes', 
                    incremental=incremental, 
                    iters=iters, 
                    partial_fit=partial_fit,
                    algorithm=algorithm
                )
            
            # Calculate p-values
            self.get_raw_p_values(gene_set_name)
            
            # Store results
            gene_set_result = GeneSetResult(
                self.subset, 
                self.subsetlist, 
                self.outliers, 
                self.nullgenesetsize,
                self.svd
            )
            
            gene_set_result.custom_name = f"GeneSetResult {gene_set_name}"
            gene_set_result.test_l1 = self.test_l1
            gene_set_result.non_adj_p = self.p_value
            gene_set_result.non_adj_med_exp_p = self.med_exp_p_value
            gene_set_result.test_median_exp = self.test_median_exp
            gene_set_result.projections_1 = self.projections_1
            gene_set_result.projections_2 = self.projections_2
            
            results[gene_set_name] = gene_set_result
        
        # Assess significance with multiple testing correction
        assessed_results = self.assess_significance(results)
        
        # Store results
        self.adata.uns['ROMA'] = assessed_results
        self.adata.uns['ROMA_stats'] = self.p_values_in_frame(assessed_results)
        self.select_active_modules(self.q_L1_threshold, self.q_Med_Exp_threshold)
        
        # Update status
        self.custom_name = f"{color.BOLD}{color.GREEN}scROMA{color.END}: module activities are computed"
        print(f"{color.BOLD}{color.PURPLE}Finished{color.END}", end=': ')
        
        # Update plotting module
        self.pl.adata = self.adata

    def compute_batch(self, 
                        selected_gene_sets: List[str], 
                        batch_size: int = 50,
                        **kwargs) -> None:
        """
        Process gene sets in batches for memory efficiency.
        
        Parameters
        ----------
        selected_gene_sets : List[str]
            Gene sets to analyze
        batch_size : int, default=50
            Number of gene sets per batch
        **kwargs
            Additional arguments passed to compute()
        """
        n_batches = (len(selected_gene_sets) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(selected_gene_sets))
            batch = selected_gene_sets[start:end]
            
            print(f"Processing batch {i+1}/{n_batches} ({len(batch)} gene sets)")
            self.compute(batch, **kwargs)
            
            # Clear memory
            import gc
            gc.collect()

    def select_active_modules(self, 
                            q_L1_threshold: float = 0.05, 
                            q_Med_Exp_threshold: float = 0.05) -> None:
        """
        Select significantly active modules based on q-value thresholds.
        
        Parameters
        ----------
        q_L1_threshold : float, default=0.05
            Q-value threshold for L1 significance
        q_Med_Exp_threshold : float, default=0.05
            Q-value threshold for median expression significance
        """
        df = self.adata.uns['ROMA_stats']
        active_modules = df[(df['q L1'] <= q_L1_threshold) | (df['q Med Exp'] <= q_Med_Exp_threshold)]
        self.adata.uns['ROMA_active_modules'] = active_modules

    def save_active_modules_results(self, 
                                    path: str, 
                                    only_active: bool = True, 
                                    save_adata: bool = True) -> None:
        """
        Save ROMA results to disk.
        
        Parameters
        ----------
        path : str
            Output directory path
        only_active : bool, default=True
            Whether to save only active modules
        save_adata : bool, default=True
            Whether to save the AnnData object
        """
        os.makedirs(path, exist_ok=True)
        
        active_modules = self.adata.uns['ROMA_active_modules'].index
        
        if only_active:
            selected_dict = {k: v for k, v in self.adata.uns['ROMA'].items() if k in active_modules}
        else:
            selected_dict = self.adata.uns['ROMA']
        
        # Define attributes to save
        attributes = {
            "subsetlist": "numpy.ndarray",
            "outliers": "list",
            "projections_1": "numpy.ndarray",
            "projections_2": "numpy.ndarray",
            "svd.components_": "numpy.ndarray"
        }
        
        # Save each gene set result
        for key, gene_set_result in selected_dict.items():
            key_dir = os.path.join(path, key)
            os.makedirs(key_dir, exist_ok=True)
            
            for attr, attr_type in attributes.items():
                if "." in attr:
                    parts = attr.split(".")
                    attr_value = getattr(gene_set_result, parts[0], None)
                    if attr_value is not None:
                        attr_value = getattr(attr_value, parts[1], None)
                else:
                    attr_value = getattr(gene_set_result, attr, None)
                
                if attr_value is not None:
                    file_path = os.path.join(key_dir, attr)
                    
                    if attr_type == "numpy.ndarray":
                        np.save(file_path, attr_value)
                    elif attr_type == "list":
                        with open(file_path, "wb") as f:
                            pickle.dump(attr_value, f)
        
        # Save DataFrames
        self.adata.uns['ROMA_stats'].to_csv(os.path.join(path, "ROMA_stats.csv"))
        self.adata.uns['ROMA_active_modules'].to_csv(os.path.join(path, "ROMA_active_modules.csv"))
        
        # Save AnnData if requested
        if save_adata:
            del self.adata.uns['ROMA']  # Remove large dictionary before saving
            self.adata.write(os.path.join(path, "roma_adata.h5ad"))

    def load_active_modules_results(self, path: str) -> None:
        """
        Load previously saved ROMA results.
        
        Parameters
        ----------
        path : str
            Path to saved results directory
        
        Raises
        ------
        FileNotFoundError
            If results directory doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results directory not found: {path}")
        
        loaded_dict = {}
        
        # Load each gene set result
        for key in os.listdir(path):
            key_dir = os.path.join(path, key)
            if os.path.isdir(key_dir):
                gene_set = GeneSetResult(None, None, None, None, None)
                gene_set.custom_name = key
                
                for file in os.listdir(key_dir):
                    file_path = os.path.join(key_dir, file)
                    attr_name = file.replace(".npy", "")
                    
                    if attr_name == "subsetlist":
                        gene_set.subsetlist = np.load(file_path, allow_pickle=True)
                    elif attr_name == "outliers":
                        with open(file_path, "rb") as f:
                            gene_set.outliers = pickle.load(f)
                    elif attr_name == "projections_1":
                        gene_set.projections_1 = np.load(file_path, allow_pickle=True)
                    elif attr_name == "projections_2":
                        gene_set.projections_2 = np.load(file_path, allow_pickle=True)
                    elif attr_name == "svd.components_":
                        components = np.load(file_path, allow_pickle=True)
                        gene_set.svd = SimpleNamespace(components_=components)
                
                loaded_dict[key] = gene_set
        
        # Load AnnData if not already loaded
        if self.adata is None:
            adata_path = os.path.join(path, "roma_adata.h5ad")
            if os.path.exists(adata_path):
                self.adata = sc.read_h5ad(adata_path)
            else:
                raise FileNotFoundError(f"AnnData file not found: {adata_path}")
        
        # Load DataFrames
        self.adata.uns['ROMA'] = loaded_dict
        self.adata.uns['ROMA_stats'] = pd.read_csv(os.path.join(path, "ROMA_stats.csv"), index_col=0)
        self.adata.uns['ROMA_active_modules'] = pd.read_csv(os.path.join(path, "ROMA_active_modules.csv"), index_col=0)
        
        # Update plotting module
        self.pl.adata = self.adata

    def get_module_scores(self, module_name: str) -> Optional[np.ndarray]:
        """
        Get activity scores for a specific module across all samples.
        
        Parameters
        ----------
        module_name : str
            Name of the gene set/module
        
        Returns
        -------
        scores : Optional[np.ndarray]
            Activity scores across samples, or None if module not found
        """
        if 'ROMA' not in self.adata.uns:
            raise ValueError("No ROMA results found. Run compute() first.")
        
        if module_name not in self.adata.uns['ROMA']:
            warnings.warn(f"Module '{module_name}' not found in results.")
            return None
        
        result = self.adata.uns['ROMA'][module_name]
        # Compute sample scores from SVD
        if hasattr(result, 'svd') and result.svd is not None:
            # Sample scores are the projection of samples onto PC1
            pc1 = result.svd.components_[0]
            sample_scores = self.adata.X @ result.projections_1
            return sample_scores
        
        return None

    def get_top_genes(self, 
                        module_name: str, 
                        n_genes: int = 10, 
                        absolute: bool = True) -> Optional[pd.DataFrame]:
        """
        Get top contributing genes for a module.
        
        Parameters
        ----------
        module_name : str
            Name of the gene set/module
        n_genes : int, default=10
            Number of top genes to return
        absolute : bool, default=True
            Whether to sort by absolute weight values
        
        Returns
        -------
        top_genes : Optional[pd.DataFrame]
            DataFrame with gene names and weights
        """
        if 'ROMA' not in self.adata.uns:
            raise ValueError("No ROMA results found. Run compute() first.")
        
        if module_name not in self.adata.uns['ROMA']:
            warnings.warn(f"Module '{module_name}' not found in results.")
            return None
        
        result = self.adata.uns['ROMA'][module_name]
        
        if result.projections_1 is None or result.subsetlist is None:
            return None
        
        # Create DataFrame with genes and their weights
        weights = result.projections_1 * self.correct_pc_sign
        df = pd.DataFrame({
            'gene': result.subsetlist,
            'weight': weights,
            'abs_weight': np.abs(weights)
        })
        
        # Sort and get top genes
        if absolute:
            df = df.nlargest(n_genes, 'abs_weight')
        else:
            df = df.nlargest(n_genes, 'weight')
        
        return df[['gene', 'weight']]

    def export_results(self, 
                        output_file: str, 
                        format: str = 'csv',
                        include_projections: bool = False) -> None:
        """
        Export ROMA results to various formats.
        
        Parameters
        ----------
        output_file : str
            Output file path
        format : str, default='csv'
            Output format ('csv', 'excel', 'h5')
        include_projections : bool, default=False
            Whether to include gene projections
        """
        if 'ROMA_stats' not in self.adata.uns:
            raise ValueError("No ROMA results found. Run compute() first.")
        
        stats_df = self.adata.uns['ROMA_stats']
        
        if format == 'csv':
            stats_df.to_csv(output_file)
        elif format == 'excel':
            with pd.ExcelWriter(output_file) as writer:
                stats_df.to_excel(writer, sheet_name='Module_Statistics')
                
                if 'ROMA_active_modules' in self.adata.uns:
                    self.adata.uns['ROMA_active_modules'].to_excel(
                        writer, sheet_name='Active_Modules'
                    )
                
                if include_projections:
                    for module_name, result in self.adata.uns['ROMA'].items():
                        if result.projections_1 is not None:
                            proj_df = pd.DataFrame({
                                'gene': result.subsetlist,
                                'PC1': result.projections_1,
                                'PC2': result.projections_2
                            })
                            # Excel sheet names have length limit
                            sheet_name = f"Proj_{module_name[:25]}"
                            proj_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        elif format == 'h5':
            stats_df.to_hdf(output_file, key='module_stats', mode='w')
            
            if 'ROMA_active_modules' in self.adata.uns:
                self.adata.uns['ROMA_active_modules'].to_hdf(
                    output_file, key='active_modules', mode='a'
                )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def compare_conditions(self, 
                            condition_key: str,
                            module_name: str,
                            test: str = 'wilcoxon') -> Dict[str, Any]:
        """
        Compare module activity between conditions.
        
        Parameters
        ----------
        condition_key : str
            Key in adata.obs containing condition labels
        module_name : str
            Name of module to compare
        test : str, default='wilcoxon'
            Statistical test to use ('wilcoxon', 't-test', 'mann-whitney')
        
        Returns
        -------
        results : Dict[str, Any]
            Test results including p-value and effect size
        """
        from scipy.stats import mannwhitneyu, ttest_ind
        
        scores = self.get_module_scores(module_name)
        if scores is None:
            raise ValueError(f"Module '{module_name}' not found.")
        
        if condition_key not in self.adata.obs:
            raise ValueError(f"Condition key '{condition_key}' not found in adata.obs.")
        
        conditions = self.adata.obs[condition_key].unique()
        if len(conditions) != 2:
            raise ValueError(f"Expected 2 conditions, found {len(conditions)}.")
        
        # Split scores by condition
        mask1 = self.adata.obs[condition_key] == conditions[0]
        mask2 = self.adata.obs[condition_key] == conditions[1]
        
        scores1 = scores[mask1]
        scores2 = scores[mask2]
        
        # Perform test
        if test == 'wilcoxon':
            stat, pval = mannwhitneyu(scores1, scores2, alternative='two-sided')
        elif test == 't-test':
            stat, pval = ttest_ind(scores1, scores2)
        elif test == 'mann-whitney':
            stat, pval = mannwhitneyu(scores1, scores2)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(scores1) - np.mean(scores2)
        pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'module': module_name,
            'condition1': conditions[0],
            'condition2': conditions[1],
            'n1': len(scores1),
            'n2': len(scores2),
            'mean1': np.mean(scores1),
            'mean2': np.mean(scores2),
            'median1': np.median(scores1),
            'median2': np.median(scores2),
            'statistic': stat,
            'pvalue': pval,
            'cohens_d': cohens_d,
            'test': test
        }

    def validate_results(self) -> Dict[str, List[str]]:
        """
        Validate ROMA results for potential issues.
        
        Returns
        -------
        issues : Dict[str, List[str]]
            Dictionary of potential issues found
        """
        issues = {
            'missing_genes': [],
            'small_modules': [],
            'high_outliers': [],
            'low_variance': [],
            'convergence': []
        }
        
        if 'ROMA' not in self.adata.uns:
            issues['missing_genes'].append("No ROMA results found.")
            return issues
        
        for module_name, result in self.adata.uns['ROMA'].items():
            # Check for missing genes
            original_size = len(self.genesets.get(module_name, []))
            found_size = len(result.subsetlist) if result.subsetlist is not None else 0
            missing_pct = (original_size - found_size) / original_size * 100 if original_size > 0 else 0
            
            if missing_pct > 50:
                issues['missing_genes'].append(
                    f"{module_name}: {missing_pct:.1f}% genes missing"
                )
            
            # Check module size
            if found_size < self.min_n_genes:
                issues['small_modules'].append(
                    f"{module_name}: only {found_size} genes found"
                )
            
            # Check outlier proportion
            if result.outliers is not None and found_size > 0:
                outlier_pct = len(result.outliers) / found_size * 100
                if outlier_pct > 30:
                    issues['high_outliers'].append(
                        f"{module_name}: {outlier_pct:.1f}% outliers"
                    )
            
            # Check variance explained
            if result.test_l1 is not None and result.test_l1 < 0.1:
                issues['low_variance'].append(
                    f"{module_name}: only {result.test_l1:.3f} variance explained"
                )
        
        return issues

    def summary(self) -> None:
        """Print summary of ROMA analysis results."""
        if 'ROMA_stats' not in self.adata.uns:
            print("No ROMA results found. Run compute() first.")
            return
        
        stats = self.adata.uns['ROMA_stats']
        active = self.adata.uns.get('ROMA_active_modules', pd.DataFrame())
        
        print(f"\n{color.BOLD}=== ROMA Analysis Summary ==={color.END}\n")
        print(f"Total gene sets analyzed: {len(stats)}")
        print(f"Active modules (overdispersed): {len(active[active['q L1'] <= self.q_L1_threshold])}")
        print(f"Active modules (shifted): {len(active[active['q Med Exp'] <= self.q_Med_Exp_threshold])}")
        
        if len(self.unprocessed_genesets) > 0:
            print(f"\nUnprocessed gene sets (too small): {len(self.unprocessed_genesets)}")
            if len(self.unprocessed_genesets) <= 10:
                print(f"  {', '.join(self.unprocessed_genesets)}")
        
        # Top modules by L1
        print(f"\n{color.BOLD}Top 5 overdispersed modules:{color.END}")
        top_l1 = stats.nsmallest(5, 'q L1')[['L1', 'q L1']]
        for idx, row in top_l1.iterrows():
            print(f"  {idx}: L1={row['L1']:.3f}, q={row['q L1']:.3e}")
        
        # Top modules by median expression
        print(f"\n{color.BOLD}Top 5 shifted modules:{color.END}")
        top_med = stats.nsmallest(5, 'q Med Exp')[['Median Exp', 'q Med Exp']]
        for idx, row in top_med.iterrows():
            print(f"  {idx}: Med={row['Median Exp']:.3f}, q={row['q Med Exp']:.3e}")
        
        # Validation summary
        issues = self.validate_results()
        total_issues = sum(len(v) for v in issues.values())
        if total_issues > 0:
            print(f"\n{color.BOLD}Potential issues found: {total_issues}{color.END}")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"  {issue_type}: {len(issue_list)} modules")

    def __del__(self):
        """Cleanup when object is destroyed."""
        # Clean up any temporary files or large objects
        if hasattr(self, 'parallel_results') and self.parallel_results is not None:
            try:
                self.parallel_results.clear()
            except:
                pass
            self.parallel_results.clear()
        if hasattr(self, 'null_distributions'):
            try:
                self.null_distributions.clear()
            except:
                pass