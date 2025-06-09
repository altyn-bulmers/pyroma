# test_pyroma.py

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
import os
from unittest.mock import Mock, patch
import warnings
from scipy import sparse

# Import the ROMA class (adjust path as needed)
from pyroma import ROMA, GeneSetResult, color


class TestROMA:
    """Test suite for PyROMA implementation."""
    
    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData for testing."""
        np.random.seed(42)
        n_obs = 100
        n_vars = 500
        
        # Create random expression data
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
        
        # Create AnnData object
        adata = sc.AnnData(X=X)
        adata.obs['cell_type'] = np.random.choice(['A', 'B', 'C'], size=n_obs)
        adata.obs['condition'] = np.random.choice(['control', 'treated'], size=n_obs)
        adata.var['gene_names'] = [f'Gene_{i}' for i in range(n_vars)]
        adata.var_names = adata.var['gene_names']
        
        return adata
    
    @pytest.fixture
    def sample_gmt_file(self):
        """Create sample GMT file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gmt', delete=False) as f:
            f.write("PATHWAY1\tDescription1\tGene_0\tGene_1\tGene_2\tGene_3\tGene_4\n")
            f.write("PATHWAY2\tDescription2\tGene_10\tGene_11\tGene_12\tGene_13\tGene_14\tGene_15\n")
            f.write("PATHWAY3\tDescription3\tGene_50\tGene_51\tGene_52\n")  # Small pathway
            f.write("PATHWAY4\tDescription4\t" + "\t".join([f"Gene_{i}" for i in range(100, 150)]) + "\n")  # Large pathway
            f.write("PATHWAY5\tDescription5\tGene_500\tGene_501\tGene_502\n")  # Missing genes
            return f.name
    
    @pytest.fixture
    def roma_instance(self):
        """Create ROMA instance."""
        return ROMA()
    
    def test_initialization(self, roma_instance):
        """Test ROMA initialization."""
        assert roma_instance.adata is None
        assert roma_instance.gmt is None
        assert roma_instance.genesets == {}
        assert roma_instance.min_n_genes == 10
        assert roma_instance.approx_int == 20
        assert roma_instance.q_L1_threshold == 0.05
        assert roma_instance.q_Med_Exp_threshold == 0.05
    
    def test_read_gmt_to_dict(self, roma_instance, sample_gmt_file):
        """Test GMT file reading."""
        genesets = roma_instance.read_gmt_to_dict(sample_gmt_file)
        
        assert len(genesets) == 5
        assert 'PATHWAY1' in genesets
        assert len(genesets['PATHWAY1']) == 5
        assert 'Gene_0' in genesets['PATHWAY1']
        
        # Test invalid file
        with pytest.raises(FileNotFoundError):
            roma_instance.read_gmt_to_dict('nonexistent.gmt')
        
        # Test invalid format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gmt', delete=False) as f:
            f.write("INVALID\n")  # Missing required fields
            f.flush()
            
            with pytest.raises(ValueError):
                roma_instance.read_gmt_to_dict(f.name)
        
        os.unlink(sample_gmt_file)
    
    def test_indexing(self, roma_instance, sample_adata):
        """Test gene indexing."""
        roma_instance.indexing(sample_adata)
        assert len(roma_instance.idx) == 500
        assert 'Gene_0' in roma_instance.idx
    
    def test_subsetting(self, roma_instance, sample_adata):
        """Test data subsetting."""
        roma_instance.adata = sample_adata
        roma_instance.indexing(sample_adata)
        
        geneset = np.array(['Gene_0', 'Gene_1', 'Gene_2', 'Gene_999'])  # One missing
        subset, subsetlist = roma_instance.subsetting(sample_adata, geneset)
        
        assert subset.shape[1] == 3  # Only 3 genes found
        assert len(subsetlist) == 3
        assert 'Gene_999' not in subsetlist
    
    def test_double_mean_center_matrix(self, roma_instance):
        """Test double centering."""
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        centered = roma_instance.double_mean_center_matrix(matrix)
        
        # Check that row and column means are approximately zero
        assert np.allclose(np.mean(centered, axis=1), 0, atol=1e-10)
        assert np.allclose(np.mean(centered, axis=0), 0, atol=1e-10)
    
    def test_loocv(self, roma_instance, sample_adata):
        """Test leave-one-out cross-validation."""
        # Create subset with clear outlier
        subset = sample_adata[:, :20]
        subset.X[:, 0] = subset.X[:, 0] * 100  # Make first gene an outlier
        
        outliers = roma_instance.loocv(subset, verbose=1)
        
        # Should detect the outlier
        assert 0 in outliers or len(outliers) > 0
    
    def test_fisher_outlier_filter(self, roma_instance):
        """Test Fisher's exact test for outliers."""
        gene_outlier_counts = {'Gene_A': 5, 'Gene_B': 1, 'Gene_C': 0}
        gene_pathway_counts = {'Gene_A': 10, 'Gene_B': 10, 'Gene_C': 2}
        
        flags = roma_instance.fisher_outlier_filter(
            gene_outlier_counts,
            gene_pathway_counts,
            outlier_fisher_thr=0.05,
            min_gene_sets=3
        )
        
        # Gene_C should not be flagged (too few gene sets)
        assert flags['Gene_C'] == False
        # Gene_A should have a boolean value
        assert 'Gene_A' in flags
        assert isinstance(flags['Gene_A'], (bool, np.bool_))
    
    def test_prepare_data(self, roma_instance):
        """Test data preparation with sparse matrices."""
        # Dense array
        dense = np.random.rand(10, 20)
        prepared = roma_instance._prepare_data(dense)
        assert isinstance(prepared, np.ndarray)
        
        # Sparse array with high density
        sparse_high = sparse.random(10, 20, density=0.6)
        prepared = roma_instance._prepare_data(sparse_high)
        assert isinstance(prepared, np.ndarray)
        
        # Sparse array with low density
        sparse_low = sparse.random(10, 20, density=0.1)
        prepared = roma_instance._prepare_data(sparse_low)
        # Should remain sparse for efficiency
        assert sparse.issparse(prepared) or isinstance(prepared, np.ndarray)
    
    def test_robust_svd(self, roma_instance, sample_adata):
        """Test robust SVD computation."""
        roma_instance.indexing(sample_adata)
        subsetlist = np.array(['Gene_0', 'Gene_1', 'Gene_2', 'Gene_3', 'Gene_4'])
        outliers = [1]  # Exclude Gene_1
        
        svd, X = roma_instance.robustTruncatedSVD(
            sample_adata, subsetlist, outliers, algorithm='randomized'
        )
        
        assert svd.n_components == 2
        assert X.shape[0] == 4  # 5 genes - 1 outlier
        assert X.shape[1] == 100   # samples
        assert hasattr(svd, 'explained_variance_ratio_')
    
    def test_fix_pc_sign(self, roma_instance):
        """Test PC sign correction."""
        gene_score = np.array([0.5, -0.3, 0.8, -0.2])
        sample_score = np.array([1.0, -0.5, 0.7, -0.3, 0.2])
        
        # Test PreferActivation mode
        sign = roma_instance.fix_pc_sign(
            gene_score, sample_score, Mode='PreferActivation'
        )
        assert sign in [-1, 1]
        
        # Test with weights
        weights = np.array([1.0, -1.0, 1.0, -1.0])
        sign = roma_instance.fix_pc_sign(
            gene_score, sample_score, Wei=weights, Mode='UseKnownWeights'
        )
        assert sign in [-1, 1]
    
    def test_compute_median_exp(self, roma_instance, sample_adata):
        """Test median expression computation."""
        from sklearn.decomposition import TruncatedSVD
        
        X = np.random.randn(10, 5)
        svd = TruncatedSVD(n_components=2)
        svd.fit(X)
        
        raw_subset = np.random.randn(5, 10)
        
        median_exp, proj1, proj2 = roma_instance.compute_median_exp(
            svd, X, raw_subset, 'test_gene_set'
        )
        
        assert isinstance(median_exp, float)
        assert proj1.shape[0] == X.shape[0]
        assert proj2.shape[0] == X.shape[0]
    
    def test_compute_full_pipeline(self, roma_instance, sample_adata, sample_gmt_file):
        """Test full computation pipeline."""
        # Setup
        roma_instance.adata = sample_adata
        roma_instance.read_gmt_to_dict(sample_gmt_file)
        roma_instance.min_n_genes = 5 
        
        # Run computation
        roma_instance.compute(
            selected_gene_sets=['PATHWAY4'],
            parallel=False,
            loocv_on=True,
            iters=10  # Small number for testing
        )
        
        # Check results
        assert 'ROMA' in roma_instance.adata.uns
        assert 'ROMA_stats' in roma_instance.adata.uns
        assert 'ROMA_active_modules' in roma_instance.adata.uns
        
        stats = roma_instance.adata.uns['ROMA_stats']
        assert len(stats) >= 1  # At least one pathway processed
        assert 'L1' in stats.columns
        assert 'q L1' in stats.columns
        
        os.unlink(sample_gmt_file)
    
    def test_parallel_computation(self, roma_instance, sample_adata):
        """Test parallel null distribution computation."""
        roma_instance.adata = sample_adata
        roma_instance.adata.raw = sample_adata
        roma_instance.indexing(sample_adata)
        roma_instance.nullgenesetsize = 10
        
        subsetlist = np.array(['Gene_0', 'Gene_1', 'Gene_2', 'Gene_3', 'Gene_4'])
        outliers = []
        
        # Mock raw data
        roma_instance.raw_X_subset = sample_adata[:, subsetlist].X.T
        
        roma_instance.randomset_parallel(
            subsetlist, outliers, verbose=0, iters=20
        )
        
        assert roma_instance.nulll1 is not None
        assert roma_instance.null_median_exp is not None
        assert len(roma_instance.nulll1) == 20
    
    def test_save_load_results(self, roma_instance, sample_adata, sample_gmt_file):
        """Test saving and loading results."""
        # Run analysis
        roma_instance.adata = sample_adata
        roma_instance.read_gmt_to_dict(sample_gmt_file)
        roma_instance.compute(selected_gene_sets=['PATHWAY1'], iters=5)
        
        # Save results
        with tempfile.TemporaryDirectory() as tmpdir:
            roma_instance.save_active_modules_results(tmpdir)
            
            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, 'ROMA_stats.csv'))
            assert os.path.exists(os.path.join(tmpdir, 'roma_adata.h5ad'))
            
            # Load results
            new_roma = ROMA()
            new_roma.load_active_modules_results(tmpdir)
            
            assert 'ROMA' in new_roma.adata.uns
            assert 'ROMA_stats' in new_roma.adata.uns
        
        os.unlink(sample_gmt_file)
    
    def test_get_module_scores(self, roma_instance, sample_adata, sample_gmt_file):
        """Test module score extraction."""
        roma_instance.adata = sample_adata
        roma_instance.read_gmt_to_dict(sample_gmt_file)
        roma_instance.compute(selected_gene_sets=['PATHWAY1'], iters=5)
        
        scores = roma_instance.get_module_scores('PATHWAY1')
        assert scores is not None
        
        # Test missing module
        with warnings.catch_warnings(record=True):
            scores = roma_instance.get_module_scores('NONEXISTENT')
            assert scores is None
        
        os.unlink(sample_gmt_file)
    
    def test_edge_cases(self, roma_instance):
        """Test edge cases and error handling."""
        # Test compute without data
        with pytest.raises(ValueError):
            roma_instance.compute()
        
        # Test with empty gene sets
        roma_instance.adata = sc.AnnData(np.random.rand(10, 10))
        roma_instance.genesets = {'EMPTY': np.array([])}
        
        with warnings.catch_warnings(record=True):
            roma_instance.compute(selected_gene_sets=['EMPTY'])
    
    def test_sparse_matrix_handling(self, roma_instance):
        """Test sparse matrix support."""
        # Create sparse data
        X_sparse = sparse.random(100, 500, density=0.1)
        adata_sparse = sc.AnnData(X=X_sparse)
        adata_sparse.var_names = [f'Gene_{i}' for i in range(500)]
        
        roma_instance.adata = adata_sparse
        roma_instance.genesets = {'SPARSE_TEST': np.array(['Gene_0', 'Gene_1', 'Gene_2'])}
        
        # Should work without errors
        roma_instance.compute(selected_gene_sets=['SPARSE_TEST'], iters=5)
        
        assert 'ROMA_stats' in roma_instance.adata.uns


def run_tests():
    """Run all tests with coverage report."""
    import subprocess
    import sys
    
    # Install pytest and coverage if not available
    required = ['pytest', 'pytest-cov']
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Run tests with coverage
    cmd = [
        sys.executable, '-m', 'pytest', 
        'test_pyroma.py', 
        '-v',  # Verbose
        '--cov=pyroma',  # Coverage for pyroma module
        '--cov-report=term-missing',  # Show missing lines
        '--cov-report=html',  # Generate HTML report
        '-x',  # Stop on first failure
        '--tb=short'  # Short traceback
    ]
    
    subprocess.run(cmd)


if __name__ == '__main__':
    # Quick test runner
    print("Running PyROMA tests...")
    run_tests()