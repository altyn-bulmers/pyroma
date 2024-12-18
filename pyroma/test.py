def assess_significance(self, results):
    """
    Computes the empirical p-value based on the null distribution of L1 scores, L1/L2 ratios, and PC1 means.
    Adjusts p-values using the Benjamini-Hochberg procedure.
    """
    from statsmodels.stats.multitest import multipletests
    import numpy as np
    from scipy.stats import wilcoxon

    # Initialize lists to store p-values and test statistics
    p_values_l1 = []
    p_values_l1l2 = []
    p_values_pc1_mean = []
    wilcoxon_p_values_l1 = []
    wilcoxon_p_values_l1l2 = []
    wilcoxon_p_values_pc1_mean = []
    test_stats = []

    # Iterate over gene set results
    for gene_set_name, gene_set_result in results.items():
        # Extract observed statistics
        test_l1 = gene_set_result.svd.explained_variance_ratio_[0]
        #test_l2 = gene_set_result.svd.explained_variance_ratio_[1]
        #test_l1l2 = test_l1 / test_l2
        test_pc1_mean = np.median(gene_set_result.projections_1)

        # Extract null distributions
        null_l1 = gene_set_result.nulll1[0]
        #null_l1l2 = gene_set_result.null_l1_l2_ratio
        null_pc1_mean = gene_set_result.null_pc1_mean

        # Calculate empirical p-values
        p_value_l1 = (np.sum(null_l1 >= test_l1) + 1) / (len(null_l1) + 1)
        #p_value_l1l2 = (np.sum(null_l1l2 >= test_l1l2) + 1) / (len(null_l1l2) + 1)
        p_value_pc1_mean = (np.sum(np.abs(null_pc1_mean) >= np.abs(test_pc1_mean)) + 1) / (len(null_pc1_mean) + 1)

        # Perform Wilcoxon signed-rank tests
        if len(null_l1) > 5:
            _, wilcoxon_p_l1 = wilcoxon(null_l1 - test_l1, alternative='greater')
        else:
            wilcoxon_p_l1 = np.nan

        #if len(null_l1l2) > 5:
        #    _, wilcoxon_p_l1l2 = wilcoxon(null_l1l2 - test_l1l2, alternative='greater')
        #else:
        #    wilcoxon_p_l1l2 = np.nan

        if len(null_pc1_mean) > 5:
            _, wilcoxon_p_pc1_mean = wilcoxon(null_pc1_mean - test_pc1_mean, alternative='greater')
        else:
            wilcoxon_p_pc1_mean = np.nan

        # Store p-values and test statistics
        p_values_l1.append(p_value_l1)
        #p_values_l1l2.append(p_value_l1l2)
        p_values_pc1_mean.append(p_value_pc1_mean)
        wilcoxon_p_values_l1.append(wilcoxon_p_l1)
        #wilcoxon_p_values_l1l2.append(wilcoxon_p_l1l2)
        wilcoxon_p_values_pc1_mean.append(wilcoxon_p_pc1_mean)

        gene_set_result.test_l1 = test_l1
        #gene_set_result.test_l1l2 = test_l1l2
        gene_set_result.test_pc1_mean = test_pc1_mean

    # Combine all p-values for adjustment
    #all_p_values = np.array(p_values_l1 + p_values_l1l2 + p_values_pc1_mean)
    all_p_values = np.array(p_values_l1 + p_values_pc1_mean)
    _, adjusted_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')

    # Split adjusted p-values back into their respective categories
    n = len(results)
    adjusted_p_values_l1 = adjusted_p_values[:n]
    adjusted_p_values_l1l2 = adjusted_p_values[n:2*n]
    adjusted_p_values_pc1_mean = adjusted_p_values[2*n:]

    # Assign adjusted p-values back to results
    for i, (_, gene_set_result) in enumerate(results.items()):
        gene_set_result.adjusted_p_l1 = adjusted_p_values_l1[i]
        #gene_set_result.adjusted_p_l1l2 = adjusted_p_values_l1l2[i]
        gene_set_result.adjusted_p_pc1_mean = adjusted_p_values_pc1_mean[i]
        gene_set_result.empirical_p_l1 = p_values_l1[i]
        #gene_set_result.empirical_p_l1l2 = p_values_l1l2[i]
        gene_set_result.empirical_p_pc1_mean = p_values_pc1_mean[i]
        gene_set_result.wilcoxon_p_l1 = wilcoxon_p_values_l1[i]
        #gene_set_result.wilcoxon_p_l1l2 = wilcoxon_p_values_l1l2[i]
        gene_set_result.wilcoxon_p_pc1_mean = wilcoxon_p_values_pc1_mean[i]

    return results

#### 

def assess_significance(self, results):
    """
    Computes empirical p-values and performs multiple testing correction.
    
    Parameters:
        results (dict): Dictionary of results per gene set
        
    Returns:
        dict: Updated results with p-values and statistics
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    import numpy as np
    
    ps = np.zeros(shape=len(results)) 
    qs = np.zeros(shape=len(results))
    
    for i, (_, gene_set_result) in enumerate(results.items()):
        # Get null distributions
        null_l1_dist = gene_set_result.nulll1[0]  
        null_median_dist = gene_set_result.null_median_exp

        # L1 statistics
        test_l1 = gene_set_result.svd.explained_variance_ratio_[0]
        
        # Calculate empirical p-value for L1
        _, wilcoxon_p_l1 = wilcoxon(null_l1_dist - test_l1, 
                                   alternative='two-sided',
                                   method='exact')
        ps[i] = wilcoxon_p_l1
        
        # Store L1 test statistic
        gene_set_result.test_l1 = test_l1

        # Median Expression statistics
        test_median_exp, projections_1, projections_2 = self.compute_median_exp(
            gene_set_result.svd,
            gene_set_result.X
        )
        
        # Calculate empirical p-value for median expression
        _, wilcoxon_p_med = wilcoxon(null_median_dist - test_median_exp,
                                    alternative='greater')
        qs[i] = wilcoxon_p_med
        
        # Store results
        gene_set_result.test_median_exp = test_median_exp
        gene_set_result.projections_1 = projections_1
        gene_set_result.projections_2 = projections_2

    # Multiple testing correction using B-H
    _, adjusted_ps = multipletests(ps, method='fdr_bh')[:2]
    _, adjusted_qs = multipletests(qs, method='fdr_bh')[:2]

    # Store adjusted and raw p-values in results
    for i, (_, gene_set_result) in enumerate(results.items()):
        gene_set_result.p_value = adjusted_ps[i]
        gene_set_result.non_adj_p = ps[i]
        gene_set_result.q_value = adjusted_qs[i]
        gene_set_result.non_adj_q = qs[i]

    return results

def p_values_in_frame(self, assessed_results):
    """
    Converts results to pandas DataFrame.
    
    Parameters:
        assessed_results (dict): Results with p-values
        
    Returns:
        pd.DataFrame: DataFrame with test statistics and p-values
    """
    import pandas as pd

    p_dict = {}
    l1_dict = {}
    q_dict = {}
    median_exp_dict = {}
    non_adj_L1_p_values = {}
    non_adj_Med_Exp_p_values = {}
    
    for k, v in assessed_results.items():
        l1_dict[k] = v.test_l1
        p_dict[k] = v.p_value
        median_exp_dict[k] = v.test_median_exp
        q_dict[k] = v.q_value
        non_adj_L1_p_values[k] = v.non_adj_p
        non_adj_Med_Exp_p_values[k] = v.non_adj_q

    df = pd.DataFrame()
    df['L1'] = pd.Series(l1_dict)
    df['ppv L1'] = pd.Series(non_adj_L1_p_values) 
    df['Median Exp'] = pd.Series(median_exp_dict)
    df['ppv Med Exp'] = pd.Series(non_adj_Med_Exp_p_values)
    df['q L1'] = pd.Series(p_dict)
    df['q Med Exp'] = pd.Series(q_dict)
    
    return df


### Claude from rROMA
def randomset_parallel(self, subsetlist, outliers, verbose=1, prefer_type='processes', 
                      incremental=False, iters=100, partial_fit=False, algorithm='randomized'):
    """
    Calculates scores for random gene sets and returns null distributions.
    
    Parameters:
        subsetlist: List of genes in current set
        outliers: List of outlier indices
        verbose: Print progress
        prefer_type: Parallel processing type
        incremental: Use incremental PCA
        iters: Number of iterations
        partial_fit: Use partial_fit for iPCA
        algorithm: SVD algorithm type
        
    Returns:
        Updates self.null_distributions with computed distributions
    """
    from joblib import Parallel, delayed
    import time
    import numpy as np
    
    start = time.time()
    
    # Get null geneset size from filtered set
    candidate_nullgeneset_size = self.nullgenesetsize
    
    # Check if distribution exists for this size
    if candidate_nullgeneset_size in self.null_distributions:
        self.nulll1, self.null_median_exp = self.null_distributions[candidate_nullgeneset_size]
        if verbose:
            print('Using existing null distribution')
        return
        
    # Setup parallel processing
    sequence = np.arange(self.adata.shape[1])
    idx = self.adata.var.index.to_numpy()

    # Run parallel iterations
    results = Parallel(n_jobs=-1, prefer=prefer_type)(
        delayed(self.process_iteration)(sequence, idx, iteration, incremental, 
                                      partial_fit, algorithm) 
        for iteration in range(iters)
    )

    # Unpack results
    nulll1, null_median_exp, null_projections_1, null_projections_2 = zip(*results)
    
    # Convert to arrays
    nulll1_array = np.array(nulll1)
    null_median_exp = np.array(null_median_exp)
    null_projections_1 = np.array(null_projections_1)
    null_projections_2 = np.array(null_projections_2)
    null_projections = np.stack((null_projections_1, null_projections_2), axis=1)
    
    # Store results
    self.null_distributions[candidate_nullgeneset_size] = [
        np.copy(nulll1_array), 
        np.copy(null_median_exp)
    ]
    self.nulll1 = np.copy(nulll1_array)
    self.null_median_exp = np.copy(null_median_exp)
    self.null_projections = np.copy(null_projections)

    if verbose:
        end = time.time()
        elapsed = end - start
        minutes, seconds = divmod(elapsed, 60)
        print(f"Running time: {int(minutes):02}:{seconds:05.2f}")

def process_iteration(self, sequence, idx, iteration, incremental, partial_fit, algorithm):
    """
    Process single iteration of random sampling.
    
    Returns:
        tuple: (l1_score, median_exp, projections_1, projections_2)
    """
    np.random.seed(iteration)
    
    # Sample random genes
    subset = np.random.choice(sequence, self.nullgenesetsize, replace=False)
    gene_subset = np.array([x for i, x in enumerate(idx) if i in subset])
    
    # Detect outliers
    outliers = self.loocv(self.adata[:,[x for x in gene_subset]], 
                         for_randomset=True)
    
    # Compute SVD/PCA
    if incremental:
        svd_, X = self.robustIncrementalPCA(self.adata, gene_subset, outliers,
                                           for_randomset=True, 
                                           partial_fit=partial_fit)
    else:
        svd_, X = self.robustTruncatedSVD(self.adata, gene_subset, outliers,
                                         for_randomset=True,
                                         algorithm=algorithm)
    
    # Get scores
    l1 = svd_.explained_variance_ratio_
    median_exp, null_projections_1, null_projections_2 = self.compute_median_exp(svd_, X)

    return l1, median_exp, null_projections_1, null_projections_2


### Claude
import numpy as np
from scipy import stats
from typing import Tuple, List, Optional
import pandas as pd

class RomaStatistics:
    def __init__(self, n_permutations: int = 1000, random_state: Optional[int] = None):
        """
        Initialize ROMA statistics calculator.
        
        Args:
            n_permutations: Number of permutations for null distribution
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(random_state)
        
    def compute_pc_scores(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Compute L1 (variance explained by PC1) and L1/L2 ratio scores.
        
        Args:
            data: Gene expression matrix (genes x samples)
            
        Returns:
            Tuple of (L1 score, L1/L2 ratio)
        """
        # Center the data
        data_centered = data - np.mean(data, axis=1).reshape(-1, 1)
        
        # Compute covariance matrix
        cov_matrix = np.cov(data_centered)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.real(eigenvalues)  # Get real parts
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
        
        # Calculate L1 (variance explained by PC1)
        total_variance = np.sum(eigenvalues)
        L1 = eigenvalues[0] / total_variance if total_variance > 0 else 0
        
        # Calculate L1/L2 ratio
        L1_L2_ratio = (eigenvalues[0] / eigenvalues[1]) if len(eigenvalues) > 1 and eigenvalues[1] > 0 else np.inf
        
        return L1, L1_L2_ratio
    
    def generate_null_distribution(self, 
                                 data: np.ndarray, 
                                 module_size: int) -> Tuple[List[float], List[float]]:
        """
        Generate null distributions for L1 and L1/L2 through random sampling.
        
        Args:
            data: Full gene expression matrix
            module_size: Size of the gene module to test
            
        Returns:
            Tuple of (L1 null distribution, L1/L2 null distribution)
        """
        L1_null = []
        L1_L2_null = []
        
        n_genes = data.shape[0]
        
        for _ in range(self.n_permutations):
            # Randomly sample genes
            random_indices = self.rng.choice(n_genes, size=module_size, replace=False)
            random_module = data[random_indices, :]
            
            # Compute scores for random module
            L1, L1_L2 = self.compute_pc_scores(random_module)
            
            L1_null.append(L1)
            L1_L2_null.append(L1_L2)
            
        return L1_null, L1_L2_null
    
    def compute_significance(self, 
                           module_data: np.ndarray, 
                           full_data: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute statistical significance of module overdispersion and coordination.
        
        Args:
            module_data: Expression data for genes in the module
            full_data: Full expression dataset
            
        Returns:
            Tuple of (L1 score, L1 p-value, L1/L2 ratio, L1/L2 p-value)
        """
        # Compute actual scores
        L1, L1_L2 = self.compute_pc_scores(module_data)
        
        # Generate null distributions
        L1_null, L1_L2_null = self.generate_null_distribution(
            full_data, 
            module_data.shape[0]
        )
        
        # Compute empirical p-values
        p_value_L1 = np.mean(np.array(L1_null) >= L1)
        p_value_L1_L2 = np.mean(np.array(L1_L2_null) >= L1_L2)
        
        return L1, p_value_L1, L1_L2, p_value_L1_L2

def example_usage():
    """Example of how to use the RomaStatistics class."""
    # Generate example data
    n_genes, n_samples = 1000, 50
    rng = np.random.RandomState(42)
    
    # Create full dataset
    full_data = rng.normal(size=(n_genes, n_samples))
    
    # Create a module with some correlation structure
    module_size = 20
    module_data = rng.normal(size=(module_size, n_samples))
    module_data += rng.normal(size=(1, n_samples))  # Add common signal
    
    # Initialize and run statistical testing
    roma_stats = RomaStatistics(n_permutations=1000, random_state=42)
    L1, p_val_L1, L1_L2, p_val_L1_L2 = roma_stats.compute_significance(
        module_data, 
        full_data
    )
    
    print(f"Results:")
    print(f"L1 Score: {L1:.3f} (p-value: {p_val_L1:.3f})")
    print(f"L1/L2 Ratio: {L1_L2:.3f} (p-value: {p_val_L1_L2:.3f})")

if __name__ == "__main__":
    example_usage()