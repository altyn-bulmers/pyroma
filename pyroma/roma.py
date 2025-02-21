import numpy as np
import time
from scipy import stats
import scanpy as sc
import multiprocessing
from .utils import *

### TODO: solve the confusion btwn namings: q values (which are adj p-values) and p_values()
### Should be: p values L1, p values Med Exp, q values L1, q values Med Exp

class ROMA:
    
    # TODO in plotting : handle many genesets, heatmap (?) 
    from .plotting import plotting as pl 
    #TODO: initialize pl.adata with roma.adata
    pl = pl()
    
    def __init__(self, computation_mode="bulk"):
        """
        Parameters:
          computation_mode: str, "bulk" (CPU/Sklearn-based computations)
                            or "sc" (for single-cell data, using GPU via JAX).
        """
        self.computation_mode = computation_mode  # "bulk" or "sc"
        self.adata = None
        self.gmt = None
        self.genesets = {}
        self.idx = None
        self.approx_int = 20 # Granularity of the null geneset size, from 0 to 100, less is more precise
        self.min_n_genes = 10
        self.nullgenesetsize = None
        self.subset = None
        self.subsetlist = None
        self.outliers = []
        self.svd = None
        self.X = None
        self.raw_X_subset = None
        self.nulll1 = []
        self.results = {}
        self.null_distributions = {}
        manager = multiprocessing.Manager()
        self.parallel_results = manager.dict()
        self.custom_name = color.BOLD + color.GREEN + ('scROMA' if self.computation_mode == 'sc' else 'pyROMA') + color.END
        self.q_L1_threshold=0.05 
        self.q_Med_Exp_threshold=0.05
        # params for fix_pc_sign
        self.gene_weights = {}
        self.pc_sign_mode = 'PreferActivation'  # Mode for PC1 sign correction: 'UseAllWeights', 'UseMeanExpressionAllWeights'
        self.pc_sign_thr = 0.90  # Threshold for extreme weights
        self.def_wei = 1  # Default weight for missing weights
        self.cor_method = 'pearson'  # Correlation method

        # Attributes for gene signs and extreme percentage
        self.gene_signs = {}  # Dictionary to store gene signs per gene set
        self.extreme_percent = 0.1  # Hyperparameter for extreme weights percentage

    def __repr__(self) -> str:
        return self.custom_name
    
    def __str__(self) -> str:
        return self.custom_name

    import warnings
    warnings.filterwarnings("ignore") #worked to supperss the warning message about copying the dataframe

    def read_gmt_to_dict(self, gmt):
        # gmt = an absolute path to .gmt file 
        genesets = {}
        
        file_name = f'{gmt}'
        
        with open(file_name, 'r') as file:
            lines = [line.rstrip('\n') for line in file]

        for line in lines:
            geneset = line.split('\t')
            name = geneset[0]
            genesets[name] = geneset[2:]
            
        for k, v in genesets.items():
            genesets[k] = np.array([gene for gene in v if gene != ''])
        self.genesets = genesets
        return genesets
        
    def indexing(self, adata):
        idx = adata.var.index.tolist()
        idx_set = set(idx)
        self.idx = list(idx_set)
        return 
    
    def subsetting(self, adata, geneset, verbose=0):
        #adata
        #returns subset and subsetlist

        if verbose:
            print(' '.join(x for x in geneset))
        
        # TODO: errors if idx is not there
        #idx = adata.var.index.tolist()
        #idx_set = set(idx)
        
        if not self.idx: 
            print('No adata idx detected in ROMA')
        # self.idx must be a list 
        
        #subsetlist = list(set(idx) & set(geneset))
        
        subsetlist = geneset[np.isin(geneset, self.idx)]
        subset = adata[:, subsetlist]
        self.subset = subset
        self.subsetlist = subsetlist
        return subset, subsetlist
    
    def double_mean_center_matrix(self, matrix):
        # Calculate the overall mean of the matrix
        overall_mean = np.mean(matrix)
        
        # Calculate row means and column means
        row_means = np.mean(matrix, axis=1, keepdims=True)
        col_means = np.mean(matrix, axis=0, keepdims=True)
        
        # Center the matrix
        centered_matrix = matrix - row_means - col_means + overall_mean
        
        return centered_matrix

    def loocv(self, subset, verbose=0, for_randomset=False):
        # TODO: incremental PCA if it's used in the main coompute function
        
        from sklearn.decomposition import TruncatedSVD
        from sklearn.model_selection import LeaveOneOut

        # Since the ROMA computes PCA in sample space the matrix needs to be transposed
        X = subset.X.T
        #X = X - X.mean(axis=0)
        X = np.asarray(X)

        n_samples, n_features = X.shape

        if n_samples < 2:
            # If there are fewer than 2 samples, we can't perform LOOCV
            if verbose:
                print(f"Cannot perform LOOCV with {n_samples} samples.")
            return []

        l1scores = []
        svd = TruncatedSVD(n_components=1, algorithm='randomized', n_oversamples=2)

        loo = LeaveOneOut()
        for train_index, _ in loo.split(X):
            svd.fit(X[train_index])
            l1 = svd.explained_variance_ratio_[0]
            l1scores.append(l1)

        if len(l1scores) > 1:
            u = np.mean(l1scores)
            std = np.std(l1scores)
            zmax = 3
            zvalues = [(x - u) / std for x in l1scores]
            outliers = [i for i, z in enumerate(zvalues) if abs(z) > zmax]
        else:
            outliers = []

        if verbose:
            print(f"Number of samples: {n_samples}, Number of features: {n_features}")
            print(f"Number of outliers detected: {len(outliers)}")

        return outliers


    def robustTruncatedSVD(self, adata, subsetlist, outliers, for_randomset=False, algorithm='randomized'):
        from sklearn.decomposition import TruncatedSVD

        # TODO: here we can calculate the average proportion of the outliers 
        # updating the avg score by each iteration...
        if for_randomset:
            subset = [x for i, x in enumerate(subsetlist)]
            # here calculate the proportion (outliers variable per iteration comes from loocv)
            #self.outliers_avg_proportion += len(outliers)/len(subsetlist)
            #self.outliers_avg_proportion /= 2 
        else:    
            subset = [x for i, x in enumerate(subsetlist) if i not in outliers]
        subset_adata = adata[:, [x for x in subset]]

        # Omitting the centering of the subset to obtain global centering: 
        #subset_adata.X = subset_adata.X - subset_adata.X.mean(axis=1, keepdims=True)
        #matrix = subset_adata.X.T
        #row_means = np.mean(matrix, axis=1, keepdims=True)
        #X = matrix - row_means
        #X = self.double_mean_center_matrix(matrix).T

        X = np.asarray(subset_adata.X.T) 
        # Compute the SVD of X without the outliers
        svd = TruncatedSVD(n_components=2, algorithm=algorithm)#, n_oversamples=2) #algorithm='arpack')
        svd.fit(X)
        #svd.explained_variance_ratio_ = (s ** 2) / (X.shape[0] - 1)
        if not for_randomset:
            #print(svd.singular_values_)
            self.svd = svd
            self.X = X
        return svd, X

    def robustPCA(self, adata, subsetlist, outliers, for_randomset=False, algorithm='auto'):
        from sklearn.decomposition import PCA

        # TODO: here we can calculate the average proportion of the outliers 
        # updating the avg score by each iteration...
        if for_randomset:
            subset = [x for i, x in enumerate(subsetlist)]
            # here calculate the proportion (outliers variable per iteration comes from loocv)
            #self.outliers_avg_proportion += len(outliers)/len(subsetlist)
            #self.outliers_avg_proportion /= 2 
        else:    
            subset = [x for i, x in enumerate(subsetlist) if i not in outliers]
        subset = adata[:, [x for x in subset]]

        # Omitting the centering of the subset to obtain global centering: 
        #X = subset.X - subset.X.mean(axis=0)
        X = np.asarray(subset.X.T) 
        # Compute the SVD of X without the outliers
        svd = PCA(n_components=2, svd_solver=algorithm) #algorithm='arpack')
        svd.fit(X)

        if not for_randomset:
            self.svd = svd
            self.X = X
        return svd, X

    def robustIncrementalPCA(self, adata, subsetlist, outliers, for_randomset=False, partial_fit=False):
        
        #TODO: make the batch size as a global hyperparameter
        from sklearn.decomposition import IncrementalPCA

        outliers = outliers or []
        # Exclude outliers from the subset list
        subset = [x for i, x in enumerate(subsetlist) if i not in outliers]
        subset = adata[:, [x for x in subset]]

        # Omitting the centering of the subset to obtain global centering: 
        # Center the data by subtracting the mean
        #X = subset.X - subset.X.mean(axis=0)
        # Since the ROMA computes PCA in sample space the matrix needs to be transposed
        X = subset.X.T
        X = np.asarray(X.T) # normally it shouldn't be transpose - double checking for rROMA

        # Initialize IncrementalPCA for 1 component
        svd = IncrementalPCA(n_components=2, batch_size=1000)
        if partial_fit:
            svd.partial_fit(X)
        else:            
            svd.fit(X)
        
        # Store in the object if not for_randomset
        if not for_randomset:
            self.svd = svd
            self.X = X
        return svd, X

    def truncated_robustGPUSVD(self, adata, subsetlist, outliers, for_randomset=False):
        from scipy.sparse import issparse
        import jax.numpy as jnp

        # Select the subset of genes: if not for_randomset, remove outliers.
        if for_randomset:
            subset = [x for i, x in enumerate(subsetlist)]
        else:
            subset = [x for i, x in enumerate(subsetlist) if i not in outliers]
        subset_adata = adata[:, subset].copy()
        
        # Convert the expression matrix to a JAX array.
        # Note: We assume that the expression matrix is stored such that rows are genes and columns are samples.
        # Therefore, we transpose to have shape (n_samples, n_genes).
        if issparse(subset_adata.X):
            subset_adata.X = subset_adata.X.todense()

        X = jnp.array(subset_adata.X.T, dtype=jnp.float32)

        # Define a helper function to compute a truncated SVD for the first k components using power iteration.
        def truncated_svd(X, k=2, num_iters=200, eps=1e-6):
            n, m = X.shape
            U_list = []
            s_list = []
            V_list = []
            X_deflated = X
            for i in range(k):
                # Initialize a random vector (here simply ones) of size m.
                v = jnp.ones((m,))
                v = v / (jnp.linalg.norm(v) + eps)
                # Power iteration loop.
                for _ in range(num_iters):
                    u = X_deflated @ v
                    u = u / (jnp.linalg.norm(u) + eps)
                    v = X_deflated.T @ u
                    v = v / (jnp.linalg.norm(v) + eps)
                # Compute the corresponding singular value.
                sigma = jnp.dot(u, X_deflated @ v)
                U_list.append(u)
                s_list.append(sigma)
                V_list.append(v)
                # Deflate the matrix to remove the contribution of the computed singular triplet.
                X_deflated = X_deflated - sigma * jnp.outer(u, v)
            U = jnp.stack(U_list, axis=1)   # shape: (n, k)
            s = jnp.stack(s_list, axis=0)     # shape: (k,)
            Vh = jnp.stack(V_list, axis=0)    # shape: (k, m)
            return U, s, Vh

        # Compute the truncated SVD (first 2 components).
        U, s, Vh = truncated_svd(X, k=2, num_iters=100)
        # The total variance equals the squared Frobenius norm of X.
        total_variance = jnp.sum(X ** 2)
        explained_variance_ratio_ = (s ** 2) / total_variance

        # Create a lightweight object to mimic the scikit-learn SVD output.
        class GPUSVD:
            def __init__(self, components_, explained_variance_ratio_):
                self.components_ = components_
                self.explained_variance_ratio_ = explained_variance_ratio_
        
        svd = GPUSVD(components_=Vh, explained_variance_ratio_=explained_variance_ratio_)
        if not for_randomset:
            self.svd = svd
            self.X = X
        return svd, X

    

    def robustGPUSVD(self, adata, subsetlist, outliers, for_randomset=False):
        """
        Compute the SVD on GPU using jax.scipy.linalg.svd with float64 precision.
        This method returns the first 2 components, ensuring consistency with the CPU version.
        """
        from scipy.sparse import issparse
        import jax.numpy as jnp
        import jax.scipy.linalg as jsp_linalg
        import numpy as np

        # Select the subset of genes: if not for_randomset, remove outliers.
        if for_randomset:
            subset = subsetlist
        else:
            subset = [x for i, x in enumerate(subsetlist) if i not in outliers]
        subset_adata = adata[:, subset]

        # Convert the expression matrix to a dense NumPy array with float64 precision.
        if issparse(subset_adata.X):
            denseX = subset_adata.X.toarray()
        else:
            denseX = np.asarray(subset_adata.X)
        denseX = np.asarray(denseX, dtype=np.float64)

        # We assume that the matrix is stored such that rows are genes and columns are samples,
        # so transpose it to have shape (n_samples, n_genes).
        X = jnp.array(denseX.T, dtype=jnp.float64)
        n_samples = X.shape[0]
        
        print("shape of X: ", X.shape)
        # Use the robust full SVD from jax.scipy.linalg.svd.
        U, s, Vh = jsp_linalg.svd(X, full_matrices=True)
        # Select only the first 2 components.
        U2 = U[:, :2]    # shape: (n_samples, 2)
        s2 = s[:2]       # shape: (2,)
        Vh2 = Vh[:2, :]  # Vh2 has shape (2, n_genes)
        #s = jnp.sort(s, descending=True) #not necessary

        # Compute total variance and explained variance ratio for the first component.
        #variances = s**2 / (X.shape[0]-1)
        #explained_variances = variances / np.sum(variances)
        #total_variance = jnp.sum(s ** 2)
        #explained_variance_ratio = (s[0] ** 2 ) / total_variance
        #explained_variance = (s2[0] ** 2) / (n_samples - 1)
        #total_variance = jnp.sum(jnp.var(X, axis=0, ddof=1))
        #L1_ratio = explained_variance / total_variance
        #explained_variance_ratio = L1_ratio
        
        # Compute the transformed data as X_transformed = X dot Vh2.T.
        X_transformed = X @ Vh2.T   # shape: (n_samples, 2)
        # Calculate explained variance per component.
        exp_var = jnp.var(X_transformed, axis=0)
        # Compute full variance from X.
        full_var = jnp.var(X, axis=0).sum()
        explained_variance_ratio = exp_var / full_var
        
        #print("exp var ratio: ", explained_variance_ratio)
        #print("singular values: ", s2)

        # Create a lightweight SVD object to mimic scikit-learn’s SVD output.
        class GPUSVD:
            def __init__(self, components_, explained_variance_ratio_):
                self.components_ = components_
                self.explained_variance_ratio_ = explained_variance_ratio_
        svd = GPUSVD(components_=Vh2, explained_variance_ratio_=explained_variance_ratio)
        
        if not for_randomset:
            self.svd = svd
            self.X = X
        return svd, X


    def fix_pc_sign(self, GeneScore, SampleScore, Wei=None, Mode='none', DefWei=1,
                    Thr=None, Grouping=None, ExpMat=None, CorMethod="pearson",
                    gene_set_name=None):
        """
        Python equivalent of the R FixPCSign function.

        Parameters:
            GeneScore (np.ndarray): Array of gene scores (PC loadings).
            SampleScore (np.ndarray): Array of sample scores (PC projections).
            Wei (np.ndarray or None): Gene weights array aligned with GeneScore.
            Mode (str): Mode to correct the sign (e.g., 'none', 'PreferActivation', ...).
            DefWei (float): Default weight for missing weights.
            Thr (float or None): Quantile threshold or p-value threshold depending on context.
            Grouping (dict, pd.Series, or None): Maps sample names to groups.
            ExpMat (numpy.array or None): Expression matrix (genes x samples).
            CorMethod (str): One of "pearson", "spearman", "kendall".

        Returns:
            int: +1 or -1 indicating the orientation of the PC.
        """
        
        

        #import numpy as np
        #import pandas as pd
        from scipy.stats import pearsonr, spearmanr, kendalltau
        from scipy.sparse import issparse
        import os
        
        output_dir = '/home/az/Projects/01_Curie/06.1_pyROMA_Sofia_results/pyroma_debug/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Helper functions
        
        def apply_threshold_to_genescore(gscore, thr):
            # Apply quantile thresholding as in R code:
            # abs(gscore) >= quantile(abs(gscore), Thr)
            q_val = np.quantile(np.abs(gscore), thr)
            return np.abs(gscore) >= q_val

        def correlation_function(method):
            if method == 'pearson':
                return pearsonr
            elif method == 'spearman':
                return spearmanr
            elif method == 'kendall':
                return kendalltau
            else:
                raise ValueError("Invalid CorMethod. Choose 'pearson', 'spearman', or 'kendall'.")

        def cor_test(x, y, method, thr):
            """
            Emulate cor.test logic:
            If Thr is not None, we return (pvalue, estimate).
            If Thr is None, we return (nan, correlation) because no test is required per the original R code.
            """
            func = correlation_function(method)
            corr, pval = func(x, y)
            if Thr is None:
                # No p-value thresholding in R means just return correlation with no p-value
                return (np.nan, corr)
            else:
                # With Thr, we actually consider p-value from the test
                return (pval, corr)

        def print_msg(msg):
            # Just print messages as R code does
            print(msg)

        # Ensure Wei is a numpy array if provided
        if Wei is not None:
            Wei = np.asarray(Wei, dtype=float)
        else:
            Wei = np.full_like(GeneScore, np.nan)

        # MODE: 'none'
        if Mode == 'none':
            print_msg("Orienting PC using a random direction")
            return 1

        # MODE: 'PreferActivation'
        if Mode == 'PreferActivation':
            print_msg("Orienting PC by preferential activation")
            ToUse = np.full(len(GeneScore), True, dtype=bool)
            if Thr is not None:
                ToUse = apply_threshold_to_genescore(GeneScore, Thr)

            if np.sum(GeneScore[ToUse]) < 0:
                return -1
            else:
                return 1

        # MODE: 'UseAllWeights'
        if Mode == 'UseAllWeights':
            print_msg(f"Missing gene weights will be replaced by {DefWei}")
            Wei = np.where(np.isnan(Wei), DefWei, Wei)
            Mode = 'UseKnownWeights'

        # MODE: 'UseKnownWeights'
        if Mode == 'UseKnownWeights':
            print_msg("Orienting PC by combining PC weights and gene weights")
            print_msg("Missing gene weights will be replaced by 0")
            Wei = np.where(np.isnan(Wei), 0, Wei)

            ToUse = np.full(len(GeneScore), True)
            if Thr is not None:
                ToUse = apply_threshold_to_genescore(GeneScore, Thr)

            mask = (~np.isnan(Wei)) & ToUse
            if np.sum(mask) < 1:
                print_msg("Not enough weights, PC will be oriented randomly")
                return 1

            if np.sum(Wei[mask] * GeneScore[mask]) < 0:
                return -1
            else:
                return 1

        # MODE: 'CorrelateAllWeightsByGene'
        if Mode == 'CorrelateAllWeightsByGene':
            print_msg(f"Missing gene weights will be replaced by {DefWei}")
            Wei = np.where(np.isnan(Wei), DefWei, Wei)
            Mode = 'CorrelateKnownWeightsByGene'

        # MODE: 'CorrelateKnownWeightsByGene'
        if Mode == 'CorrelateKnownWeightsByGene':
            print_msg(f"Orienting PC by correlating gene expression and sample score ({CorMethod})")

            if np.sum(~np.isnan(Wei)) < 1:
                print_msg("Not enough weights, PC will be oriented randomly")
                return 1

            if Grouping is not None:
                print_msg("Using groups")
                group_series = pd.Series(Grouping)
                # Check if we have enough groups
                TB = group_series.value_counts()
                if (TB > 0).sum() < 2:
                    print_msg("Not enough groups, PC will be oriented randomly")
                    return 1

                # Subset ExpMat to genes with non-NA Wei
                SelGenesWei_mask = ~np.isnan(Wei)
                SelGenes = ExpMat.index[SelGenesWei_mask]
                # Compute group medians for each gene
                # GroupMedians: for each gene, median expression by group
                # We'll store a dict of Series: gene -> Series of medians by group
                GroupMedians = {}
                for gene in SelGenes:
                    df_gene = pd.DataFrame({'val': ExpMat.loc[gene, :].values,
                                            'Group': group_series[ExpMat.columns].values})
                    median_by_group = df_gene.groupby('Group')['val'].median()
                    GroupMedians[gene] = median_by_group

                # MedianProj: median of SampleScore by group
                df_score = pd.DataFrame({'Score': SampleScore, 'Group': group_series[ExpMat.columns].values})
                MedianProj = df_score.groupby('Group')['Score'].median()

                print_msg("Computing correlations")
                # We must compute correlations for each gene (with non-NA Wei) between gene group medians and MedianProj
                # Cor.Test.Vect: 2D structure. We'll store in arrays:
                # We'll store p-values and estimates for each gene
                gene_list = list(GroupMedians.keys())
                pvals = []
                cors = []
                for gene in gene_list:
                    x = GroupMedians[gene]
                    # Align with MedianProj:
                    # Ensure same groups in both
                    common_groups = x.index.intersection(MedianProj.index)
                    if len(common_groups) < 3:
                        # Not enough data to correlate meaningfully
                        pvals.append(np.nan)
                        cors.append(np.nan)
                        continue

                    gx = x.loc[common_groups].values
                    gy = MedianProj.loc[common_groups].values
                    pval, cor_est = cor_test(gx, gy, CorMethod, Thr)
                    pvals.append(pval)
                    cors.append(cor_est)

                CorTestVect = np.array([pvals, cors])

                # Apply weights
                SelGenesWei = Wei[SelGenesWei_mask]
                # Align SelGenesWei with gene_list
                # gene_list are the selected genes in order. SelGenes is also in order of ExpMat index
                # Assume order matches ExpMat indexing:
                # We'll create a map gene->index to ensure alignment is correct
                # In R code: names(SelGenesWei) <- names(GroupMedians)
                # Just assume alignment by order:
                # CorTestVect[2,:] is correlation estimates for these genes
                # Now we filter by Wei and correlation:
                ToUse = ~np.isnan(CorTestVect[1, :])
                if Thr is not None:
                    # p-value threshold: (CorTestVect[0,i] < Thr)
                    ToUse = (CorTestVect[0, :] < Thr) & ToUse

                # Weighted sum of correlations:
                # Only use genes with not NA Wei
                if np.sum(ToUse) == 0:
                    # No usable genes
                    return 1

                weighted_sum = np.sum(CorTestVect[1, ToUse] * SelGenesWei[ToUse])
                if weighted_sum > 0:
                    return 1
                else:
                    return -1

            else:
                print_msg("Not using groups")
                # names(SampleScore) <- colnames(ExpMat)
                # Compute correlation gene by gene:
                print_msg("Computing correlations")
                pvals = []
                cors = []
                for i, gene_val in enumerate(ExpMat):
                    # gene_val is expression vector of a gene across samples
                    # Filter if enough variation:
                    if len(np.unique(gene_val)) <= 2 or len(np.unique(SampleScore)) <= 2:
                        pvals.append(np.nan)
                        cors.append(np.nan)
                        continue
                    pval, cor_est = cor_test(gene_val, SampleScore, CorMethod, Thr)
                    pvals.append(pval)
                    cors.append(cor_est)

                CorTestVect = np.array([pvals, cors])

                print_msg("Correcting using weights")
                # If sum(!is.na(Wei))>1 then multiply corresponding correlations by Wei
                non_na_wei_mask = ~np.isnan(Wei)
                if np.sum(non_na_wei_mask) > 1:
                    CorTestVect[1, non_na_wei_mask] = CorTestVect[1, non_na_wei_mask] * Wei[non_na_wei_mask]

                ToUse = ~np.isnan(CorTestVect[1, :])
                if Thr is not None:
                    ToUse = (CorTestVect[0, :] < Thr) & ToUse

                if np.sum(CorTestVect[1, ToUse]) > 0:
                    return 1
                else:
                    return -1

        # MODE: 'CorrelateAllWeightsBySample'
        if Mode == 'CorrelateAllWeightsBySample':
            print_msg(f"Missing gene weights will be replaced by {DefWei}")
            Wei = np.where(np.isnan(Wei), DefWei, Wei)
            Mode = 'CorrelateKnownWeightsBySample'

        # MODE: 'CorrelateKnownWeightsBySample'
        if Mode == 'CorrelateKnownWeightsBySample':
            print_msg(f"Orienting PC by correlating gene expression and PC weights ({CorMethod})")

            if np.sum(~np.isnan(Wei)) < 1:
                print_msg("Not enough weights, PC will be oriented randomly")
                return 1

            # GeneScore * Wei
            WeightedGeneScore = np.copy(GeneScore)
            WeightedGeneScore[np.isnan(Wei)] = 0
            WeightedGeneScore = WeightedGeneScore * Wei

            if Grouping is not None:
                print_msg("Using groups")
                group_series = pd.Series(Grouping)
                TB = group_series.value_counts()
                if (TB > 0).sum() < 2:
                    print_msg("Not enough groups, PC will be oriented randomly")
                    return 1

                # Compute group medians of each gene:
                # GroupMedians: apply(ExpMat, 1, function(x) aggregate by group median)
                # We'll create a 3D structure is complicated. In R:
                # They do `GroupMedians <- apply(ExpMat, 1, function(x) {aggregate(...)})`
                # This returns a list of data frames per gene. We only need final correlation:
                # Actually, we need median expression per group for each gene?

                # Actually "CorrelateKnownWeightsBySample" block:
                # GroupMedians <- apply(ExpMat, 1, function(x) {
                #   aggregate(x, by=list(AssocitedGroups), FUN=median)
                # })
                # Then they sapply something and do correlations by row:
                # Finally they do correlation by groups of the medianed data vs GeneScore*Wei

                # Let's do: For each gene, median by group:
                # But they then do correlation per group row. Actually they do:
                # Correlation is applied row-wise to MediansByGroups:
                # MediansByGroups is a matrix with each row a group and each column a gene median?
                # In R code:
                # GroupMedians: a list for each gene. Then sapply returns a matrix (like pivot)
                # We'll construct a matrix (Groups x Genes) of median expression:
                gene_names = ExpMat.index
                sample_names = ExpMat.columns
                df_long = ExpMat.copy()
                df_long = df_long.T  # samples x genes
                df_long['Group'] = group_series[sample_names].values
                # Compute median by group for each gene
                median_by_group = df_long.groupby('Group').median()  # DataFrame with groups as rows, genes as columns

                # Now median_by_group is (Groups x Genes)
                # They do correlation row-wise with GeneScore*Wei:
                # We want to correlate each row of median_by_group (across genes) with WeightedGeneScore
                # WeightedGeneScore is per gene. So we correlate across genes:
                print_msg("Computing correlations")
                pvals = []
                cors = []
                for i in range(median_by_group.shape[0]):
                    x = median_by_group.iloc[i, :].values  # median expression of all genes in this group
                    # Correlate x with WeightedGeneScore
                    # Check if enough variation:
                    if len(np.unique(x)) > 2 and len(np.unique(WeightedGeneScore)) > 2:
                        pval, cor_est = cor_test(x, WeightedGeneScore, CorMethod, Thr)
                        pvals.append(pval)
                        cors.append(cor_est)
                    else:
                        pvals.append(np.nan)
                        cors.append(np.nan)

                CorTestVect = np.array([pvals, cors])
                ToUse = ~np.isnan(CorTestVect[1, :])
                if Thr is not None:
                    ToUse = (CorTestVect[0, :] < Thr) & ToUse

                if np.sum(CorTestVect[1, ToUse]) > 0:
                    return 1
                else:
                    return -1

            else:
                print_msg("Not using groups")

                # names(GeneScore) <- rownames(ExpMat)
                # WeightedGeneScore = GeneScore*Wei done above

                # In R code: They do correlation column-wise:
                # Cor.Test.Vect <- apply(ExpMat, 2, function(x){cor.test(x, GeneScore*Wei)})
                # So we correlate each sample (column) expression vector with WeightedGeneScore

                # Actually, at "CorrelateKnownWeightsBySample" last part:
                # They do:
                #   names(GeneScore) <- rownames(ExpMat)
                #   GeneScore <- GeneScore*Wei
                #   Cor.Test.Vect <- apply(ExpMat, 2, function(x) cor.test(x, GeneScore))
                # This means we correlate each sample's expression (vertical vector from ExpMat) with WeightedGeneScore across genes.

                pvals = []
                cors = []
                # apply(ExpMat, 2, ...) means column-wise in R, so each column is a sample
                for col in ExpMat.columns:
                    x = ExpMat[col].values  # gene expression in this sample
                    # Correlate x with WeightedGeneScore across genes
                    if len(np.unique(x)) > 2 and len(np.unique(WeightedGeneScore)) > 2:
                        pval, cor_est = cor_test(x, WeightedGeneScore, CorMethod, Thr)
                        pvals.append(pval)
                        cors.append(cor_est)
                    else:
                        pvals.append(np.nan)
                        cors.append(np.nan)

                CorTestVect = np.array([pvals, cors])
                ToUse = ~np.isnan(CorTestVect[1, :])
                if Thr is not None:
                    ToUse = (CorTestVect[0, :] < Thr) & ToUse

                if np.sum(CorTestVect[1, ToUse]) > 0:
                    return 1
                else:
                    return -1

        # MODE: 'UseMeanExpressionAllWeights'
        if Mode == 'UseMeanExpressionAllWeights':
            print_msg(f"Missing gene weights will be replaced by {DefWei}")
            Wei = np.where(np.isnan(Wei), DefWei, Wei)
            Mode = 'UseMeanExpressionKnownWeights'

        # MODE: 'UseMeanExpressionKnownWeights'
        if Mode == 'UseMeanExpressionKnownWeights':
            if np.sum(~np.isnan(Wei)) < 1:
                print_msg("Not enough weights, PC will be oriented randomly")
                return 1
            if ExpMat is None:
                print_msg("ExpMat not specified, PC will be oriented randomly")
                return 1

            ToUse = np.full(len(GeneScore), True)
            if Thr is not None:
                # (GeneScore >= max(quantile(GeneScore, Thr),0)) | (GeneScore <= min(0, quantile(GeneScore,1-Thr)))
                q_thr_high = np.quantile(GeneScore, Thr)
                q_thr_low = np.quantile(GeneScore, 1 - Thr)
                ToUse = (GeneScore >= max(q_thr_high, 0)) | (GeneScore <= min(0, q_thr_low))
                

            nbUsed = np.sum(ToUse)
            if nbUsed < 2:
                if nbUsed == 1:
                    # In R code: ExpMat <- scale(apply(ExpMat, 1, median), center=TRUE, scale=FALSE)
                    # apply(ExpMat,1,median) gives a median per gene: a vector of length=ngenes
                    # scale(...) center=TRUE means subtract mean
                    #median_per_gene = ExpMat.median(axis=1).values
                    if issparse(ExpMat):
                        median_per_gene = ExpMat.median(axis=1).A.flatten()
                    else:
                        median_per_gene = ExpMat.median(axis=1).values
                    centered_median = median_per_gene - np.mean(median_per_gene)
                    val = np.sum(GeneScore[ToUse] * Wei[ToUse] * centered_median[ToUse])
                    if val > 0:
                        return 1
                    else:
                        return -1
                else:
                    print_msg("No weight considered, PC will be oriented randomly")
                    return 1

            # For nbUsed >= 2:
            # ExpMat[ToUse, ] means subset genes
            if self.computation_mode == 'bulk':
                subset_mat = ExpMat[ToUse, :]
            if self.computation_mode == 'sc':
                if issparse(ExpMat):
                    subset_mat = ExpMat[ToUse, :].toarray()
                else:
                    subset_mat = np.atleast_2d(ExpMat[ToUse, :])

            row_medians = np.median(subset_mat, axis=1)
            centered_medians = np.array(row_medians - np.mean(row_medians))
            centered_medians = centered_medians.reshape(1, -1)
            val = np.sum(GeneScore[ToUse]*Wei[ToUse]*centered_medians)
            #centered_medians = centered_medians.reshape(-1, 1)
            
            output_file = f'{output_dir}/{gene_set_name}.txt' 

            with open(output_file, "w") as f:
                f.write(f"Module: {gene_set_name}\n")
                if centered_medians is not None:
                    shape = centered_medians.shape
                    #print(type(shape), type(centered_medians))
                    #print('shape', shape)
                    f.write(f"ExpMat Head: {centered_medians}\n")
                    #f.write(f"Subset Mat: {subset_mat}\n")
                    f.write(f"ExpMat Shape: {shape[0]} x {shape[1]}\n")
                    
                else:
                    f.write("ExpMat Shape: N/A\n")
                f.write(f"Raw ExpMat shape: {ExpMat.shape[0]} x {ExpMat.shape[1]}\n")
                f.write(f"Raw ExpMat Head: {ExpMat[:5, :5]}\n")
                f.write(f"GeneScore Shape: {len(GeneScore)}\n")
                f.write(f"Gene Score: {GeneScore}\n")
                f.write(f"GeneScore[ToUse] Shape: {len(GeneScore[ToUse])}\n")
                f.write(f"GeneScore[ToUse]: {GeneScore[ToUse]}\n")
                if val is not None:
                    f.write(f"Computed val: {val}\n")
                else:
                    f.write("Computed val: N/A\n")

                if val > 0:
                    f.write(f"Fix PC Sign output: 1" )
                    return 1
                else:
                    f.write(f"Fix PC Sign output: -1" )
                    return -1

        if Mode == 'UseExtremeWeights':
            """
            This mode:
            - Finds the top Thr fraction of genes by absolute PC weight.
            - Multiplies those gene weights by the (mean) gene expression across samples.
            - Sums the products. If the sum < 0 => flip sign (-1), else +1.
            """
            print_msg("Orienting PC by using the most extreme PC weights and gene expression.")
            if ExpMat is None:
                print_msg("No ExpMat provided. Orientation will be random.")
                return 1
            
            # Step 1: Identify the top/bottom fraction of genes by abs(PC weight)
            if Thr is None:
                # If Thr is None, define some default or return random orientation
                print_msg("No Thr provided. Using entire set of genes.")
                ToUse = np.full(len(SampleScore), True, dtype=bool)
            else:
                cutoff = np.quantile(np.abs(SampleScore), 0.15) #1 - Thr)
                ToUse = (np.abs(SampleScore) >= cutoff)

            # Step 2: Compute the average expression of each gene across samples
            #   ExpMat shape: (genes x samples)
            #   so the mean expression of gene i => np.mean(ExpMat[i, :])
            print('ok', end=' | ')
            gene_means = np.mean(ExpMat, axis=1) #1
            print('gene_means', end=' | ')
            # Step 3: Sum up (GeneScore[i] * gene_means[i]) for the selected genes
            sum_val = np.sum(SampleScore[ToUse] * gene_means[ToUse])
            print('sum_val')

            # Step 4: If sum < 0, flip sign
            if sum_val > 0:
                return 1
            else:
                return -1
        # If none of the above conditions matched, default:
        return 1


    
    def orient_pc1(self, pc1, X, raw_X_subset, gene_set_name=None):
        """
        Orient PC1 according to the methods described.
        """
        # Get gene scores (loadings) and sample scores (projections)
        sample_score = pc1
        gene_score = X @ pc1
        #gene_score = raw_X_subset @ pc1
        # Get gene weights if available
        wei = self.gene_weights.get(gene_set_name, None)
        #print("wei: ", wei)
        # exp_mat is data (genes x samples)
        print(f"GeneScore shape: {gene_score.shape}")
        print(f"ExpMat shape: {raw_X_subset.shape}")
        #print("Outliers: ", self.outliers)

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
        
            
    def compute_median_exp(self, svd_, X, raw_X_subset, gene_set_name=None):
        """
        Computes the shifted pathway 
        """

        #if X is None or X.shape[0] == 0:
        #    print(f"Warning: X is empty for gene set {gene_set_name}, returning default values.")
        #    return np.nan, np.array([]), np.array([])

        pc1, pc2 = svd_.components_
        #raw_median_exp = np.median(X @ pc1)
        # Orient PC1
        correct_sign = self.orient_pc1(pc1, X, raw_X_subset, gene_set_name=gene_set_name)
        pc1 = correct_sign * pc1
        
        projections_1 = X @ pc1 # the scores that each gene have in the sample space
        #print(f"Raw X shape: {X.shape}")
        projections_2 = X @ pc2
        #print('shape of projections should corresponds to n_genes', projections.shape)
        # Compute the median of the projections
        median_exp = np.median(projections_1) 
        # TODO: is median expression is calculated only with the pc1 projections?
        return median_exp, projections_1, projections_2 #TODO: save gene scores after pc sign correction


    def process_iteration(self, sequence, idx, iteration, incremental, partial_fit, algorithm):
        """
        Iteration step for the randomset calculation
        """
        ### ?
        #np.random.seed(42) # this is suggested to add
        
        subset = np.random.choice(sequence, self.nullgenesetsize, replace=False)
        gene_subset = np.array([x for i, x in enumerate(idx) if i in subset])
        
        outliers = self.loocv(self.adata[:,[x for x in gene_subset]], for_randomset=True)
        if incremental:
            svd_, X = self.robustIncrementalPCA(self.adata, gene_subset, outliers, for_randomset=True, partial_fit=partial_fit)
        else:    
            svd_, X = self.robustTruncatedSVD(self.adata, gene_subset, outliers, for_randomset=True, algorithm=algorithm)
            
        l1 = svd_.explained_variance_ratio_
        subsetlist = [x for i, x in enumerate(gene_subset) if i not in outliers]
        #raw_X_subset = self.adata.raw[:, subsetlist].X.T.copy()
        median_exp, null_projections_1, null_projections_2 = self.compute_median_exp(svd_, X, self.raw_X_subset)

        return l1, median_exp, null_projections_1, null_projections_2

    def process_iteration_gpu(self, sequence, idx, iteration):
        subset = np.random.choice(sequence, self.nullgenesetsize, replace=False)
        gene_subset = np.array([x for i, x in enumerate(idx) if i in subset])
        outliers = self.loocv(self.adata[:, [x for x in gene_subset]], for_randomset=True)
        svd_, X = self.robustGPUSVD(self.adata, gene_subset, outliers, for_randomset=True)
        l1 = svd_.explained_variance_ratio_
        subsetlist = [x for i, x in enumerate(gene_subset) if i not in outliers]
        median_exp, null_projections_1, null_projections_2 = self.compute_median_exp(svd_, X, self.raw_X_subset)
        return l1, median_exp, null_projections_1, null_projections_2

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
        
        # Choose which process_iteration to use based on computation_mode.
        if self.computation_mode == "sc":
            results = Parallel(n_jobs=-1, prefer=prefer_type)(
                delayed(self.process_iteration_gpu)(sequence, idx, iteration) for iteration in range(iters)
            )
        else:
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
        
        return
        
    def wilcoxon_assess_significance(self, results):
        
        ### rROMA like
        ### the correlation of p-values from R and py versions is low
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
            null_l1_dist = gene_set_result.nulll1[:,0]  
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

    def assess_significance(self, results):
       # TODO: output the median of null_L1 distribution
       # TODO: incorporate an option to compute p-values via wilcoxon 
        """
        Computes the empirical p-value based on the null distribution of L1 scores and median expression.
        Adjust p-values and q-values using the Benjamini-Hochberg procedure.
        """
        from scipy.stats import false_discovery_control as benj_hoch
        from statsmodels.stats.multitest import multipletests
        import numpy as np
        from scipy.stats import wilcoxon
        
        #valid_results = {name: result for name, result in results if result is not None}
        ps = np.zeros(shape=len(results))
        qs = np.zeros(shape=len(results))
        for i, (_, gene_set_result) in enumerate(results.items()):
            #print('PRINTING to fix the ERROR', gene_set_result.nulll1)
            #print('NULL MEDIAN EXP', gene_set_result.null_median_exp)
            if np.ndim(gene_set_result.nulll1) == 1:
                null_distribution = gene_set_result.nulll1
            else:
                null_distribution = gene_set_result.nulll1[:,0]
            null_median_distribution = gene_set_result.null_median_exp
            print('shape of the null distribution: ', null_distribution.shape)
            print('shape of null med distribution: ', null_median_distribution.shape)

            # L1 statistics
            if self.computation_mode == 'sc':
                test_l1 = np.asarray(gene_set_result.svd.explained_variance_ratio_[0]).item()
            else:
                test_l1 = gene_set_result.svd.explained_variance_ratio_[0]
            #print(type(test_l1))
            #p_value = (np.sum(null_distribution >= test_l1) + 1) / (len(null_distribution) + 1) # original value
            p_value = np.mean(np.array(null_distribution) >= test_l1)
            # changing that to wilcoxon as in rROMA
            #_, wilcoxon_p_l1 = wilcoxon(null_distribution - test_l1, alternative='two-sided', method='exact')
            #p_value = wilcoxon_p_l1
            

            # otherwise p_value could be calculated with (np.sum(null_distribution >= test_l1)) / (len(null_distribution))
            ps[i] =  p_value #if p_value <= 1.0 else 1.0
            gene_set_result.test_l1 = test_l1

            gene_set_name = gene_set_result.custom_name.split(maxsplit=1)[-1]
            # Median Exp statistic

            test_median_exp, projections_1, projections_2 = self.compute_median_exp(gene_set_result.svd, gene_set_result.X, gene_set_result.raw_X_subset, gene_set_name)
            #test_median_exp, projections_1, projections_2 = self.compute_median_exp(gene_set_result.svd, gene_set_result.X, gene_set_name)
            q_value = (np.sum(np.abs(null_median_distribution) >= np.abs(test_median_exp)) + 1) / (len(null_median_distribution) + 1)
            #q_value = (np.sum((null_median_distribution) >=(test_median_exp)) + 1) / (len(null_median_distribution) + 1)
            
            # from the rROMA 
            #_, wilcoxon_p_pc1_mean = wilcoxon(null_median_distribution - test_median_exp, alternative='greater')
            #q_value = wilcoxon_p_pc1_mean
            
            qs[i] = q_value
            gene_set_result.test_median_exp = test_median_exp
            gene_set_result.projections_1 = projections_1
            gene_set_result.projections_2 = projections_2


        #print('raw p-values', ps)
        #print('raw q-values', qs)
        adjusted_ps = benj_hoch(ps)
        adjusted_qs = benj_hoch(qs)
        # confirm the same lengths of lists
        #print('Lengths of ps and adj_ps match and match the adj_qs', len(ps) == len(adjusted_ps), len(adjusted_ps) == len(adjusted_qs) )
        
        #all_p_values = np.array(ps + qs) # if they're lists
        #all_p_values =  np.concatenate((ps,qs))
        #print('All p Values shape ', all_p_values.shape)
        #_, adjusted_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
        
        #print('All adjusted p Values shape ', adjusted_p_values.shape)


        #n = len(results)
        #adjusted_ps = adjusted_p_values[:n]
        #adjusted_qs = adjusted_p_values[n:]

        for i, (_, gene_set_result) in enumerate(results.items()):
            gene_set_result.p_value = adjusted_ps[i]
            gene_set_result.non_adj_p = ps[i]
            gene_set_result.q_value = adjusted_qs[i]
            gene_set_result.non_adj_q = qs[i]
        return results
    

    def approx_size(self, flag):
        """
        Approximate size
        For current subset and gene set -> we compute the null gene set size
        add it to the dictionary of null gene set sizes
        for the next one, we calculate if the closest size in dictionary is smaller by k(approx_int) to ours
        if smaller -> we just use the same distribution from the dictionary (as it is computed)
        is larger -> we create a new 
        """
        candidate_nullgeneset_size = sum(1 for i in range(len(self.subsetlist)) if i not in self.outliers)

        if flag:
            # just add to the self.null_distributions
            # update the self.nullgenesetsize
            self.nullgenesetsize = candidate_nullgeneset_size
        else:
            for k in self.null_distributions:
                if abs(k - candidate_nullgeneset_size) <= self.approx_int:
                    self.nullgenesetsize = k
                    return
            # update the self.nullgenesetsize for randomset_parallel()
            # in randomset_parallel just take the nullgeneset value
            self.nullgenesetsize = candidate_nullgeneset_size 
        return
    
    class GeneSetResult:
        def __init__(self, subset, subsetlist, outliers, nullgenesetsize, svd, X, raw_X_subset, nulll1, null_median_exp, null_projections):
            self.subset = subset
            self.subsetlist = subsetlist
            self.outliers = outliers
            self.nullgenesetsize = nullgenesetsize
            self.svd = svd
            self.X = X
            self.raw_X_subset = raw_X_subset
            self.projections_1 = None
            self.projections_2 = None
            self.nulll1 = nulll1
            self.null_median_exp = null_median_exp
            self.null_projections = null_projections
            self.p_value = None
            self.q_value = None
            self.non_adj_p = None
            self.non_adj_q = None
            self.test_l1 = None
            self.test_median_exp = None
        
        def __repr__(self):
            return self.custom_name

        def __str__(self):
            return self.custom_name

    
    def select_and_sort_gene_sets(self, selected_geneset_names):
        # Select gene sets that are in my_gene_sets
        selected_gene_sets = {name: genes for name, genes in self.genesets.items() if name in selected_geneset_names}

        # Sort the selected gene sets based on the number of genes (from lower to higher)
        sorted_gene_sets = sorted(selected_gene_sets.items(), key=lambda x: len(x[1]))

        # Return the sorted list of gene set names
        return [name for name, _ in sorted_gene_sets]

    def p_values_in_frame(self, assessed_results):
        """
        Puts all the values into pandas dataframe
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




    def center_sparse(self, X):
        # checked
        """
        Center a sparse matrix X (n_samples x n_features) by:
        1. Subtracting the row-wise mean (making each row zero-mean).
        2. Subtracting the column-wise mean (making each column zero-mean).
        
        Returns a centered sparse matrix.
        """
        import numpy as np
        import scipy.sparse as sp
        
        n_samples, n_features = X.shape

        # Compute row means and reshape to column vector (n_samples, 1)
        row_means = np.array(X.mean(axis=1)).flatten().reshape(-1, 1)
        # Create a sparse matrix where each row is the row mean replicated n_features times
        row_mean_matrix = sp.csr_matrix(row_means).dot(sp.csr_matrix(np.ones((1, n_features))))
        
        # Subtract row means
        X_centered = X - row_mean_matrix

        # Convert to CSC for efficient column operations
        X_centered = X_centered.tocsc()

        # Compute column means and reshape to row vector (1, n_features)
        col_means = np.array(X_centered.mean(axis=0)).flatten().reshape(1, -1)
        # Create a sparse matrix where each column is the column mean replicated n_samples times
        col_mean_matrix = sp.csc_matrix(np.ones((n_samples, 1))).dot(sp.csc_matrix(col_means))
        
        # Subtract column means
        X_centered = X_centered - col_mean_matrix

        return X_centered.tocsr()

    

    def broken_randomset_parallel_gpu(self, key, iters=100):
        """
        GPU-parallel computation of null distributions via JAX vectorization.
        
        Parameters:
        key: a JAX PRNGKey.
        iters: number of iterations.
        
        This function assumes that:
        - self.X is a dense JAX array (e.g. produced by robustGPUSVD) of shape (n_samples, n_features)
        - self.nullgenesetsize is an integer indicating how many columns (genes) to sample.
        
        It vectorizes the following per-iteration operations:
        1. Randomly sample a subset of gene indices (without replacement) using jax.random.permutation.
        2. Extract the corresponding submatrix.
        3. Compute a truncated SVD (here, for the first 2 components) using power iteration.
        4. Compute the explained variance ratio of the first component.
        5. Compute the median of the PC1 projections.
        
        The function returns (and stores) the null distributions.
        """
        import jax
        import jax.numpy as jnp
        import jax.scipy.linalg as jsp_linalg
        #import numpy as np
        
        n_samples, n_features = self.adata.X.shape
        print('n_features', n_features)
        nullsize = self.nullgenesetsize  # number of genes in the null set
        
        def one_iteration(key):
            # Generate a random permutation over gene indices and take the first 'nullsize' indices.
            perm = jax.random.permutation(key, n_features)
            selected = perm[:nullsize]
            # Extract the subset (n_samples x nullsize).
            selected_np = np.asarray(selected) 
            X_subset = self.adata.X[:, selected_np]
            X_subset = jnp.array(X_subset.toarray()) 
            #X_subset_jax = jax.numpy.asarray(X_subset.toarray())  

            print('null subset shape:', X_subset.shape)
            # Compute truncated SVD on X_subset.
            #U, s, Vh = truncated_svd(X_subset, k=2, num_iters=50)
            U, s, Vh = jsp_linalg.svd(X_subset, full_matrices=False)
            # Compute the transformed data as X_transformed = X_subset @ Vh.T
            X_transformed = X_subset @ Vh.T  # shape: (n_samples, 2)
            # Compute explained variance for each component.
            exp_var = jnp.var(X_transformed, axis=0)  # shape: (2,)
            # Compute full variance of X_subset.
            full_var = jnp.var(X_subset, axis=0).sum()
            # L1 explained variance ratio.
            expl_ratio = exp_var[0] / full_var
            # For projections, we use the first component loadings.
            pc1 = Vh[0, :]  # shape: (nullsize,)
            projections = X_subset @ pc1  # shape: (n_samples,)
            med_exp = jnp.median(projections)
            return expl_ratio, med_exp, projections

        # Split the key into 'iters' subkeys.
        keys = jax.random.split(key, iters)
        # Vectorize the one_iteration function over the keys.
        expl_ratios, med_exps, projections_all = jax.vmap(one_iteration)(keys)
        
        #print("null explained ratios", expl_ratios)

        # Optionally store these in your object:
        self.nulll1 = expl_ratios  # Null distribution of explained variance ratios.
        self.null_median_exp = med_exps
        self.null_projections = projections_all  # This will be a 2D array: (iters, n_samples)
        
        return expl_ratios, med_exps, projections_all

    def randomset_parallel_gpu(self, key, iters=100):
        """
        GPU-parallel computation of null distributions via batching.
        
        Instead of calling jax.vmap directly on a function that performs sparse indexing
        (which causes tracer conversion errors), we first generate concrete random subsets
        and extract the corresponding dense submatrices from self.adata.X.
        
        Then we stack these submatrices (shape: (iters, n_samples, nullsize)) and use jax.vmap
        to compute, for each, the explained variance ratio (L1 score) and median of PC1 projections,
        computed in the same way as in scikit-learn's TruncatedSVD.
        
        Assumes:
        - self.X is NOT used here; instead, we use self.adata.X (a scipy sparse matrix)
        - self.nullgenesetsize is an integer (nullsize)
        """
        import jax
        import jax.numpy as jnp
        import jax.scipy.linalg as jsp_linalg
        import numpy as np

        n_samples, n_features = self.adata.X.shape  # self.adata.X is sparse (genes x samples)
        # Note: In our SVD routines we assume X is (n_samples, n_genes) so we will transpose later.
        # Here, self.adata.X.shape is (n_genes, n_samples). 
        # We therefore define:
        n_genes = n_features  # if self.adata.X.shape is (n_genes, n_samples)
        nullsize = self.nullgenesetsize  # number of genes (columns in the transposed X) to sample
        # TODO: to compute on parallel cores
        # Step 1: Generate concrete (non-traced) random subsets of gene indices.
        # We work in the gene domain (indices from 0 to n_genes-1).
        keys = jax.random.split(key, iters)
        selected_indices = []
        for k in keys:
            # Convert the key to a concrete integer seed:
            seed = int(jax.device_get(k)[0])
            # Generate a random permutation using numpy (with the same seed)
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_genes)
            selected = perm[:nullsize]  # concrete numpy array of indices
            selected_indices.append(selected)
        selected_indices = np.stack(selected_indices, axis=0)  # shape: (iters, nullsize)

        # Step 2: For each set of indices, extract the corresponding columns from self.adata.X,
        # then transpose so that the resulting dense submatrix has shape (n_samples, nullsize).
        # Since self.adata.X is sparse (shape: (n_genes, n_samples)), we index rows here.
        # (Remember: in the full pipeline we use X = (n_samples, n_genes), so here we extract and then transpose.)
        X_subsets = []
        for inds in selected_indices:
            # Extract rows corresponding to the selected gene indices.
            # self.adata.X is sparse, so we use .toarray() on the resulting submatrix.
            X_subset = self.adata.X[inds, :].toarray()  # shape: (nullsize, n_samples)
            X_subsets.append(X_subset.T)  # transpose to shape: (n_samples, nullsize)
        X_subsets = np.stack(X_subsets, axis=0)  # shape: (iters, n_samples, nullsize)

        # Convert the batched dense submatrices to a JAX array with float64 precision.
        X_subsets = jnp.array(X_subsets, dtype=jnp.float64)

        # Step 3: Define a function that computes the explained variance ratio in the same way as TruncatedSVD.
        def compute_metrics(X_subset):
            # X_subset is of shape (n_samples, nullsize)
            # Compute full SVD on the subset.
            U, s, Vh = jsp_linalg.svd(X_subset, full_matrices=False)
            # Compute the transformed data: X_transformed = X_subset @ Vh.T  (shape: (n_samples, 2))
            X_transformed = X_subset @ Vh.T
            # Compute explained variance for each component as variance along axis 0.
            exp_var = jnp.var(X_transformed, axis=0)  # shape: (k,), here k= nullsize? But we only need first 2.
            # In practice, we only computed full SVD; we now consider only the first component.
            # Compute full variance of X_subset: sum of variance across features.
            full_var = jnp.var(X_subset, axis=0).sum()
            # Explained variance ratio (L1) is the variance of the first component divided by full variance.
            expl_ratio = exp_var[0] / full_var
            # For projections, we use the first component loadings from Vh.
            pc1 = Vh[0, :]  # shape: (nullsize,)
            projections = X_subset @ pc1  # shape: (n_samples,)
            med_exp = jnp.median(projections)
            return expl_ratio, med_exp, projections

        # Step 4: Vectorize the metric computation over the first axis (each batch element).
        expl_ratios, med_exps, projections_all = jax.vmap(compute_metrics)(X_subsets)

        # Optionally store these in your object:
        self.nulll1 = expl_ratios  # shape: (iters,)
        self.null_median_exp = med_exps  # shape: (iters,)
        self.null_projections = projections_all  # shape: (iters, n_samples)

        return expl_ratios, med_exps, projections_all


    def batched_randomset_parallel_gpu(self, key, total_iters=100, batch_size=100):
        """
        Run the GPU parallel randomset computation in batches.
        
        Parameters:
        key: A JAX PRNGKey.
        total_iters: Total number of iterations desired.
        batch_size: Number of iterations to run in each batch.
        
        Returns:
        Combined null distribution arrays.
        """

        import jax
        import numpy as np
        import gc

        num_batches = total_iters // batch_size
        all_expl_ratios = []
        all_med_exps = []
        all_projections = []
        current_key = key
        for i in range(num_batches):
            # Split the key for this batch.
            current_key, subkey = jax.random.split(current_key)
            # Run a batch of iterations.
            expl_ratios, med_exps, projections = self.randomset_parallel_gpu(subkey, iters=batch_size)
            # Convert results to numpy arrays and store.
            all_expl_ratios.append(np.array(expl_ratios))
            all_med_exps.append(np.array(med_exps))
            all_projections.append(np.array(projections))
            # Clear JAX caches and run garbage collection to free GPU memory.
            jax.clear_caches()
            gc.collect()
            print(f"Completed batch {i+1}/{num_batches}")
        # Concatenate the batch results.
        all_expl_ratios = np.concatenate(all_expl_ratios)
        all_med_exps = np.concatenate(all_med_exps)
        all_projections = np.concatenate(all_projections, axis=0)
        return all_expl_ratios, all_med_exps, all_projections


    def compute(self, selected_gene_sets, parallel=False, incremental=False, iters=100, partial_fit=False, algorithm='randomized', loocv_on=True, double_mean_centering=False):
        #pl.adata = self.adata
        """
        Computes ROMA
        min_n_genes = 10 (default) minimum geneset size of genes present in the provided dataset.
        approx_int = 20 (default) granularity of the null geneset size, 
                    from 0 to 100, what is the minimum distance in the n of genes between sizes of the genesets.  
        
        """
        from scipy import sparse
        limit_preallocation = False
        # Handle sparse adata.X for efficiency:
        if sparse.issparse(self.adata.X):
            if self.computation_mode == "sc":
                print("adata.X is sparse")#; converting to dense for GPU computation.")
                limit_preallocation = True
            else:
                print("adata.X is sparse, convert to dense for bulk computation")
                self.adata.X = self.adata.X.toarray()
                #self.adata.X = sparse.csr_matrix(self.adata.X)
        else:
            if self.computation_mode == "sc":
                limit_preallocation = True
                print("adata.X is not sparse, converting it")
                self.adata.X = sparse.csr_matrix(self.adata.X)
        
        # pre allocation
        if limit_preallocation:
            import os
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        results = {}
        
        # Centering expression of each gene in the global matrix, copying the original in adata.raw
        # Centering over samples (genes will have 0 mean)
        # in rROMA columns are samples, 
        # and the "scale" function centering is done by subtracting the column means of x from their corresponding columns
        self.adata.raw = self.adata.copy()
        X = self.adata.X.T 
        #X_raw = X.copy()
        
        # centering
        if self.computation_mode == "sc":
            X_centered = self.center_sparse(X)
        else:
            if double_mean_centering:
                # centering across samples and genes
                X_centered = self.double_mean_center_matrix(X)

            else:
                # centering over samples, genes have 0 mean
                # replicates the behavior in R
                X_centered = X - np.mean(X, axis=1, keepdims=True)
                X_centered = X_centered - np.mean(X_centered, axis=0, keepdims=True)

        self.adata.X = X_centered.T 
        
        # for pc sign
        #adata_raw = self.adata.copy()
        #X_centered = X_raw - X_raw.mean(axis=0)
        #adata_raw.X = X_centered.T


        self.indexing(self.adata)
        self.read_gmt_to_dict(self.gmt)

        # to mark the first one
        flag = True
        
        # TODO: handle different selection of genesets 
        if selected_gene_sets == 'all':
            selected_gene_sets = self.genesets.keys()
        else:
            selected_gene_sets = list(set(selected_gene_sets) & set(self.genesets.keys()))

        unprocessed_genesets = []

        # TODO: here we then need to sort the gene sets by their size first
        # Sort the selected genesets by by their size 
        sorted_gene_sets = self.select_and_sort_gene_sets(selected_gene_sets)

        for gene_set_name in sorted_gene_sets:
            print(f'Processing gene set: {color.BOLD}{color.DARKCYAN}{gene_set_name}{color.END}', end=' | ')
            self.subsetting(self.adata, self.genesets[gene_set_name])
            print('len of subsetlist:', color.BOLD, len(self.subsetlist), color.END)
            if len(self.subsetlist) < self.min_n_genes:
                unprocessed_genesets.append(gene_set_name)
                continue
            # TODO: loocv should be also computed on GPU in "sc" compute mode
            if loocv_on:
                self.loocv(self.subset)
            
            self.approx_size(flag)
            flag = False
            
            # SVD
            if self.computation_mode == "sc":
                self.robustGPUSVD(self.adata, self.subsetlist, self.outliers)
            else:
                if incremental:
                    self.robustIncrementalPCA(self.adata, self.subsetlist, self.outliers)
                    #self.robustKernelPCA(self.adata, self.subsetlist, self.outliers)
                else:
                    self.robustTruncatedSVD(self.adata, self.subsetlist, self.outliers, algorithm=algorithm)
                
            # take the raw uncentered X for the fix pc sign calculation 
            # should be genes x samples
            # TODO: include outliers, as they're not considered in the raw subsetting. potential shape mismatch of subset and raw_subset
            subsetlist_no_out = [x for i, x in enumerate(self.subsetlist) if i not in self.outliers]
            self.raw_X_subset = self.adata.raw[:, subsetlist_no_out].X.T.copy()

            # parallelization
            if parallel:
                if self.computation_mode == 'bulk':
                    self.randomset_parallel(self.adata, self.subsetlist, 
                                        self.outliers, prefer_type='processes', incremental=incremental, iters=iters, partial_fit=partial_fit, 
                                        algorithm=algorithm)
                elif self.computation_mode == 'sc':
                    # Run the GPU-based version.
                    import jax
                    key = jax.random.PRNGKey(1)
                    #self.randomset_parallel_gpu(key, iters=iters)
                    self.nulll1, self.null_median_exp, self.null_projections = self.batched_randomset_parallel_gpu(key, total_iters=iters, batch_size=100)
            #print('self.nullgenesetsize', self.nullgenesetsize)
            #print('self.nulll1 :', self.nulll1)
            # Store the results for this gene set in a new instance of GeneSetResult
            
            gene_set_result = self.GeneSetResult(self.subset, self.subsetlist, self.outliers, self.nullgenesetsize, 
                                                 self.svd, self.X , self.raw_X_subset, #instead of raw_X_subset
                                                 self.nulll1, self.null_median_exp, self.null_projections)

            gene_set_result.custom_name = f"GeneSetResult {gene_set_name}"
            # Store the instance of GeneSetResult in the dictionary using gene set name as the key
            results[gene_set_name] = gene_set_result

        #print(' RESULTS:', results)
        # calculate p_value adjusted for multiple-hypotheses testing
        assessed_results = self.assess_significance(results)
        #self.results = assessed_results
        self.adata.uns['ROMA'] = assessed_results
        self.adata.uns['ROMA_stats'] = self.p_values_in_frame(assessed_results)
        self.select_active_modules(self.q_L1_threshold, self.q_Med_Exp_threshold)
        self.unprocessed_genesets = unprocessed_genesets
        self.custom_name = color.BOLD + color.GREEN + 'scROMA' + color.END +': module activities are computed'
        print(color.BOLD, color.PURPLE, 'Finished', color.END, end=': ')
        
        # plotting functions inherit adata from the ROMA class 
        self.pl.adata = self.adata

        return 
    
    def select_active_modules(self, q_L1_threshold=0.05, q_Med_Exp_threshold=0.05):
        """
        Selects the active pathways above the threshold
        """

        df = self.adata.uns['ROMA_stats']
        active_modules = df[(df['q L1'] <= q_L1_threshold) | (df['q Med Exp'] <= q_Med_Exp_threshold)]
        self.adata.uns['ROMA_active_modules'] = active_modules

        return
    

    def _randomset_jax(self, subsetlist, outliers, verbose=1, iters=12):
        import time 
        import jax.numpy as jnp

        nullgenesetsize = sum(1 for i in range(len(subsetlist)) if i not in outliers)
        self.nullgenesetsize = nullgenesetsize
        sequence = np.arange(self.adata.shape[1])
        idx = self.adata.var.index.to_numpy()

        subset = np.random.choice(sequence, self.nullgenesetsize, replace=False)
        #idx = self.adata.var.index.to_numpy()
        gene_subset = np.array([x for i, x in enumerate(idx) if i in subset])
        outliers = self.loocv(self.adata[:,[x for x in gene_subset]], for_randomset=True)
        np.random.seed(iteration)

        #svd_, X = self.robustPCA(self.adata, gene_subset, outliers, for_randomset=True)
        
        for loop_i, x in enumerate(Xs):
            u, s, vt = jnp.linalg.svd(x, full_matrices=False)
            l1, l2 = svd_.explained_variance_ratio_


        if verbose:
            minutes, seconds = divmod(tac - tic, 60)
            print(f"loop {i} time: " + "{:0>2}:{:05.2f}".format(int(minutes), seconds))   

        return 

    
    def save_ROMA_results(adata, path):
        # saves the adata to a path
        import pickle 
        d = adata.uns['ROMA']

        with open(f'{path}.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del adata.uns['ROMA']
        adata.write(f"{path}.h5ad")

        return

    def load_ROMA_results(path):
        # loads the results into adata
        import pickle
        import scanpy as sc 

        with open(f'{path}.pickle', 'rb') as handle:
            d = pickle.load(handle)

        adata = sc.read_h5ad(f'{path}.h5ad')
        adata.uns['ROMA'] = d

        return adata
        
    
    
    