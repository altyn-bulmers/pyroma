import numpy as np
import time
from scipy import stats
import scanpy as sc
import multiprocessing
#from .utils import *

### TODO: solve the confusion btwn namings: q values (which are adj p-values) and p_values()
### Should be: p values L1, p values Med Exp, q values L1, q values Med Exp

class ROMA:
    
    # TODO in plotting : handle many genesets, heatmap (?) 
    #from .plotting import plotting as pl 
    #TODO: initialize pl.adata with roma.adata
    #pl = pl()
    
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
        # TODO: create an option for several gmt files to put in one dict
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


    def orient_pc1(self, pc1, X, raw_X_subset, gene_set_name=None):
        """
        Orient PC1 using GPU Cupy arrays.
        
        Parameters:
        pc1: Cupy 1D array for the principal component (sample score).
        X: Cupy array (e.g. genes x samples) used to compute gene scores.
        raw_X_subset: (Typically on CPU or already converted) used for PC sign correction.
        gene_set_name: Optional gene set name for weight lookup.
        
        Returns:
        +1 or -1 (an integer) indicating the correct orientation.
        """
        # In GPU mode, pc1 (sample score) is a Cupy array.
        sample_score = pc1
        # Compute gene scores (loadings) on GPU
        gene_score = X @ pc1  
        #print(f"GeneScore shape: {gene_score.shape}")
        #print(f"ExpMat shape: {raw_X_subset.shape}")
        
        # Look up gene weights (if provided – assumed to be stored as Cupy arrays)
        wei = self.gene_weights.get(gene_set_name, None)
        
        # Call the GPU-adapted fix_pc_sign routine.
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


    def fix_pc_sign(self, GeneScore, SampleScore, Wei=None, Mode='none', DefWei=1,
                    Thr=None, Grouping=None, ExpMat=None, CorMethod="pearson",
                    gene_set_name=None):
        """
        GPU-adapted version of the FixPCSign function.
        
        All primary numerical operations use Cupy so that data remain on the GPU.
        When using CPU-only routines (e.g. correlation tests or pandas-based grouping)
        the relevant arrays are converted to NumPy.
        
        Parameters:
        GeneScore (cp.ndarray): Cupy array of gene scores (PC loadings).
        SampleScore (cp.ndarray): Cupy array of sample scores (PC projections).
        Wei (cp.ndarray or None): Gene weights (if available).
        Mode (str): Mode of sign correction.
        DefWei (float): Default weight value.
        Thr (float or None): Threshold for gene selection.
        Grouping: (dict or pandas.Series or None) for group-based correlation.
        ExpMat: Expression matrix (for grouping or mean calculations). Could be a pandas DataFrame.
        CorMethod (str): 'pearson', 'spearman', or 'kendall'.
        gene_set_name: Optional gene set name.
        
        Returns:
        int: +1 or -1 indicating the orientation.
        """

        import cupy as cp
        #import numpy as np
        from scipy.stats import pearsonr, spearmanr, kendalltau
        import os

        # Ensure Wei is a Cupy array (or create an array of NaNs matching GeneScore)
        if Wei is not None:
            if not isinstance(Wei, cp.ndarray):
                Wei = cp.asarray(Wei, dtype=cp.float64)
        else:
            Wei = cp.full_like(GeneScore, cp.nan)
        
        # --- Helper functions ---
        def apply_threshold_to_genescore(gscore, thr):
            # Use Cupy quantile and absolute value
            q_val = cp.quantile(cp.abs(gscore), thr)
            return cp.abs(gscore) >= q_val

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
            # x and y here are assumed to be NumPy arrays.
            func = correlation_function(method)
            corr, pval = func(x, y)
            if thr is None:
                return (np.nan, corr)
            else:
                return (pval, corr)

        def print_msg(msg):
            print(msg)

        # --- Begin Mode Branches ---

        # MODE: 'none'
        if Mode == 'none':
            print_msg("Orienting PC using a random direction")
            return 1

        # MODE: 'PreferActivation'
        if Mode == 'PreferActivation':
            print_msg("Orienting PC by preferential activation")
            ToUse = cp.full(GeneScore.shape, True, dtype=bool)
            if Thr is not None:
                ToUse = apply_threshold_to_genescore(GeneScore, Thr)
            # cp.sum returns a Cupy scalar; extract the Python number with .item()
            if cp.sum(GeneScore[ToUse]).item() < 0:
                return -1
            else:
                return 1

        # MODE: 'UseAllWeights'
        if Mode == 'UseAllWeights':
            print_msg(f"Missing gene weights will be replaced by {DefWei}")
            Wei = cp.where(cp.isnan(Wei), DefWei, Wei)
            Mode = 'UseKnownWeights'

        # MODE: 'UseKnownWeights'
        if Mode == 'UseKnownWeights':
            print_msg("Orienting PC by combining PC weights and gene weights")
            print_msg("Missing gene weights will be replaced by 0")
            Wei = cp.where(cp.isnan(Wei), 0, Wei)
            ToUse = cp.full(GeneScore.shape, True, dtype=bool)
            if Thr is not None:
                ToUse = apply_threshold_to_genescore(GeneScore, Thr)
            mask = (~cp.isnan(Wei)) & ToUse
            if cp.sum(mask).item() < 1:
                print_msg("Not enough weights, PC will be oriented randomly")
                return 1
            if cp.sum(Wei[mask] * GeneScore[mask]).item() < 0:
                return -1
            else:
                return 1

        # MODE: 'CorrelateAllWeightsByGene'
        if Mode == 'CorrelateAllWeightsByGene':
            print_msg(f"Missing gene weights will be replaced by {DefWei}")
            Wei = cp.where(cp.isnan(Wei), DefWei, Wei)
            Mode = 'CorrelateKnownWeightsByGene'

        # MODE: 'CorrelateKnownWeightsByGene'
        if Mode == 'CorrelateKnownWeightsByGene':
            print_msg(f"Orienting PC by correlating gene expression and sample score ({CorMethod})")
            # For correlation tests we convert Cupy arrays to NumPy.
            Wei_cpu = cp.asnumpy(Wei) if isinstance(Wei, cp.ndarray) else Wei
            GeneScore_cpu = cp.asnumpy(GeneScore) if isinstance(GeneScore, cp.ndarray) else GeneScore
            SampleScore_cpu = cp.asnumpy(SampleScore) if isinstance(SampleScore, cp.ndarray) else SampleScore

            if Grouping is not None:
                print_msg("Using groups")
                import pandas as pd
                group_series = pd.Series(Grouping)
                TB = group_series.value_counts()
                if (TB > 0).sum() < 2:
                    print_msg("Not enough groups, PC will be oriented randomly")
                    return 1
                # Assume ExpMat is a pandas DataFrame
                SelGenesWei_mask = ~np.isnan(Wei_cpu)
                SelGenes = ExpMat.index[SelGenesWei_mask]
                GroupMedians = {}
                for gene in SelGenes:
                    df_gene = pd.DataFrame({
                        'val': ExpMat.loc[gene, :].values,
                        'Group': group_series[ExpMat.columns].values
                    })
                    median_by_group = df_gene.groupby('Group')['val'].median()
                    GroupMedians[gene] = median_by_group
                df_score = pd.DataFrame({
                    'Score': SampleScore_cpu,
                    'Group': group_series[ExpMat.columns].values
                })
                MedianProj = df_score.groupby('Group')['Score'].median()
                print_msg("Computing correlations")
                gene_list = list(GroupMedians.keys())
                pvals = []
                cors = []
                for gene in gene_list:
                    x = GroupMedians[gene]
                    common_groups = x.index.intersection(MedianProj.index)
                    if len(common_groups) < 3:
                        pvals.append(np.nan)
                        cors.append(np.nan)
                        continue
                    gx = x.loc[common_groups].values
                    gy = MedianProj.loc[common_groups].values
                    pval, cor_est = cor_test(gx, gy, CorMethod, Thr)
                    pvals.append(pval)
                    cors.append(cor_est)
                CorTestVect = np.array([pvals, cors])
                SelGenesWei = Wei_cpu[SelGenesWei_mask]
                ToUse = ~np.isnan(CorTestVect[1, :])
                if Thr is not None:
                    ToUse = (CorTestVect[0, :] < Thr) & ToUse
                if np.sum(CorTestVect[1, ToUse]) > 0:
                    return 1
                else:
                    return -1
            else:
                print_msg("Not using groups")
                print_msg("Computing correlations")
                pvals = []
                cors = []
                # Assume ExpMat is a NumPy array here
                for i, gene_val in enumerate(ExpMat):
                    if len(np.unique(gene_val)) <= 2 or len(np.unique(SampleScore_cpu)) <= 2:
                        pvals.append(np.nan)
                        cors.append(np.nan)
                        continue
                    pval, cor_est = cor_test(gene_val, SampleScore_cpu, CorMethod, Thr)
                    pvals.append(pval)
                    cors.append(cor_est)
                CorTestVect = np.array([pvals, cors])
                print_msg("Correcting using weights")
                non_na_wei_mask = ~np.isnan(Wei_cpu)
                if np.sum(non_na_wei_mask) > 1:
                    CorTestVect[1, non_na_wei_mask] = CorTestVect[1, non_na_wei_mask] * Wei_cpu[non_na_wei_mask]
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
            Wei = cp.where(cp.isnan(Wei), DefWei, Wei)
            Mode = 'CorrelateKnownWeightsBySample'

        # MODE: 'CorrelateKnownWeightsBySample'
        if Mode == 'CorrelateKnownWeightsBySample':
            print_msg(f"Orienting PC by correlating gene expression and PC weights ({CorMethod})")
            if cp.sum(~cp.isnan(Wei)).item() < 1:
                print_msg("Not enough weights, PC will be oriented randomly")
                return 1
            WeightedGeneScore = cp.copy(GeneScore)
            WeightedGeneScore = cp.where(cp.isnan(Wei), 0, WeightedGeneScore)
            WeightedGeneScore = WeightedGeneScore * Wei
            if Grouping is not None:
                print_msg("Using groups")
                import pandas as pd
                group_series = pd.Series(Grouping)
                TB = group_series.value_counts()
                if (TB > 0).sum() < 2:
                    print_msg("Not enough groups, PC will be oriented randomly")
                    return 1
                gene_names = ExpMat.index
                sample_names = ExpMat.columns
                df_long = ExpMat.copy().T  # samples x genes
                df_long['Group'] = group_series[sample_names].values
                median_by_group = df_long.groupby('Group').median()
                print_msg("Computing correlations")
                pvals = []
                cors = []
                # For each group, correlate the median expression (across genes) with the weighted gene score.
                for i in range(median_by_group.shape[0]):
                    x = median_by_group.iloc[i, :].values
                    # Convert WeightedGeneScore to NumPy if needed
                    WGS_cpu = cp.asnumpy(WeightedGeneScore)
                    if len(np.unique(x)) > 2 and len(np.unique(WGS_cpu)) > 2:
                        pval, cor_est = cor_test(x, WGS_cpu, CorMethod, Thr)
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
                pvals = []
                cors = []
                import pandas as pd
                # Here we assume ExpMat is a pandas DataFrame with columns as samples.
                WGS_cpu = cp.asnumpy(WeightedGeneScore)
                for col in ExpMat.columns:
                    x = ExpMat[col].values
                    if len(np.unique(x)) > 2 and len(np.unique(WGS_cpu)) > 2:
                        pval, cor_est = cor_test(x, WGS_cpu, CorMethod, Thr)
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
            Wei = cp.where(cp.isnan(Wei), DefWei, Wei)
            Mode = 'UseMeanExpressionKnownWeights'

        # MODE: 'UseMeanExpressionKnownWeights'
        if Mode == 'UseMeanExpressionKnownWeights':
            if cp.sum(~cp.isnan(Wei)).item() < 1:
                print_msg("Not enough weights, PC will be oriented randomly")
                return 1
            if ExpMat is None:
                print_msg("ExpMat not specified, PC will be oriented randomly")
                return 1
            ToUse = cp.full(GeneScore.shape, True, dtype=bool)
            if Thr is not None:
                q_thr_high = cp.quantile(GeneScore, Thr)
                q_thr_low = cp.quantile(GeneScore, 1 - Thr)
                ToUse = (GeneScore >= cp.maximum(q_thr_high, 0)) | (GeneScore <= cp.minimum(0, q_thr_low))
            nbUsed = cp.sum(ToUse).item()
            if nbUsed < 2:
                if nbUsed == 1:
                    # If ExpMat is a sparse matrix on CPU, leave it; otherwise, assume it is a pandas DataFrame.
                    import scipy.sparse
                    if isinstance(ExpMat, scipy.sparse.spmatrix):
                        median_per_gene = cp.asarray(ExpMat.median(axis=1).A.flatten())
                    else:
                        median_per_gene = cp.asarray(ExpMat.median(axis=1).values)
                    centered_median = median_per_gene - cp.mean(median_per_gene)
                    val = cp.sum(GeneScore[ToUse] * Wei[ToUse] * centered_median[ToUse])
                    if val.item() > 0:
                        return 1
                    else:
                        return -1
                else:
                    print_msg("No weight considered, PC will be oriented randomly")
                    return 1
            # For a proper subset, extract the corresponding rows from ExpMat.
            if self.computation_mode == 'bulk':
                subset_mat = ExpMat[ToUse, :]
            if self.computation_mode == 'sc':
                #import scipy.sparse
                import cupyx.scipy.sparse as cpsp
                #if scipy.sparse.issparse(ExpMat):
                #print("type expmat:", type(ExpMat))
                #print("ExpMat:", ExpMat)
                #print("type ToUse", type(ToUse))    
                ToUse = cp.asnumpy(ToUse)
                if cpsp.issparse(ExpMat):
                    subset_mat = ExpMat[ToUse, :].toarray()
                else:
                    #subset_mat = np.atleast_2d(ExpMat[ToUse, :])
                    subset_mat = ExpMat[ToUse, :]
                #subset_mat = cp.asnumpy(subset_mat)
            # Compute medians on GPU (convert subset_mat to Cupy if needed)
            #print("subset_mat type", type(subset_mat))
            subset_mat_cp_sparse = cpsp.csc_matrix(subset_mat)
            subset_mat_cp = subset_mat_cp_sparse.toarray()
            #print("subset_mat_cp type", type(subset_mat_cp))
            row_medians = cp.median(subset_mat_cp, axis=1)
            centered_medians = row_medians - cp.mean(row_medians)
            centered_medians = centered_medians.reshape(1, -1)
            val = cp.sum(GeneScore[ToUse] * Wei[ToUse] * centered_medians)
            output_dir = '/home/az/Projects/01_Curie/06.1_pyROMA_Sofia_results/pyroma_debug/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = f'{output_dir}/{gene_set_name}.txt'
            with open(output_file, "w") as f:
                f.write(f"Module: {gene_set_name}\n")
                shape = centered_medians.shape
                f.write(f"ExpMat Head: {centered_medians.get()}\n")
                f.write(f"ExpMat Shape: {shape[0]} x {shape[1]}\n")
                f.write(f"Raw ExpMat shape: {ExpMat.shape[0]} x {ExpMat.shape[1]}\n")
                f.write(f"Raw ExpMat Head: {ExpMat[:5, :5]}\n")
                f.write(f"GeneScore Shape: {len(GeneScore)}\n")
                f.write(f"Gene Score: {GeneScore.get()}\n")
                f.write(f"GeneScore[ToUse] Shape: {cp.sum(ToUse).item()}\n")
                f.write(f"GeneScore[ToUse]: {GeneScore[ToUse].get()}\n")
                if val is not None:
                    f.write(f"Computed val: {val.get()}\n")
                else:
                    f.write("Computed val: N/A\n")
                if val.item() > 0:
                    f.write("Fix PC Sign output: 1")
                    return 1
                else:
                    f.write("Fix PC Sign output: -1")
                    return -1

        # MODE: 'UseExtremeWeights'
        if Mode == 'UseExtremeWeights':
            print_msg("Orienting PC by using the most extreme PC weights and gene expression.")
            if ExpMat is None:
                print_msg("No ExpMat provided. Orientation will be random.")
                return 1
            if Thr is None:
                print_msg("No Thr provided. Using entire set of genes.")
                ToUse = cp.full(SampleScore.shape, True, dtype=bool)
            else:
                cutoff = cp.quantile(cp.abs(SampleScore), 0.15)
                ToUse = cp.abs(SampleScore) >= cutoff
            gene_means = cp.mean(ExpMat, axis=1)
            sum_val = cp.sum(SampleScore[ToUse] * gene_means[ToUse])
            if sum_val.item() > 0:
                return 1
            else:
                return -1

        # If none of the above modes match, default:
        return 1

            
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
        #correct_sign = 1
        pc1 = correct_sign * pc1
        
        projections_1 = X @ pc1 # the scores that each gene have in the sample space
        #print(f"Raw X shape: {X.shape}")
        projections_2 = X @ pc2
        #print('shape of projections should corresponds to n_genes', projections.shape)
        # Compute the median of the projections
        median_exp = np.median(projections_1) 
        # TODO: is median expression is calculated only with the pc1 projections?
        return median_exp, projections_1, projections_2 #TODO: save gene scores after pc sign correction



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
        import cupy as cp
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
            null_median_distribution = cp.asnumpy(null_median_distribution)
            #print('shape of the null distribution: ', null_distribution.shape)
            #print('shape of null med distribution: ', null_median_distribution.shape)

            # L1 statistics
            if self.computation_mode == 'sc':                
                test_l1 = gene_set_result.svd.explained_variance_ratio_[0]
                if isinstance(test_l1, cp.ndarray):
                    test_l1 = cp.asnumpy(test_l1)
            else:
                test_l1 = gene_set_result.svd.explained_variance_ratio_[0]

            #p_value = (np.sum(null_distribution >= test_l1) + 1) / (len(null_distribution) + 1) # original value
            #print("test L1 type:", type(test_l1))
            #print(type(null_distribution))
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
            #print(np.abs(test_median_exp))
            if isinstance(test_median_exp, cp.ndarray):
                test_median_exp = cp.asnumpy(test_median_exp) 
            #test_median_exp, projections_1, projections_2 = self.compute_median_exp(gene_set_result.svd, gene_set_result.X, gene_set_name)
            #print('type null med distrib', type(null_median_distribution))
            #print(null_distribution.shape)
            # print("null med distr", null_median_distribution)
            # print("type test med exp:", type(test_median_exp))
            # print('np.abs', np.abs(null_median_distribution))
            # print('>=', np.abs(null_median_distribution) >= np.abs(test_median_exp))
            # print('np.sum', (np.sum(np.abs(null_median_distribution) >= np.abs(test_median_exp)) ))
            # print('+1', (np.sum(np.abs(null_median_distribution) >= np.abs(test_median_exp)) + 1) )
            # print('shape[0]', ((null_median_distribution.shape[0]) + 1))

            q_value = (np.sum(np.abs(null_median_distribution) >= np.abs(test_median_exp)) + 1) / ((null_median_distribution.shape[0]) + 1)
            #q_value = (np.sum(np.abs(null_median_distribution) >= np.abs(test_median_exp)) + 1) / (len(null_median_distribution) + 1) #from bulk
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

    def randomset_parallel_gpu(self, key, iters=100):
        """
        GPU-parallel computation of null distributions via JAX vectorization,
        
        This function uses self.adata.X (a scipy sparse matrix of shape (n_samples, n_genes)).
        For each iteration:
        1. Generate a set of gene indices (of length nullsize).
        2. Use a JAX pure callback to extract the dense submatrix corresponding to these
            indices from self.adata.X (thus converting only a small subset to dense).
        3. Compute the full SVD on the dense subset.
        4. Compute the transformed data (X_transformed = X_subset @ Vh.T),
            then compute the explained variance ratio as:
                expl_ratio = variance(X_transformed[:, 0]) / sum(variance(X_subset, axis=0))
        5. Compute PC1 projections and its median.
        
        The whole process is vectorized via jax.vmap.
        """
        import jax
        import jax.numpy as jnp
        import jax.scipy.linalg as jsp_linalg
        import numpy as np

        # Use self.adata.X which is sparse.
        n_samples, n_genes = self.adata.X.shape  # shape: (n_samples, n_genes)
        nullsize = self.nullgenesetsize         # number of genes to sample per iteration

        # Step 1: Generate batched random indices (concrete integers).
        def gen_indices(key):
            perm = jax.random.permutation(key, n_genes)
            return perm[:nullsize]
        keys = jax.random.split(key, iters)
        batched_indices = jax.vmap(gen_indices)(keys)  # shape: (iters, nullsize)

        # Step 2: Define a Python function that extracts the dense subset.
        def get_dense_subset(indices):
            # indices: a numpy array of shape (nullsize,)
            # Extract the columns corresponding to these indices from the sparse matrix.
            # self.adata.X has shape (n_samples, n_genes)
            return self.adata.X[:, indices].toarray()  # returns a (n_samples, nullsize) dense array

        expected_shape = (n_samples, nullsize)

        # Step 3: Define the per-iteration function.
        def one_iteration(indices):
            # Use jax.pure_callback to extract the dense submatrix.
            X_subset = jax.pure_callback(
                get_dense_subset,
                jnp.zeros(expected_shape, dtype=jnp.float64),
                indices
            )
            # Now X_subset is a JAX array (dense) with shape (n_samples, nullsize).
            # Compute full SVD on X_subset.
            U, s, Vh = jsp_linalg.svd(X_subset, full_matrices=False)
            # Compute transformed data: X_transformed = X_subset @ Vh.T
            X_transformed = X_subset @ Vh.T  # shape: (n_samples, k); k equals number of components returned
            # Compute explained variance per component (using variance along axis=0)
            exp_var = jnp.var(X_transformed, axis=0)
            # Compute the full variance of X_subset (sum over features)
            full_var = jnp.var(X_subset, axis=0).sum()
            # Compute L1 explained variance ratio (for the first component)
            expl_ratio = exp_var[0] / full_var
            # For PC1 projections, use the first row of Vh.
            pc1 = Vh[0, :]  # shape: (nullsize,)
            projections = X_subset @ pc1  # shape: (n_samples,)
            med_exp = jnp.median(projections)
            return expl_ratio, med_exp, projections

        # Step 4: Vectorize one_iteration over the batched indices.
        expl_ratios, med_exps, projections_all = jax.vmap(one_iteration)(batched_indices)

        # Store the computed null distributions in your object.
        self.nulll1 = expl_ratios
        self.null_median_exp = med_exps
        self.null_projections = projections_all

        return expl_ratios, med_exps, projections_all

    

    
    class SVDResult:
        def __init__(self, U, s, Vt, explained_variance_ratio):
            self.U = U
            self.s = s
            self.Vt = Vt
            self.explained_variance_ratio_ = [explained_variance_ratio]  # Expected as a list
            
            self.components_ = (Vt[0], Vt[1]) if Vt.shape[0] > 1 else (Vt[0], None)

    def compute_rapids(self, 
        selected_gene_sets, 
        loocv_on=True, 
        managed_memory=True,
        pool_allocator=True,
        devices=0,
        null_iters=100):

        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        
        rmm.reinitialize(
        managed_memory=managed_memory,
        pool_allocator=pool_allocator,
        devices=devices, # GPU device IDs to register. By default registers only GPU 0.
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)

        results = {}
        

        
        # Convert to CuPy sparse matrix
        if not cpsp.issparse(self.adata.X):
            X = cpsp.csr_matrix(cp.asarray(self.adata.X.T)) 
        else:
            X = self.adata.X.T
        self.adata.X = X.T 
        self.adata.raw = self.adata.copy()

        # Center the sprase matrix
        row_means = X.mean(axis=1).get()  # Convert to NumPy array (avoids sparse issues)
        row_means = cp.asarray(row_means).reshape(-1, 1)  # Ensure correct shape
        row_means_sparse = cpsp.csr_matrix(row_means)  # Convert back to sparse format

        # **Expand row_means manually by repeating it across columns**
        row_means_expanded = cpsp.csr_matrix(cp.tile(row_means, (1, X.shape[1])))

        # **Row-wise subtraction:**
        X_centered = X - row_means_expanded  # Now the shapes match correctly

        # Compute column means (1, n_features)
        col_means = X_centered.mean(axis=0).get()  # Convert to NumPy
        col_means = cp.asarray(col_means).reshape(1, -1)  # Ensure correct shape
        col_means_sparse = cpsp.csr_matrix(col_means)  # Convert to sparse format

        # **Expand col_means manually by repeating it across rows**
        col_means_expanded = cpsp.csr_matrix(cp.tile(col_means, (X.shape[0], 1)))

        # **Column-wise subtraction:**
        X_centered = X_centered - col_means_expanded 

        self.adata.X = X_centered.T
        del X, #X_centered

        self.indexing(self.adata)
        self.read_gmt_to_dict(self.gmt)

        if selected_gene_sets == 'all':
            selected_gene_sets = self.genesets.keys()
        else:
            selected_gene_sets = list(set(selected_gene_sets) & set(self.genesets.keys()))
        
        # list of pathway names sorted by its geneset size 
        sorted_gene_sets = self.select_and_sort_gene_sets(selected_gene_sets)

        for gene_set_name in sorted_gene_sets:
            # Extract the subset matrix for the gene set (n_samples x n_genes)
            print(f'Processing gene set: {color.BOLD}{color.DARKCYAN}{gene_set_name}{color.END}', end=' | ')
            
            self.subsetting(self.adata, self.genesets[gene_set_name])
            gene_indices = self.subsetlist
            #print(type(gene_indices),gene_indices)
            n_genes = gene_indices.shape[0]
            print('len of subsetlist:', color.BOLD, n_genes, color.END)
            
            #print("subsetlist: ", type(self.subsetlist), self.subsetlist)
            X_subset = self.adata[:, gene_indices].X
            #print("X_subset type", type(X_subset))
            # Compute SVD on the gene set to obtain the test explained variance ratio
            test_expl_ratio, U, s, Vt = self.compute_svd_explained_variance(X_subset)
            # Generate the null distribution of explained variance ratios for random gene sets of the same size
            null_expl_ratios, null_median_exp, null_projections_1 = self.compute_null_distribution(n_genes, null_iters)
            # Compute the empirical p-value (proportion of null ratios that are at least as large as the test ratio)
            #p_value = cp.mean(null_expl_ratios >= test_expl_ratio)

            svd_result = self.SVDResult(U, s, Vt, test_expl_ratio)
 
            #null_median_exp, null_projections_1, null_projections_2 = self.compute_median_exp(
            #    svd_result, X_subset, self.adata.raw[:, gene_indices].X, gene_set_name
            #)

            # TODO: in compute_null_distribution add pc2 projections and all dependencies
            null_projections_2 = None

            gene_set_result = self.GeneSetResult(
                subset = None, 
                subsetlist = gene_indices, 
                outliers = None, 
                nullgenesetsize = n_genes, 
                svd = svd_result,
                X = X_subset.copy(), 
                raw_X_subset = self.adata.raw[:, gene_indices].X, 
                nulll1 = cp.asnumpy(null_expl_ratios),
                null_median_exp = null_median_exp, 
                null_projections = (null_projections_1, null_projections_2)
            )
            gene_set_result.custom_name = f"GeneSetResult {gene_set_name}"
            results[gene_set_name] = gene_set_result

        assessed_results = self.assess_significance(results)
        self.adata.uns['ROMA'] = assessed_results
        #self.adata.uns['ROMA'] = results
        self.adata.uns['ROMA_stats'] = self.p_values_in_frame(assessed_results)
        self.select_active_modules(self.q_L1_threshold, self.q_Med_Exp_threshold)
        #self.unprocessed_genesets = unprocessed_genesets
        self.custom_name = color.BOLD + color.GREEN + 'scROMA' + color.END +': module activities are computed'
        print(color.BOLD, color.PURPLE, 'Finished', color.END, end=': ')
        
        # plotting functions inherit adata from the ROMA class 
        #self.pl.adata = self.adata
        return
    

    def compute_svd_explained_variance(self, X_subset):
        """
        Compute the SVD of a given submatrix and return the explained variance ratio of the first component.

        Parameters:
          X_subset: Cupy array of shape (n_samples, n_genes_subset).

        Returns:
          expl_ratio: Explained variance ratio for the first singular component.
          U, s, Vt: SVD factors (with s containing singular values).
        """
        import cupy as cp
        import cupy.linalg as cpla

        X_subset = X_subset.toarray()
        #print("after .toarray(): ", type(X_subset))

        # Compute SVD using GPU-accelerated routine; full_matrices=False for efficiency
        U, s, Vt = cpla.svd(X_subset, full_matrices=False)
        # Calculate the proportion of variance explained by the first component
        expl_ratio = (s[0] ** 2) / cp.sum(s ** 2)
        return expl_ratio, U, s, Vt

    def compute_null_distribution(self, n_genes, null_iters=100):
        """
        Generate a null distribution by repeatedly sampling random gene subsets (of size n_genes)
        from the full expression matrix and computing the explained variance ratio for each.

        Parameters:
          n_genes: Number of genes in the target gene set.
          null_iters: Number of random iterations to perform.

        Returns:
          Cupy array of explained variance ratios (shape: (null_iters,)).
        """

        import cupy as cp
        import cupy.linalg as cpla
        import cupyx.scipy.sparse as cpsp

        n_samples, total_genes = self.adata.X.shape

        # Build a batch of submatrices corresponding to random gene subsets.
        # For each iteration, randomly sample n_genes indices (without replacement)
        submatrices = []
        for i in range(null_iters):
            idx = cp.random.choice(total_genes, size=n_genes, replace=False)
            # Extract submatrix: shape (n_samples, n_genes)
            X_subset = self.adata.X[:, idx]
            if cpsp.issparse(X_subset):
                X_subset = X_subset.toarray()  # Convert to dense
            #X_subset = cp.asarray(X_subset)
            submatrices.append(X_subset)
        # Stack into a single 3D array: (null_iters, n_samples, n_genes)
        X_batch = cp.stack(submatrices, axis=0)
        # Center each submatrix along the gene (column) axis
        #X_batch_centered = X_batch - cp.mean(X_batch, axis=2, keepdims=True)
        # Perform batched SVD on the 3D array (all iterations in one call)
        # Cupy’s SVD supports batched inputs if the array is 3D.
        U, s, Vt = cpla.svd(X_batch, full_matrices=False)
        # Compute explained variance ratio for the first component for each batch element.
        expl_ratios = (s[:, 0] ** 2) / cp.sum(s ** 2, axis=1)

        # Extract the first principal component for each iteration.
        # Vt has shape (null_iters, n_components, n_genes) where n_components = min(n_samples, n_genes)
        pc1 = Vt[:, 0, :]  # shape: (null_iters, n_genes)

        # Compute the PC1 projections: for each iteration i, multiply X_batch[i] (shape: n_samples x n_genes)
        # by the corresponding pc1[i] (shape: n_genes,) to yield a projection vector (n_samples,).
        # This is equivalent to a batch-wise dot product.
        null_projections_1 = cp.sum(X_batch * pc1[:, cp.newaxis, :], axis=2)  # shape: (null_iters, n_samples)

        # Compute the median of the PC1 projections along the sample axis for each iteration.
        null_medians = cp.median(null_projections_1, axis=1)

        return expl_ratios, null_medians, null_projections_1 

    def select_active_modules(self, q_L1_threshold=0.05, q_Med_Exp_threshold=0.05):
        """
        Selects the active pathways above the threshold
        """

        df = self.adata.uns['ROMA_stats']
        active_modules = df[(df['q L1'] <= q_L1_threshold) | (df['q Med Exp'] <= q_Med_Exp_threshold)]
        self.adata.uns['ROMA_active_modules'] = active_modules

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
        
    
    
 # printing class 
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

   