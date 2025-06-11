import numpy as np
import time
from scipy import stats
import scanpy as sc
import multiprocessing
from .utils import *

class ROMA:
    
    # TODO in plotting : handle many genesets, heatmap (?) 
    from .plotting import plotting as pl 
    #TODO: initialize pl.adata with roma.adata
    pl = pl()
    
    def __init__(self):
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
        self.loocv_scores = {}
        self.global_gene_counts = {} # fisher
        self.global_outlier_counts = {} # fisher
        self.svd = None
        self.X = None
        self.raw_X_subset = None
        self.nulll1 = []
        self.test_l1 = None
        self.p_value = None
        self.test_median_exp = None
        self.med_exp_p_value = None
        self.projections_1 = None
        self.projections_2 = None
        self.results = {}
        self.null_distributions = {}
        manager = multiprocessing.Manager()
        self.parallel_results = manager.dict()
        self.custom_name = color.BOLD + color.GREEN + 'scROMA' + color.END
        self.q_L1_threshold=0.05 
        self.q_Med_Exp_threshold=0.05
        # params for fix_pc_sign
        self.gene_weights = {}
        self.pc_sign_mode = 'PreferActivation'  # Mode for PC1 sign correction: 'UseAllWeights', 'UseMeanExpressionAllWeights'
        self.pc_sign_thr = 0.90  # Threshold for extreme weights
        self.def_wei = 1  # Default weight for missing weights
        self.cor_method = 'pearson'  # Correlation method
        self.correct_pc_sign = 1    # Store after orient_pc to keep track of orientation
        # New attributes for gene signs and extreme percentage
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
            
        loocv_scores = {}
        for i, g in enumerate(subset.var_names):
            loocv_scores[g] = l1scores[i]  
        self.loocv_scores = loocv_scores

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

    

    def fisher_outlier_filter(self, gene_outlier_counts, gene_pathway_counts,
                            outlier_fisher_thr=0.05, min_gene_sets=3):
        """
        Determines whether each gene should be considered an outlier based on a Fisher exact test.
        
        For each gene, the function compares the number of times it was flagged as an outlier
        (gene_outlier_counts) to the total number of gene sets in which it appears (gene_pathway_counts)
        against the corresponding overall totals from all genes. The 2x2 contingency table is constructed as:
        
            [[ total_outliers - gene_outlier_count, gene_outlier_count ],
            [ total_pathways - gene_pathway_count, gene_pathway_count ]]
        
        Then a one-sided Fisher test is performed (using the 'greater' alternative) to test
        whether the gene’s proportion of outlier calls is significantly higher than the global average.
        
        Additionally, if a gene appears in fewer than `min_gene_sets` collections,
        it is always kept (i.e. not marked as an outlier) regardless of the Fisher test.
        
        Parameters
        ----------
        gene_outlier_counts : dict
            Dictionary mapping gene name to the number of gene sets in which it was flagged as an outlier.
        gene_pathway_counts : dict
            Dictionary mapping gene name to the total number of gene sets in which the gene appears.
        outlier_fisher_thr : float, optional
            The significance threshold for the Fisher test p-value. A gene is considered
            significantly aberrant (and hence an outlier) if p < outlier_fisher_thr.
            (Default is 0.05.)
        min_gene_sets : int, optional
            Minimum number of gene sets a gene must appear in to be eligible for being
            flagged as an outlier. Genes with fewer appearances are always kept.
            (Default is 3.)
        
        Returns
        -------
        gene_outlier_flags : dict
            Dictionary mapping each gene to a Boolean value: True if the gene is considered
            an outlier (i.e. its aberrant behavior is statistically significant), False otherwise.
        """

        from scipy.stats import fisher_exact
        # Compute overall totals for the entire collection
        total_outliers = sum(gene_outlier_counts.values())
        total_pathways = sum(gene_pathway_counts.values())
        
        gene_outlier_flags = {}
        for gene, pathways_count in gene_pathway_counts.items():
            # If gene appears in too few gene sets, do not mark it as an outlier
            if pathways_count < min_gene_sets:
                gene_outlier_flags[gene] = False
                continue
            
            # Get the number of outlier calls for this gene (defaulting to 0 if not present)
            gene_outlier = gene_outlier_counts.get(gene, 0)
            
            # Build the contingency table:
            # Row 1: counts for "other genes"
            # Row 2: counts for the gene of interest
            table = [
                [total_outliers - gene_outlier, gene_outlier],
                [total_pathways - pathways_count, pathways_count]
            ]
            
            # Perform a one-sided Fisher exact test (alternative='greater' checks if the gene's
            # proportion of outliers is higher than that of the remainder)
            _, p_value = fisher_exact(table, alternative='greater')
            
            # According to the R implementation, if the p-value is high (>= threshold)
            # then the gene is NOT significantly aberrant (and hence kept). Here, we flag a gene
            # as an outlier if p < threshold.
            gene_outlier_flags[gene] = (p_value < outlier_fisher_thr)
            
        return gene_outlier_flags

    # Optional: a helper function to limit the fraction of genes in a given gene set that
    # are removed as outliers. For example, if too many genes are flagged by LOOCV + Fisher test,
    # you may wish to only remove the ones with the most extreme behavior.
    def limit_outliers_per_geneset(self, gene_set, gene_flags, loocv_scores, max_outlier_prop=0.5):
        """
        Given a list of genes in a gene set and a dictionary of flags indicating whether each gene
        is considered an outlier (from the Fisher test), limit the fraction of genes flagged as outliers
        to at most max_outlier_prop (e.g. 50%).
        
        Parameters
        ----------
        gene_set : list of str
            List of gene names in the gene set.
        gene_flags : dict
            Dictionary mapping gene name to Boolean flag (True if flagged as outlier).
        loocv_scores : dict
            Dictionary mapping gene name to a LOOCV statistic (or other measure of extremity).
            Genes with more extreme LOOCV scores are given priority to be marked as outliers.
        max_outlier_prop : float, optional
            Maximum proportion of genes in the gene set that can be removed as outliers.
            (Default is 0.5.)
        
        Returns
        -------
        filtered_flags : dict
            Updated gene_flags dictionary in which at most max_outlier_prop fraction of genes
            in gene_set are flagged as outliers.
        """
        # Identify genes flagged as outliers in the gene set
        flagged_genes = [g for g in gene_set if gene_flags.get(g, False)]
        allowed_num = int(max_outlier_prop * len(gene_set))
        if len(flagged_genes) > allowed_num:
            # Sort the flagged genes by the absolute value of their LOOCV score (or other score)
            # in descending order so that the most extreme ones remain flagged.
            flagged_sorted = sorted(flagged_genes, key=lambda g: abs(loocv_scores.get(g, 0)), reverse=True)
            # Only the top 'allowed_num' genes remain flagged as outliers.
            keep_flagged = set(flagged_sorted[:allowed_num])
            # Update flags: unflag genes not in the top allowed group
            for g in gene_set:
                if g in flagged_genes:
                    gene_flags[g] = (g in keep_flagged)
        return gene_flags

    # ===== Example usage =====
    # Suppose that during your analysis you have tallied for each gene:
    #   - gene_outlier_counts: the number of gene sets in which the gene was flagged by LOOCV.
    #   - gene_pathway_counts: the total number of gene sets in which the gene appears.
    #
    # For example:
    #
    # gene_outlier_counts = {'GeneA': 5, 'GeneB': 1, 'GeneC': 0, 'GeneD': 4}
    # gene_pathway_counts = {'GeneA': 10, 'GeneB': 3, 'GeneC': 8, 'GeneD': 10}
    #
    # You can then decide which genes are significantly aberrant:
    #
    # flags = fisher_outlier_filter(gene_outlier_counts, gene_pathway_counts,
    #                               outlier_fisher_thr=0.05, min_gene_sets=3)
    #
    # For a specific gene set (list of genes) and if you have LOOCV scores for each gene,
    # you can further limit the fraction of outliers:
    #
    # gene_set = ['GeneA', 'GeneB', 'GeneC', 'GeneD']
    # # Suppose loocv_scores contains a measure of extremity for each gene:
    # loocv_scores = {'GeneA': 2.3, 'GeneB': 1.5, 'GeneC': 0.5, 'GeneD': 3.1}
    # filtered_flags = limit_outliers_per_geneset(gene_set, flags, loocv_scores, max_outlier_prop=0.5)
    #
    # In this example, even if the Fisher test would flag more than 50% of genes as outliers,
    # the final decision is limited to the most extreme ones.


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
        import os
        
        #output_dir = '/home/az/Projects/01_Curie/06.1_pyROMA_Sofia_results/pyroma_debug/'
        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)
        
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
            #print_msg(f"Missing gene weights will be replaced by {DefWei}")
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
                    #median_per_gene = ExpMat.median(axis=1).values #before
                    median_per_gene = np.median(ExpMat, axis=1)
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
            
            subset_mat = ExpMat[ToUse, :]

            row_medians = np.median(subset_mat, axis=1)
            centered_medians = np.array(row_medians - np.mean(row_medians))
            centered_medians = centered_medians.reshape(1, -1)
            val = np.sum(GeneScore[ToUse]*Wei[ToUse]*centered_medians)
            #centered_medians = centered_medians.reshape(-1, 1)
            
            #output_file = f'{output_dir}/{gene_set_name}.txt' 
#
            #with open(output_file, "w") as f:
            #    f.write(f"Module: {gene_set_name}\n")
            #    if centered_medians is not None:
            #        shape = centered_medians.shape
            #        #print(type(shape), type(centered_medians))
            #        #print('shape', shape)
            #        f.write(f"ExpMat Head: {centered_medians}\n")
            #        #f.write(f"Subset Mat: {subset_mat}\n")
            #        f.write(f"ExpMat Shape: {shape[0]} x {shape[1]}\n")
            #        
            #    else:
            #        f.write("ExpMat Shape: N/A\n")
            #    f.write(f"Raw ExpMat shape: {ExpMat.shape[0]} x {ExpMat.shape[1]}\n")
            #    f.write(f"Raw ExpMat Head: {ExpMat[:5, :5]}\n")
            #    f.write(f"GeneScore Shape: {len(GeneScore)}\n")
            #    f.write(f"Gene Score: {GeneScore}\n")
            #    f.write(f"GeneScore[ToUse] Shape: {len(GeneScore[ToUse])}\n")
            #    f.write(f"GeneScore[ToUse]: {GeneScore[ToUse]}\n")
            #    if val is not None:
            #        f.write(f"Computed val: {val}\n")
            #    else:
            #        f.write("Computed val: N/A\n")
#
            if val > 0:
            #        f.write(f"Fix PC Sign output: 1" )
                return 1
            else:
            #        f.write(f"Fix PC Sign output: -1" )
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
        import scipy.sparse as sp
        # Get gene scores (loadings) and sample scores (projections)
        sample_score = pc1
        gene_score = X @ pc1
        #gene_score = raw_X_subset @ pc1
        # Get gene weights if available
        wei = self.gene_weights.get(gene_set_name, None)
        #print("wei: ", wei)
        # exp_mat is data (genes x samples)
        # TODO: actually test if raw_X_subset is numpy.ndarray and not sparse matrix e.g.
        #print(f"GeneScore shape: {gene_score.shape}")
        #print(f"ExpMat shape: {raw_X_subset.shape}")
        if isinstance(raw_X_subset, sp.spmatrix):
            raw_X_subset = raw_X_subset.toarray() #np.float64
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
        Compute the shifted pathway 
        """

        #if X is None or X.shape[0] == 0:
        #    print(f"Warning: X is empty for gene set {gene_set_name}, returning default values.")
        #    return np.nan, np.array([]), np.array([])

        pc1, pc2 = svd_.components_
        #raw_median_exp = np.median(X @ pc1)
        # Orient PC1
        correct_sign = self.orient_pc1(pc1, X, raw_X_subset, gene_set_name=gene_set_name)
        self.correct_pc_sign = correct_sign
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
        import numpy as np
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
        
    ### C from rROMA
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

    def get_raw_p_values(self, gene_set_name=None):
        
        """
        Extract raw p values from null distribution
        """

        null_distribution = self.nulll1[:,0]
        null_median_distribution = self.null_median_exp

        # L1 statistics
        test_l1 = self.svd.explained_variance_ratio_[0]
        p_value = np.mean(np.array(null_distribution) >= test_l1)

        self.test_1 = test_l1
        self.p_value = p_value

        # Median Exp statistic
        test_median_exp, projections_1, projections_2 = self.compute_median_exp(self.svd, self.X, self.raw_X_subset, gene_set_name)
        med_exp_p_value = (np.sum(np.abs(null_median_distribution) >= np.abs(test_median_exp)) + 1) / (len(null_median_distribution) + 1)

        self.test_median_exp = test_median_exp
        self.med_exp_p_value = med_exp_p_value
        self.projections_1 = projections_1 # are already w corrected sign
        self.projections_2 = projections_2

        # instead of del 
        self.nulll1 = None; self.null_median_exp = None; self.raw_X_subset = None

        return

    
    def assess_significance(self, results):
        
        # TODO: incorporate an option to compute p-values via wilcoxon 
        """
        Computes the empirical p-value based on the null distribution of L1 scores and median expression.
        Adjust p-values and q-values using the Benjamini-Hochberg procedure.
        """
        from scipy.stats import false_discovery_control as benj_hoch
        from statsmodels.stats.multitest import multipletests
        import numpy as np
        from scipy.stats import wilcoxon

        ps = np.zeros(shape=len(results))
        med_ps = np.zeros(shape=len(results))
        # loop to collect p-values and med_p-values
        for i, (_, gene_set_result) in enumerate(results.items()):
            ps[i] =  gene_set_result.non_adj_p
            med_ps[i] = gene_set_result.non_adj_med_exp_p

        # Adjusted p-values
        qs = benj_hoch(ps)
        med_exp_qs = benj_hoch(med_ps)
        
        for i, (_, gene_set_result) in enumerate(results.items()):
            gene_set_result.q_value = qs[i]
            gene_set_result.med_exp_q_value = med_exp_qs[i]

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
        def __init__(self, subset, subsetlist, outliers, nullgenesetsize, svd ):
            self.subset = subset
            self.subsetlist = subsetlist
            self.outliers = outliers
            self.nullgenesetsize = nullgenesetsize
            self.svd = svd
            self.X = None
            self.raw_X_subset = None
            self.projections_1 = None
            self.projections_2 = None
            self.nulll1 = None
            self.null_median_exp = None
            self.null_projections = None
            self.q_value = None
            self.med_exp_q_value = None
            self.non_adj_p = None
            self.non_adj_med_exp_p = None
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

        q_dict = {} # adj_l1_p_values
        l1_dict = {}
        q_med_exp_dict = {}
        median_exp_dict = {}
        non_adj_L1_p_values = {}
        non_adj_Med_Exp_p_values = {}
        for k, v in assessed_results.items():
            l1_dict[k] = v.test_l1
            q_dict[k] = v.q_value
            median_exp_dict[k] = v.test_median_exp
            q_med_exp_dict[k] = v.med_exp_q_value
            non_adj_L1_p_values[k] = v.non_adj_p
            non_adj_Med_Exp_p_values[k] = v.non_adj_med_exp_p

        df = pd.DataFrame() 
        df['L1'] = pd.Series(l1_dict) 
        df['ppv L1'] = pd.Series(non_adj_L1_p_values)
        df['Median Exp'] = pd.Series(median_exp_dict)
        df['ppv Med Exp'] = pd.Series(non_adj_Med_Exp_p_values)
        df['q L1'] = pd.Series(q_dict)
        df['q Med Exp'] = pd.Series(q_med_exp_dict)
        return df
    
    def compute(self, selected_gene_sets, parallel=True, incremental=False, iters=100, partial_fit=False, algorithm='randomized', loocv_on=True, double_mean_centering=False, outlier_fisher_thr=0.05, min_gene_sets=3, max_outlier_prop=0.5):        
        
        #pl.adata = self.adata
        """
        Computes ROMA module activity scores
        
        Parameters
        ----------
        min_n_genes = 10 (default) minimum geneset size of genes present in the provided dataset.
        approx_int = 20 (default) granularity of the null geneset size, 
                    from 0 to 100, what is the minimum distance in the n of genes between sizes of the genesets.  
        
        Returns
        ----------
        None
        """

        results = {}
        
        # Centering expression of each gene in the global matrix, copying the original in adata.raw
        # Centering over samples (genes will have 0 mean)
        # in rROMA columns are samples, 
        # and the "scale" function centering is done by subtracting the column means of x from their corresponding columns
        self.adata.raw = self.adata.copy()
        X = self.adata.X.T 
        #X_raw = X.copy()
        
        # TODO: test for various types.
        if not isinstance(X, np.ndarray):
            X = X.toarray() 

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

        unprocessed_genesets = []

        # TODO: here we then need to sort the gene sets by their size first
        # Sort the selected genesets by by their size 
        sorted_gene_sets = self.select_and_sort_gene_sets(selected_gene_sets)

        for gene_set_name in sorted_gene_sets:
            print(f'Processing gene set: {color.BOLD}{color.DARKCYAN}{gene_set_name}{color.END}', end=' | ')
            self.subsetting(self.adata, self.genesets[gene_set_name])
            print('len of subsetlist:', color.BOLD, len(self.subsetlist), color.END, end = ' ')
            if len(self.subsetlist) < self.min_n_genes:
                unprocessed_genesets.append(gene_set_name)
                print("| smaller than min n genes for geneset |")
                continue
            else:
                print()
            if loocv_on:
                self.loocv(self.subset)
            
            if len(self.outliers) > 0:
                print(self.outliers, self.subsetlist[self.outliers[0]])
            # ===== Fisher test =====
            # Update global counts (or local counts per gene set) for each gene:
            for gene in self.subsetlist:
                self.global_gene_counts[gene] = self.global_gene_counts.get(gene, 0) + 1
            for outlier_idx in self.outliers:
                gene_name = self.subsetlist[outlier_idx]
                self.global_outlier_counts[gene_name] = self.global_outlier_counts.get(gene_name, 0) + 1

            # Apply Fisher test to decide if a gene’s outlier proportion is significantly higher
            gene_flags = self.fisher_outlier_filter(self.global_outlier_counts, self.global_gene_counts,
                                            outlier_fisher_thr=outlier_fisher_thr, min_gene_sets=min_gene_sets)

            # Optionally, limit the proportion of outliers in the current gene set
            # LOOCV score per gene in a dictionary self.loocv_scores
            gene_flags = self.limit_outliers_per_geneset(self.subsetlist, gene_flags, self.loocv_scores, max_outlier_prop=max_outlier_prop)

            # Update the outliers list for the current gene set based on the refined gene_flags
            self.outliers = [i for i, gene in enumerate(self.subsetlist) if gene_flags.get(gene, False)]

            self.approx_size(flag)
            flag = False

            if incremental:
                #self.robustPCA(self.adata, self.subsetlist, self.outliers)
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
                self.randomset_parallel(self.adata, self.subsetlist, 
                                        self.outliers, prefer_type='processes', incremental=incremental, iters=iters, partial_fit=partial_fit, 
                                        algorithm=algorithm)

            # Store the results for this gene set in a new instance of GeneSetResult
            
            ### here we can calcualte the raw p_values of L1s and Med_Exps
            self.get_raw_p_values(gene_set_name)
            

            gene_set_result = self.GeneSetResult(self.subset, self.subsetlist, self.outliers, self.nullgenesetsize, 
                                                 self.svd)

            gene_set_result.custom_name = f"GeneSetResult {gene_set_name}"
            gene_set_result.test_l1 = self.test_1
            gene_set_result.non_adj_p = self.p_value
            gene_set_result.non_adj_med_exp_p = self.med_exp_p_value 
            gene_set_result.test_median_exp = self.test_median_exp 
            gene_set_result.projections_1 = self.projections_1
            gene_set_result.projections_2 = self.projections_2

            # Store the instance of GeneSetResult in the dictionary using gene set name as the key
            results[gene_set_name] = gene_set_result
            #print('null geneset size:', self.nullgenesetsize)

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

    
    def save_active_modules_results(self, path, only_active=True, save_adata=True):
        """ 
             choose to save only active or all pathways 
        """
        import pickle
        import os


        output_dir = path
        adata = self.adata 

        active_modules = adata.uns['ROMA_active_modules'].index
        if only_active:
            selected_dict = {k: v for k, v in adata.uns['ROMA'].items() if k in active_modules}
        else:
            selected_dict = adata.uns['ROMA']

        attributes = {
           "subsetlist": "numpy.ndarray",
            "outliers": "list",
            "projections_1": "numpy.ndarray",
            "projections_2": "numpy.ndarray",
            "svd.components_": "numpy.ndarray"
        }

        # Loop over each key in the filtered dictionary
        for key, gene_set_result in selected_dict.items():
            # Create a subfolder for each key
            key_dir = os.path.join(output_dir, key)
            os.makedirs(key_dir, exist_ok=True)
            
            # Loop over each attribute defined in the mapping
            for attr, attr_type in attributes.items():
                # Retrieve attribute value, handling the dotted attribute for "svd.components_"
                if "." in attr:
                    parts = attr.split(".")
                    attr_value = getattr(gene_set_result, parts[0], None)
                    if attr_value is not None:
                        attr_value = getattr(attr_value, parts[1], None)
                else:
                    attr_value = getattr(gene_set_result, attr, None)
                
                # If the attribute exists, save it to a file
                if attr_value is not None:
                    file_name = f"{attr}"
                    file_path = os.path.join(key_dir, file_name)
                    
                    if attr_type == "numpy.ndarray":
                        # Save numpy array (np.save writes binary files; extension can be arbitrary)
                        np.save(file_path, attr_value)
                    elif attr_type == "list":
                        # Save list using pickle
                        with open(file_path, "wb") as f:
                            pickle.dump(attr_value, f)

        
        adata.uns['ROMA_stats'].to_csv(f"{path}/ROMA_stats.csv") 
        adata.uns['ROMA_active_modules'].to_csv(f"{path}/ROMA_active_modules.csv")
        
        # is this part actually necessary now ?     
        del adata.uns['ROMA'] 
        if save_adata:
            adata.write(f"{path}/roma_adata.h5ad")

        return
    

    def load_active_modules_results(self, path):

        # load adata with ROMA results from the path
        # if roma.adata is empty, loads it to roma.adata, considering adata_roma.h5ad was saved before

        import os
        import numpy as np
        import pandas as pd
        import pickle
        from types import SimpleNamespace

        output_dir = path

        loaded_dict = {}

        # Loop through each folder (each folder name is expected to be a key)
        for key in os.listdir(output_dir):
            key_dir = os.path.join(output_dir, key)
            if os.path.isdir(key_dir):
                # Create a new GeneSetResult object
                gene_set = self.GeneSetResult(None, None, None, None, None)
                gene_set.custom_name = key

                # Iterate over files within the folder
                for file in os.listdir(key_dir):
                    file_path = os.path.join(key_dir, file)
                    # Attribute name is the filename without the ".file" extension
                    attr_name = file.replace(".npy", "")
                    #print(attr_name)
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
                        # Load the numpy array for svd.components_
                        components = np.load(file_path, allow_pickle=True)
                        # Create a dummy object (using SimpleNamespace) to hold the attribute
                        gene_set.svd = SimpleNamespace(components_=components)
                        
                loaded_dict[key] = gene_set
        if not self.adata:
            self.adata = sc.read_h5ad(f"{path}/roma_adata.h5ad")
        self.adata.uns['ROMA'] = loaded_dict
        self.adata.uns['ROMA_stats'] = pd.read_csv(f"{path}/ROMA_stats.csv") 
        self.adata.uns['ROMA_active_modules'] = pd.read_csv(f"{path}/ROMA_active_modules.csv")

        return

    
