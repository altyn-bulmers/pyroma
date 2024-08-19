import numpy as np
import time
from scipy import stats
import scanpy as sc
import multiprocessing
from .utils import *

class ROMA:
    def __init__(self):
        self.adata = None
        self.gmt = None
        self.genesets = {}
        self.approx_int = 20
        self.min_n_genes = 10
        self.nullgenesetsize = None
        self.subset = None
        self.subsetlist = None
        self.outliers = []
        self.svd = None
        self.X = None
        self.nulll1 = []
        self.results = {}
        self.null_distributions = {}
        manager = multiprocessing.Manager()
        self.parallel_results = manager.dict()
        self.custom_name = color.BOLD + color.GREEN + 'scROMA' + color.END
        self.p_threshold=0.05 
        self.q_threshold=0.05

    def __repr__(self) -> str:
        return self.custom_name
    
    def __str__(self) -> str:
        return self.custom_name

    import warnings
    warnings.filterwarnings("ignore") #worked to supperss the warning message about copying the dataframe

    def read_gmt_to_dict(self, gmt):
        genesets = {}
        
        file_name = f'genesets/{gmt}.gmt'
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
        
    def subsetting(self, adata, geneset, verbose=0):
        #adata
        #returns subset and subsetlist

        if verbose:
            print(' '.join(x for x in geneset))
        idx = adata.var.index.tolist()
        subsetlist = list(set(idx) & set(geneset))
        subset = adata[:, [x for x in subsetlist]]
        self.subset = subset
        self.subsetlist = subsetlist
        return subset, subsetlist
    
        
    def loocv(self, subset, verbose=0, for_randomset=False):
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
        subset = adata[:, [x for x in subset]]

        # Omitting the centering of the subset to obtain global centering: 
        #X = subset.X - subset.X.mean(axis=0)
        X = np.asarray(subset.X.T) 
        # Compute the SVD of X without the outliers
        svd = TruncatedSVD(n_components=2, algorithm=algorithm, n_oversamples=2) #algorithm='arpack')
        svd.fit(X)
        #svd.explained_variance_ratio_ = (s ** 2) / (X.shape[0] - 1)
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
        X = np.asarray(X) 

        # Initialize IncrementalPCA for 1 component
        svd = IncrementalPCA(n_components=1, batch_size=1000)
        if partial_fit:
            svd.partial_fit(X)
        else:            
            svd.fit(X)
        
        # Store in the object if not for_randomset
        if not for_randomset:
            self.svd = svd
            self.X = X
        return svd, X


    def orient_pc1(self, pc1, data):
        # Orient PC1 to maximize positive correlation with mean expression
        # TODO: if the user knows the orientation -> make it a hyperparameter 
        # (e.g. in the direction of the gene expression of a certain gene)
         
        mean_expr = data.mean(axis=0) #genes are in rows
        if np.corrcoef(pc1, mean_expr)[0, 1] < 0:
            return -pc1
        return pc1

    def compute_median_exp(self, svd_, X):
        """
        Computes the shifted pathway 
        """

        pc1, pc2 = svd_.components_
        # Orient PC1
        pc1 = self.orient_pc1(pc1, X)
        projections_1 = X @ pc1 # the scores that each gene have in the sample space
        projections_2 = X @ pc2
        #print('shape of projections should corresponds to n_genes', projections.shape)
        # Compute the median of the projections
        median_exp = np.median(projections_1) 
        # TODO: is median expression is calculated only with the pc1 projections?
        return median_exp, projections_1, projections_2


    def process_iteration(self, sequence, idx, iteration, incremental, partial_fit, algorithm):
        """
        Iteration step for the randomset calculation
        """

        #np.random.seed(iteration)
        subset = np.random.choice(sequence, self.nullgenesetsize, replace=False)
        gene_subset = np.array([x for i, x in enumerate(idx) if i in subset])
        outliers = self.loocv(self.adata[:,[x for x in gene_subset]], for_randomset=True)
        if incremental:
            svd_, X = self.robustIncrementalPCA(self.adata, gene_subset, outliers, for_randomset=True, partial_fit=partial_fit)
        else:    
            svd_, X = self.robustTruncatedSVD(self.adata, gene_subset, outliers, for_randomset=True, algorithm=algorithm)
            
        l1 = svd_.explained_variance_ratio_
        median_exp, null_projections_1, null_projections_2 = self.compute_median_exp(svd_, X)

        return l1, median_exp, null_projections_1, null_projections_2
        
    def randomset_parallel(self, subsetlist, outliers, verbose=1, prefer_type='processes', incremental=False, iters=100, partial_fit=False, algorithm='randomized'):
        """
        Calculates scores for the random gene set of the same size and returns null distributions of scores.
        """
        from joblib import Parallel, delayed
        import time

        # Start timer
        start = time.time()

        # Calculate null gene set size by finding the closest size 
        # from filtered geneset sizes by approx_sample in the log scale 
        
        candidate_nullgeneset_size = self.nullgenesetsize
        #len(self.subsetlist)

        # If the null distribution with this null geneset size was caclulated, pass to the next pathway
        if candidate_nullgeneset_size in self.null_distributions:
            self.nulll1, self.null_median_exp = self.null_distributions[candidate_nullgeneset_size]
            print('Took null distribution from previous calculation')
        else: 
            # Define the number of iterationsself.null_geneset_sizes
            num_iterations = iters
            sequence = np.arange(self.adata.shape[1])
            idx = self.adata.var.index.to_numpy()

            # Use parallel processing to process iterations
            results = Parallel(n_jobs=-1, prefer=prefer_type)(
                delayed(self.process_iteration)(sequence, idx, iteration, incremental, partial_fit, algorithm) for iteration in range(num_iterations)
            )

            # Unzip the results
            nulll1, null_median_exp, null_projections_1, null_projections_2 = list(zip(*results))
            nulll1_array = np.array(nulll1)
            null_median_exp = np.array(null_median_exp)
            null_projections_1 = np.array(null_projections_1)
            null_projections_2 = np.array(null_projections_2)
            null_projections = np.stack((null_projections_1, null_projections_2), axis=1)
            # update the dictiorary with null distributions 
            self.null_distributions[candidate_nullgeneset_size] = [np.copy(nulll1_array), np.copy(null_median_exp)]
            # Store results in the object
            self.nulll1 = np.copy(nulll1_array)
            self.null_median_exp = np.copy(null_median_exp)
            self.null_projections = np.copy(null_projections)

            # Calculate elapsed time
            end = time.time()
            elapsed_time = end - start
            minutes, seconds = divmod(elapsed_time, 60)

            # Verbose output
            if verbose:
                print(f"Running time (min): {int(minutes):02}:{seconds:05.2f}")

        return

    def assess_significance(self, results):
        """
        Computes the empirical p-value based on the null distribution of L1 scores and median expression.
        Adjust p-values and q-values using the Benjamini-Hochberg procedure.
        """
        from scipy.stats import false_discovery_control as benj_hoch
        
        #valid_results = {name: result for name, result in results if result is not None}
        ps = np.zeros(shape=len(results))
        qs = np.zeros(shape=len(results))
        for i, (_, gene_set_result) in enumerate(results.items()):
            #print('PRINTING to fix the ERROR', gene_set_result.nulll1)
            #print('NULL MEDIAN EXP', gene_set_result.null_median_exp)
            null_distribution = gene_set_result.nulll1[0]
            null_median_distribution = gene_set_result.null_median_exp

            # L1 statistics
            test_l1 = gene_set_result.svd.explained_variance_ratio_[0]
            p_value = (np.sum(null_distribution >= test_l1) + 1) / (len(null_distribution) + 1)
            # otherwise p_value could be calculated with (np.sum(null_distribution >= test_l1)) / (len(null_distribution))
            ps[i] =  p_value #if p_value <= 1.0 else 1.0
            gene_set_result.test_l1 = test_l1

            # Median Exp statistic
            test_median_exp, projections_1, projections_2 = self.compute_median_exp(gene_set_result.svd, gene_set_result.X)
            q_value = (np.sum(null_median_distribution >= test_median_exp) + 1) / (len(null_median_distribution) + 1)
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
        
        for i, (_, gene_set_result) in enumerate(results.items()):
            gene_set_result.p_value = adjusted_ps[i]
            gene_set_result.q_value = adjusted_qs[i]
        return results

    def approx_size(self, flag):
        """
        # Approximate size
        # For current subset and gene set -> we compute the null gene set size
        # add it to the dictionary of null gene set sizes
        # for the next one, we calculate if the closest size in dictionary is smaller by k(approx_int) to ours
        # if smaller -> we just use the same distribution from the dictionary (as it is computed)
        # is larger -> we create a new 
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
        def __init__(self, subset, subsetlist, outliers, nullgenesetsize, svd, X, nulll1, null_median_exp, null_projections):
            self.subset = subset
            self.subsetlist = subsetlist
            self.outliers = outliers
            self.nullgenesetsize = nullgenesetsize
            self.svd = svd
            self.X = X
            self.projections_1 = None
            self.projections_2 = None
            self.nulll1 = nulll1
            self.null_median_exp = null_median_exp
            self.null_projections = null_projections
            self.p_value = None
            self.q_value = None
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
        for k, v in assessed_results.items():
            l1_dict[k] = v.test_l1
            p_dict[k] = v.p_value
            median_exp_dict[k] = v.test_median_exp
            q_dict[k] = v.q_value

        df = pd.DataFrame() 
        df['L1'] = pd.Series(l1_dict) 
        df['p_value'] = pd.Series(p_dict)
        df['Median_Exp'] = pd.Series(median_exp_dict)
        df['q_value'] = pd.Series(q_dict)

        return df
    
    def compute(self, selected_gene_sets, parallel=False, incremental=False, iters=100, partial_fit=False, algorithm='randomized'):        
        
        """
        Computes ROMA
        min_n_genes = 10 (default) minimum geneset size of genes present in the provided dataset.
        approx_int = 20 (default) granularity of the null geneset size, 
                    from 0 to 100, what is the minimum distance in the n of genes between sizes of the genesets.  
        
        """

        results = {}
        
        # Centering expression of each gene in the global matrix, copying the original in adata.raw
        # Centering over genes 
        self.adata.raw = self.adata.copy()
        self.adata.X -= self.adata.X.mean(axis=0)

        self.read_gmt_to_dict(self.gmt)

        # to mark the first one
        flag = True
        
        if selected_gene_sets == 'all':
            selected_gene_sets = self.genesets.keys()

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
            self.loocv(self.subset)
            
            self.approx_size(flag)
            flag = False

            if incremental:
                self.robustIncrementalPCA(self.adata, self.subsetlist, self.outliers)
                #self.robustKernelPCA(self.adata, self.subsetlist, self.outliers)
            else:
                self.robustTruncatedSVD(self.adata, self.subsetlist, self.outliers, algorithm=algorithm)
            # parallelization
            if parallel:
                self.randomset_parallel(self.adata, self.subsetlist, 
                                        self.outliers, prefer_type='processes', incremental=incremental, iters=iters, partial_fit=partial_fit, 
                                        algorithm=algorithm)

            #print('self.nullgenesetsize', self.nullgenesetsize)
            #print('self.nulll1 :', self.nulll1)
            # Store the results for this gene set in a new instance of GeneSetResult
            
            gene_set_result = self.GeneSetResult(self.subset, self.subsetlist, self.outliers, self.nullgenesetsize, 
                                                 self.svd, self.X, 
                                                 self.nulll1, self.null_median_exp, self.null_projections)

            gene_set_result.custom_name = f"GeneSetResult {gene_set_name}"
            # Store the instance of GeneSetResult in the dictionary using gene set name as the key
            results[gene_set_name] = gene_set_result
            #print('null geneset size:', self.nullgenesetsize)

        #print(' RESULTS:', results)
        # calculate p_value adjusted for multiple-hypotheses testing
        assessed_results = self.assess_significance(results)
        #self.results = assessed_results
        self.adata.uns['ROMA'] = assessed_results
        self.adata.uns['ROMA_stats'] = self.p_values_in_frame(assessed_results)
        self.select_active_modules(self.p_threshold, self.q_threshold)
        self.unprocessed_genesets = unprocessed_genesets
        self.custom_name = color.BOLD + color.GREEN + 'scROMA' + color.END +': module activities are computed'
        print(color.BOLD, color.PURPLE, 'Finished', color.END, end=': ')

        return 
    
    def select_active_modules(self, p_threshold=0.05, q_threshold=0.05):
        """
        Selects the active pathways above the threshold
        """

        df = self.adata.uns['ROMA_stats']
        active_modules = df[(df['p_value'] <= p_threshold) | (df['q_value'] <= q_threshold)]
        self.adata.uns['ROMA_active_modules'] = active_modules

        return
    
    
    # TODO: implement the plotting functions, similar to rROMA



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

    # TODO : plotting in progress
    class pl:
        def __init__(self):
            self.geneset_name = None
        
        def gene_weights(self, geneset_name):
            """
            Plotting the gene weights.
            """
            import matplotlib.pyplot as plt
            import pandas as pd 

            fig, ax1 = plt.subplots(1, 1, figsize=(7.5,16))
            fig.tight_layout()
            #sns.set(style="darkgrid")
            #sns.set_palette("Pastel1")
            plt.grid(color='white', lw = 0.5, axis='x')

            roma_result = roma.adata.uns['ROMA'][geneset_name]
            df = pd.DataFrame(roma_result.projections_1, index=roma_result.subsetlist, columns=['gene weights'])
            df = df.sort_values(by='gene weights', ascending=True).reset_index()

            sns.scatterplot(df, y='index', x='gene weights', color='k', label='gene weights', ax=ax1)
            ax1.set_title(f'{geneset_name} Gene Weights', loc = 'center', fontsize = 18)
            plt.setp(ax1, xlabel='PC1 scores')
            plt.setp(ax1, ylabel='Gene')
            plt.yticks(fontsize=8, linespacing=0.9)
            plt.grid(color='white', lw = 1, axis='both')

            #plt.title(f'Gene Weights', loc = 'right', fontsize = 18)
            plt.legend()
            plt.show()
            
            return
        
        def plot_gene_projections(self, geneset_name):
            """
            Represent the pathway genes in the pca space.
            Against null distribution , i.e. genes in the pca space of all the random genesets.
            """
            
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set(style="darkgrid")
            sns.set_palette("Pastel1")

            # Assuming roma_result is already loaded and contains the required matrices
            roma_result = roma.adata.uns['ROMA'][geneset_name]
            projections_1 = roma_result.projections_1
            projections_2 = roma_result.projections_2
            null_projections = roma_result.null_projections
            null_projections_flat = null_projections.reshape(-1, 2)

            # Setting up the plot
            plt.figure(figsize=(10, 8))

            # Setting the axis labels
            plt.axhline(0,color='k') # x = 0
            plt.axvline(0,color='k') # y = 0
            plt.xlabel('PC1')
            plt.ylabel('PC2')

            # Plotting the points from all matrices in null_projections in hollowish Tr. Blue color
            sns.scatterplot(x=null_projections_flat[:, 0], y=null_projections_flat[:, 1], color='dodgerblue', label='Null Projections', edgecolor='black', marker='o', alpha=0.2)

            # Plotting the points from projections_2 in Red color (Tr. Red)
            sns.scatterplot(x=projections_1, y=projections_2, color='red', label=f'{geneset_name}', edgecolor='black')

            # Adding grid
            plt.grid(True)

            # Adding title
            plt.title(f'{geneset_name} and Null distribution in PCA space ')

            # Removing the legend duplicates
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            # Showing the plot
            plt.show()
            return
