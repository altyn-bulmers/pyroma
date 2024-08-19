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

# plotting
#class plotting:
#    def gene_weights(self, geneset_name):
#        """
#        Plotting the gene weights.
#        """
#        import matplotlib.pyplot as plt
#        import pandas as pd 
#
#        fig, ax1 = plt.subplots(1, 1, figsize=(7.5,16))
#        fig.tight_layout()
#        #sns.set(style="darkgrid")
#        #sns.set_palette("Pastel1")
#        plt.grid(color='white', lw = 0.5, axis='x')
#
#        roma_result = roma.adata.uns['ROMA'][geneset_name]
#        df = pd.DataFrame(roma_result.projections_1, index=roma_result.subsetlist, columns=['gene weights'])
#        df = df.sort_values(by='gene weights', ascending=True).reset_index()
#
#        sns.scatterplot(df, y='index', x='gene weights', color='k', label='gene weights', ax=ax1)
#        ax1.set_title(f'{geneset_name} Gene Weights', loc = 'center', fontsize = 18)
#        plt.setp(ax1, xlabel='PC1 scores')
#        plt.setp(ax1, ylabel='Gene')
#        plt.yticks(fontsize=8, linespacing=0.9)
#        plt.grid(color='white', lw = 1, axis='both')
#
#        #plt.title(f'Gene Weights', loc = 'right', fontsize = 18)
#        plt.legend()
#        plt.show()
#        
#        return
#    
#    def plot_gene_projections(self, geneset_name):
#        """
#        Represent the pathway genes in the pca space.
#        Against null distribution , i.e. genes in the pca space of all the random genesets.
#        """
#        
#        import numpy as np
#        import matplotlib.pyplot as plt
#        import seaborn as sns
#
#        sns.set(style="darkgrid")
#        sns.set_palette("Pastel1")
#
#        # Assuming roma_result is already loaded and contains the required matrices
#        roma_result = roma.adata.uns['ROMA'][geneset_name]
#        projections_1 = roma_result.projections_1
#        projections_2 = roma_result.projections_2
#        null_projections = roma_result.null_projections
#        null_projections_flat = null_projections.reshape(-1, 2)
#
#        # Setting up the plot
#        plt.figure(figsize=(10, 8))
#
#        # Setting the axis labels
#        plt.axhline(0,color='k') # x = 0
#        plt.axvline(0,color='k') # y = 0
#        plt.xlabel('PC1')
#        plt.ylabel('PC2')
#
#        # Plotting the points from all matrices in null_projections in hollowish Tr. Blue color
#        sns.scatterplot(x=null_projections_flat[:, 0], y=null_projections_flat[:, 1], color='dodgerblue', label='Null Projections', edgecolor='black', marker='o', alpha=0.2)
#
#        # Plotting the points from projections_2 in Red color (Tr. Red)
#        sns.scatterplot(x=projections_1, y=projections_2, color='red', label=f'{geneset_name}', edgecolor='black')
#
#        # Adding grid
#        plt.grid(True)
#
#        # Adding title
#        plt.title(f'{geneset_name} and Null distribution in PCA space ')
#
#        # Removing the legend duplicates
#        handles, labels = plt.gca().get_legend_handles_labels()
#        by_label = dict(zip(labels, handles))
#        plt.legend(by_label.values(), by_label.keys())
#
#        # Showing the plot
#        plt.show()
#        return
#