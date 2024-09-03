class plotting:
    def __init__(self):
        self.geneset_name = None
        self.adata = None
    
    def gene_weights(self, geneset_name):
        """
        Plotting the gene weights.
        """
        # TODO: make it adaptable to the size of the geneset - so all the genes fit
        # TODO: initialize with  roma.adata 
        import matplotlib.pyplot as plt
        import pandas as pd 
        import seaborn as sns

        fig, ax1 = plt.subplots(1, 1, figsize=(7.5,16))
        fig.tight_layout()
        #sns.set(style="darkgrid")
        #sns.set_palette("Pastel1")
        plt.grid(color='white', lw = 0.5, axis='x')

        roma_result = self.adata.uns['ROMA'][geneset_name]
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

    def gene_projections(self, geneset_name):
        """
        Represent the pathway genes in the pca space.
        Against null distribution , i.e. genes in the pca space of all the random genesets.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="darkgrid")
        sns.set_palette("Pastel1")

        roma_result = self.adata.uns['ROMA'][geneset_name]
        projections_1 = roma_result.projections_1
        projections_2 = roma_result.projections_2
        null_projections = roma_result.null_projections
        null_projections_flat = null_projections.reshape(-1, 2)


        plt.figure(figsize=(10, 8))
        plt.axhline(0,color='k') # x = 0
        plt.axvline(0,color='k') # y = 0
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        sns.scatterplot(x=null_projections_flat[:, 0], y=null_projections_flat[:, 1], color='dodgerblue', label='Null Projections', edgecolor='black', marker='o', alpha=0.2)
        sns.scatterplot(x=projections_1, y=projections_2, color='red', label=f'{geneset_name}', edgecolor='black')
        plt.grid(True)
        plt.title(f'{geneset_name} and Null distribution in PCA space ')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
        return
