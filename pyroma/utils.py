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

def integrate_projection_results(global_adata, cell_type, gene_set_results, ct_label):
    import pandas as pd

    # Boolean mask for cells of the specified cell type.
    subset_mask = global_adata.obs[ct_label] == cell_type
    # Get the indices (order preserved) for these cells.
    subset_indices = global_adata.obs.index[subset_mask]
    if not 'pyroma_scores' in global_adata.uns:
        global_adata.uns['pyroma_scores'] = pd.DataFrame(index=global_adata.obs.index)

    for pathway, result in gene_set_results.items():
        # Get computed projection values for this pathway
        projections = result.svd.components_[0]
        if len(projections) != len(subset_indices):
            raise ValueError(
                f"Length mismatch for cell type '{cell_type}' and pathway '{pathway}': "
                f"found {len(subset_indices)} cells in adata vs. {len(projections)} projection values."
            )
        # Create a new Series (one value per cell in global_adata) initialized to 0.0
        new_col = pd.Series(0.0, index=global_adata.obs.index)
        # Fill in the computed projection values for the subset cells (order preserved)
        new_col.loc[subset_indices] = projections
        
        # Create the new column name as "{cell_type}_{pathway}"
        col_name = f"{cell_type}|{pathway}"
        global_adata.uns['pyroma_scores'][col_name] = new_col
    
    print(f"Projection columns for cell type '{cell_type}' integrated successfully.")