# Load packages
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import anndata as ad


# Set display options
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_columns', 100)

# Set figure params to now show frame on umaps
sc.settings.set_figure_params(dpi=120, format='svg', transparent=True)
sc.set_figure_params(figsize=(8, 8))



# Check working directory
os.getcwd()
os.chdir('SET TO YOUR WORKING DIRECTORY HERE')

# SET PATH TO SAVE dictionary
sc.settings.figdir = 'images/'





# Load lineage objects
lin_list =['myeloid', 'epithelial', 'lymphoid', 'stroma']

lin_adata_list=[]

for i in range(len(lin_list)):
    adata = sc.read_h5ad(f'data/hgca_{lin_list[i]}_published.h5ad')
    lin_adata_list.append(adata)


sc.pl.umap(
    lin_adata_list[0],
    color='hgca_celltype_v1',
    frameon=False,
    s=6,
    save=f'_{lin_list[0]}.svg'
    )


sc.pl.umap(
    lin_adata_list[1],
    color='hgca_celltype_v1',
    frameon=False,
    s=4,
    save=f'_{lin_list[1]}.svg'
    )


sc.pl.umap(
    lin_adata_list[2],
    color='hgca_celltype_v1',
    frameon=False,
    s=4,
    save=f'_{lin_list[2]}.svg'
    )



sc.pl.umap(
    lin_adata_list[3],
    color='hgca_celltype_v1',
    frameon=False,
    s=4,
    save=f'_{lin_list[3]}.svg'
    )








