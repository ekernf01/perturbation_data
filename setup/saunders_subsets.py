#!/usr/bin/env python
# coding: utf-8

# ### Zebrafish example
# 
# This notebook selects subsets of the Saunders zebrafish data -- see also saunders.py.
# 
import warnings
warnings.filterwarnings('ignore')
import anndata
import importlib
import scanpy as sc
import os
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

# local
import importlib
import sys
sys.path.append("setup")
import ingestion
importlib.reload(ingestion)


expression_quantified = {}
expression_quantified["train"] = sc.read_h5ad(os.path.join("../perturbations/saunders", "train.h5ad"))
expression_quantified["test"]  = sc.read_h5ad(os.path.join("../perturbations/saunders", "test.h5ad"))
perturbation_controls = expression_quantified["test"].obs["is_control"] 
expression_quantified["test"].obs.loc[perturbation_controls, "summary"] = "control"
expression_quantified["test"].obs.loc[~perturbation_controls, "summary"] = "perturbation"
expression_quantified["train"].obs["summary"] = "timeseries"

vars_to_show = ["embryo", "timepoint", "perturbation", "cell_type", "cell_type_broad", "cell_type_sub", "cell_count", 'total_counts', 'log1p_total_counts',  "summary"]
for t in ["train", "test"]:
    expression_quantified[t].obs[
        ["timepoint", "cell_type_broad", "cell_type", "cell_type_sub"]
    ].value_counts(
    ).reset_index(
    ).sort_values(
        ["timepoint", "cell_type_broad", "cell_type", "cell_type_sub"]
    ).to_csv(
        f"saunders_cell_types_{t}.csv"
    )

subsets = {
    "endoderm": ["liver", "intestine", "pancreas", "pharynx"], # saunders does not include the pharynx, but it's just a little more anterior than the liver
    "axial_mesoderm": ["notochord", "mesodermal progenitor cells (contains PSM)"], # vacuolar and sheath cells are not separately annotated
    "pigment_cells":["iridophore", "melanophore", "xanthophore", "cranial neural crest", "neural crest + pigment cell progenitor"], 
    "blood": ["myeloid cell", "red blood cell"], # No thrombocytes
}

for subset_name, cell_types in subsets.items():
    print(subset_name)
    for t in ["train", "test", "both"]:
        print(t)
        if t=="both":
            obs1 = expression_quantified["train"][expression_quantified["train"].obs["cell_type"].isin(cell_types), :].obs.copy()
            obs2 = expression_quantified["test"][expression_quantified["test"].obs["cell_type"].isin(cell_types), :].obs.copy()
            X = csr_matrix(np.concatenate([
                    expression_quantified["train"][expression_quantified["train"].obs["cell_type"].isin(cell_types), :].X.toarray(),
                    expression_quantified["test"][expression_quantified["test"].obs["cell_type"].isin(cell_types), :].X.toarray(),
                ]))
            subset = anndata.AnnData(
                X = X,
                var = expression_quantified["train"].var,
                obs = pd.concat(
                    [obs1, obs2],
                    axis=0, 
                    ignore_index=True
                )
            )
            del X
        else:
            subset = expression_quantified[t][expression_quantified[t].obs["cell_type"].isin(cell_types), :]
        if t=="train":
            sc.pp.highly_variable_genes(subset, n_bins=50, n_top_genes = subset.var.shape[0], flavor = "seurat_v3" )
        else:
            subset.var["highly_variable"] = [int(i) for i in previous_hvg.copy()]
        sc.tl.pca(subset, svd_solver='arpack', n_comps=10)
        sc.pp.neighbors(subset, n_neighbors=4, n_pcs=10)
        sc.tl.umap(subset)
        os.makedirs(f"../perturbations/saunders_{subset_name}/{t}", exist_ok=True)
        for v in vars_to_show:
            os.makedirs(f"../perturbations/saunders/{t}", exist_ok=True)
            fig = sc.pl.umap(subset, color = v, show = False)
            fig.figure.savefig(f"../perturbations/saunders_{subset_name}/{t}/{v}.pdf", bbox_inches="tight")
        subset.write(f"../perturbations/saunders_{subset_name}/{t}.h5ad")
        previous_hvg = subset.var["highly_variable_rank"].values.copy()
