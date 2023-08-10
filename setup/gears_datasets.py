#!/usr/bin/env python
# coding: utf-8

# ### Early Perturb-seq examples
# 
# This notebook prepares early Perturb-seq examples from Dixit et. al., Adamson et al, and Norman et al.
# The number of perturbations is rather small compared to some more recently available 
# data (24 distinct genes in Dixit et al mouse BMDC, 82 genes in Adamson et al K562 UPR perturb-seq, and ~150 genes plus 
# combos in Norman et al.), but GEARS gets favorable results on these data. Because of that, we are 
# motivated to test various methods. 
# We use the GEARS preprocessing, which is to select 5k variable genes and log1p-transform.
# 
import warnings
warnings.filterwarnings('ignore')
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata 
import os
import altair as alt
from collections import Counter
import shutil 
from gears import PertData
import ingestion
# Assume we run from Eric's whole-project-all-repos directory or location of this script.
try:
    os.chdir("perturbation_data/setup")
except FileNotFoundError:
    pass
pert_types = {
    "dixit": "knockout",
    "adamson": "knockdown",
    "norman": "overexpression",
}
for dataset in ("norman", "dixit", "adamson"):
    finalDataFileFolder = f"../perturbations/{dataset}"
    os.makedirs(finalDataFileFolder, exist_ok = True)
    try:
        expression_quantified = sc.read_h5ad(os.path.join(finalDataFileFolder, "test.h5ad"))
    except FileNotFoundError:
        pert_data = PertData('./gears_data_temp')
        pert_data.load(data_name = dataset)
        expression_quantified = pert_data.adata
    expression_quantified.var.index = expression_quantified.var["gene_name"]
    expression_quantified.obs["perturbation"]   =  \
        expression_quantified.obs["condition"].str.replace("+ctrl", "").str.replace("ctrl+", "").str.replace("+", ",")
    expression_quantified.obs["is_control"]     = expression_quantified.obs["perturbation"] == "ctrl"
    expression_quantified.obs["is_control_int"] = expression_quantified.obs["is_control"].astype("int")
    expression_quantified.obs["spearmanCorr"] = 0
    expression_quantified.uns["perturbations_overlap"] = True if dataset=="norman" else False
    expression_quantified = ingestion.describe_perturbation_effect(
        expression_quantified, 
        perturbation_type=pert_types[dataset], 
        multiple_genes_hit=True if dataset=="norman" else False,
    )
    sc.pp.highly_variable_genes(expression_quantified, flavor="seurat_v3", n_top_genes=expression_quantified.var.shape[0])
    breakpoint()
    perts = set().union(*[set(p.split(",")) for p in expression_quantified.obs['perturbation'].unique()])
    perturbed_genes = perts.difference({"ctrl"})
    perturbed_and_measured_genes = perturbed_genes.intersection(expression_quantified.var.index)
    perturbed_but_not_measured_genes = perturbed_genes.difference(expression_quantified.var.index)    
    print("These genes were perturbed and measured:")
    print(perturbed_and_measured_genes)
    print("These genes were perturbed but not measured:")
    print(perturbed_but_not_measured_genes)
    expression_quantified.uns["perturbed_and_measured_genes"] = list(perturbed_and_measured_genes)
    expression_quantified.uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
    expression_quantified.raw = expression_quantified.copy()
    expression_quantified.write_h5ad(os.path.join(finalDataFileFolder, "test.h5ad"))
    # Remove GEARS temp files
    try:
        shutil.rmtree("./gears_data_temp")
    except FileNotFoundError:
        pass
