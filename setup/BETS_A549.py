#!/usr/bin/env python
# coding: utf-8

# ### A549 data
# 
# This notebook prepares a glucocorticoid stimulus time-course with matched overexpression data in A549 cells, mostly from the BETS paper. Here we preprocess the dataset 
# and carry out a simple exploration in scanpy. 

import warnings
warnings.filterwarnings('ignore')
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.preprocessing
import anndata 
import os
import seaborn as sns
from collections import Counter

# local
import importlib
import sys
sys.path.append("setup")
import ingestion
importlib.reload(ingestion)
import load_perturbations
load_perturbations.set_data_path("../../perturbation_data/perturbations")

#      visualization settings
plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 300


# Universal
geneAnnotationPath = "../accessory_data/gencode.v35.annotation.gtf.gz"       # Downloaded from https://www.gencodegenes.org/human/release_35.html
humanTFPath = "../accessory_data/humanTFs.csv"                               # Downloaded from http://humantfs.ccbr.utoronto.ca/download.php
humanEpiPath = "../accessory_data/epiList.csv"                               # Downloaded from https://epifactors.autosome.org/description 

finalDataFileFolder = "../perturbations/BETS_A549"
sc.settings.figdir = finalDataFileFolder
os.makedirs(os.path.join(finalDataFileFolder, "test"),  exist_ok=True)
os.makedirs(os.path.join(finalDataFileFolder, "train"), exist_ok=True)

# Combine into anndata to keep everything together
overexpression = anndata.AnnData(
    X   = pd.read_csv("../not_ready/BETS_A549/GSE144660_overexpression.tsv", index_col = 0, header = 0).T.sort_index(),
    obs = pd.read_csv("../not_ready/BETS_A549/GSE144660_metadata.tsv", sep="\t", header = None, index_col = 0).sort_index()
)
dex_away = anndata.AnnData(
    X   = pd.read_csv("../not_ready/BETS_A549/GSE144662_post_dex_removal.tsv", index_col = 0, header = 0).T.sort_index(),
    obs = pd.read_csv("../not_ready/BETS_A549/GSE144662_metadata.tsv", sep="\t", header = None, index_col = 0).sort_index()
)
dex = anndata.AnnData(
    X = pd.concat(
        [
            pd.read_csv("../not_ready/BETS_A549/" + f, index_col = 0, header = 0).T for f in [
                "GSE91305_dex-0h.tsv",
                "GSE91198_dex-0.5h.tsv",
                "GSE91311_dex-1h.tsv",
                "GSE91358_dex-2h.tsv",
                "GSE91303_dex-3h.tsv",
                "GSE91243_dex-4h.tsv",
                "GSE91281_dex-5h.tsv",
                "GSE91229_dex-6h.tsv",
                "GSE91255_dex-7h.tsv",
                "GSE91284_dex-8h.tsv",
                "GSE91222_dex-10h.tsv",
                "GSE91212_dex-12h.tsv",
            ]
        ]
    ).sort_index(),
    obs = pd.read_csv("../not_ready/BETS_A549/dex_metadata.tsv", index_col = 1, sep = " ", header = None).sort_index(),
)
assert all(dex_away.var_names==dex.var_names), "feature names match"
assert all(dex.var_names==overexpression.var_names), "feature names match"
dex.var_names_make_unique()
dex_away.var_names_make_unique()
overexpression.var_names_make_unique()
dex.obs.columns = ["timepoint"]
dex_away.obs.columns = ["timepoint"]
overexpression.obs.columns = ["perturbation", "treatment", "timepoint"]
dex.obs["treatment"] = "dexamethasone"
dex_away.obs["treatment"] = "dexamethasone removal"
expression_quantified = dict()
expression_quantified["train"] = anndata.concat([dex, dex_away])
expression_quantified["test"]  = overexpression
expression_quantified["test"].obs["timepoint"] = expression_quantified["test"].obs["timepoint"].str.replace("h", "").astype(float)
expression_quantified["train"].obs["timepoint"] = expression_quantified["train"].obs["timepoint"].str.replace("h", "").astype(float)
# Label with perturbed genes
expression_quantified["train"].obs["perturbation"] = "control"
expression_quantified["train"].obs["is_control"] = True
expression_quantified["test"].obs["is_control"] = expression_quantified["test"].obs["perturbation"]=="GFP"
expression_quantified["test"].obs["perturbation"] = expression_quantified["test"].obs["perturbation"].replace("OCT4", "POU5F1")


for k in expression_quantified.keys():
    # Label with celltypes
    expression_quantified[k].obs["cell_type"] = "A549"
    expression_quantified[k].obs["perturbation_type"] = "overexpression"

    # Fix some annoying dtype issues
    expression_quantified[k].obs["is_control_int"] = [float(x) for x in expression_quantified[k].obs["is_control"]]
    expression_quantified[k].obs_names = [str(s) for s in expression_quantified[k].obs_names] 

    # Examine mean-variance relationship
    x = expression_quantified[k].X.mean(axis=0)
    y = expression_quantified[k].X.std(axis=0)
    plt.figure()
    fig = sns.scatterplot(x=x,y=y)
    plt.title("Mean-variance relationship")
    fig.figure.savefig(f"../perturbations/BETS_A549/{k}/mean_var.pdf", bbox_inches='tight')

    # Normalize
    expression_quantified[k].raw = expression_quantified[k].copy()
    expression_quantified[k].raw.obs = expression_quantified[k].obs
    expression_quantified[k].X = ingestion.deseq2Normalization( expression_quantified[k].raw.X.T ).T

    # Examine mean-variance relationship
    x = expression_quantified[k].raw.X.std(axis=0)
    y = expression_quantified[k].X.std(axis=0)
    plt.figure()
    fig = sns.scatterplot(x=x,y=y)
    plt.title("Mean-variance relationship after deseq2 normalization + natural log")
    fig.figure.savefig(f"../perturbations/BETS_A549/{k}/transformed_mean_var.pdf", bbox_inches='tight')

# Study perturbation effects
print("Checking perturbation effects")
controls = ["GFP"]
perturbed_genes = set.union(*[set(p.split(",")) for p in expression_quantified["test"].obs["perturbation"]]).difference(set(controls))
perturbed_and_measured_genes     = perturbed_genes.intersection(expression_quantified["test"].var_names)
perturbed_but_not_measured_genes = perturbed_genes.difference(expression_quantified["test"].var_names)
print("These genes were perturbed and measured:")
print(perturbed_and_measured_genes)
print("These genes were perturbed but not measured:")
print(perturbed_but_not_measured_genes)
expression_quantified["test"].uns["perturbed_and_measured_genes"]     = list(perturbed_and_measured_genes)
expression_quantified["test"].uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
expression_quantified["test"].uns["perturbations_overlap"] = False
expression_quantified["train"].uns["perturbed_and_measured_genes"]     = []
expression_quantified["train"].uns["perturbed_but_not_measured_genes"] = []
expression_quantified["train"].uns["perturbations_overlap"] = False
expression_quantified["test"] = ingestion.describe_perturbation_effect( adata = expression_quantified["test"], 
                                                                       perturbation_type="knockdown", 
                                                                       multiple_genes_hit = False)
expression_quantified["train"] = ingestion.describe_perturbation_effect( adata = expression_quantified["train"], 
                                                                       perturbation_type="knockdown", 
                                                                       multiple_genes_hit = False)
expression_quantified["test"].obs
status, logFC = ingestion.checkConsistency(adata = expression_quantified["test"], 
                                           perturbationType="overexpression", 
                                           verbose=False)
print(Counter(status))
expression_quantified["test"].obs["consistentW/Perturbation"] = status
expression_quantified["test"].obs["logFC"] = logFC
correlations = ingestion.computeCorrelation(expression_quantified["test"], verbose=True)
expression_quantified["test"].obs["spearmanCorr"] = correlations[0]
expression_quantified["test"].obs[ "pearsonCorr"] = correlations[1]

# Some basic exploration of results
print("Data exploration")
for t in ("train", "test"):
    print(f"Exploring {t}")
    sc.pp.log1p(expression_quantified[t])
    sc.pp.highly_variable_genes(expression_quantified[t], flavor = "seurat_v3", n_top_genes=expression_quantified[t].shape[1])
    with warnings.catch_warnings():
        sc.tl.pca(expression_quantified[t], n_comps=10)
    sc.pp.neighbors(expression_quantified[t])
    sc.tl.louvain(expression_quantified[t])    
    sc.tl.umap(expression_quantified[t])

# Regress out suspected nuisance effects
#
expression_quantified[f"train_uncorrected"] = expression_quantified["train"].copy()
expression_quantified[ f"test_uncorrected"] = expression_quantified[ "test"].copy()
sc.pp.regress_out(expression_quantified["test"], "louvain")
sc.pp.regress_out(expression_quantified["train"], ["louvain", "treatment"])
expression_quantified["both"] = anndata.concat(expression_quantified)
for t in ("both", "test_uncorrected", "train_uncorrected"):
    sc.pp.log1p(expression_quantified[t])
    sc.pp.highly_variable_genes(expression_quantified[t], flavor = "seurat_v3", n_top_genes=expression_quantified[t].shape[1])
    with warnings.catch_warnings():
        sc.tl.pca(expression_quantified[t], n_comps=20)
    sc.pp.neighbors(expression_quantified[t])
    sc.tl.louvain(expression_quantified[t])
    sc.tl.umap(expression_quantified[t])

for t in ("train", "test", "both", "test_uncorrected", "train_uncorrected"):
    print(f"Plotting {t}")
    vars_to_show = ["timepoint", "perturbation", "louvain", "treatment"]
    for v in vars_to_show:
        fig = sc.pl.umap(expression_quantified[t], color = v, show = False, legend_loc='on data')
        try:
            os.makedirs(f"../perturbations/BETS_A549/{t}", exist_ok=True)
            fig.figure.savefig(f"../perturbations/BETS_A549/{t}/{v}.pdf", bbox_inches='tight')
        except Exception as e:
            print(f"Plots failed with error {repr(e)}")
    clusters = expression_quantified[t].obs[["louvain", "perturbation"]].value_counts().reset_index()
    clusters = clusters.groupby("louvain")["perturbation"].agg(func = lambda x: " ".join(np.sort(x)))
    clusters.to_csv(f"../perturbations/BETS_A549/{t}/clusters.csv")
    expression_quantified[t].write_h5ad(os.path.join("../perturbations/BETS_A549", f"{t}.h5ad"))

load_perturbations.check_perturbation_dataset("BETS_A549")