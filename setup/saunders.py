#!/usr/bin/env python
# coding: utf-8

# ### Zebrafish example
# 
# This notebook prepares a massive zebrafish sci-RNA dataset with a densely sampled timeseries and 20 knockout experiments, 
# from [Saunders et al. 2022](https://www.biorxiv.org/content/10.1101/2022.08.04.502764v1). Here we preprocess the dataset 
# and carry out a simple exploration in scanpy. We will:
# 
# - add specific metadata needed by our benchmarking framework
# - check to see if KO'd genes have lower transcript levels
# - check for low quality cells
# - Make metacells or otherwise aggregate
# - normalize and log-transform 
# - exclude low-expressed genes and select HVG
# - visualize
# 
import os
import warnings
warnings.filterwarnings('ignore')
import regex as re
import os
import gc
import shutil
import importlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata 
import seaborn as sns
from scipy.stats import spearmanr as spearmanr
from IPython.display import display, HTML

import os, sys
import itertools as it
from scipy.stats import spearmanr, pearsonr, rankdata, f_oneway
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# local
import importlib
import sys
sys.path.append("setup")
import ingestion
importlib.reload(ingestion)

#      visualization settings
plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 300


# Universal
geneAnnotationPath = "../accessory_data/gencode.v35.annotation.gtf.gz"       # Downloaded from https://www.gencodegenes.org/human/release_35.html
humanTFPath = "../accessory_data/humanTFs.csv"                               # Downloaded from http://humantfs.ccbr.utoronto.ca/download.php
humanEpiPath = "../accessory_data/epiList.csv"                               # Downloaded from https://epifactors.autosome.org/description 

finalDataFileFolder = "perturbations/saunders"
sc.settings.figdir = finalDataFileFolder

# ### Combine into anndata to keep everything together
print("Setting up AnnData objects")
expression_quantified = {}
try:
    expression_quantified["train"] = sc.read_h5ad(os.path.join("not_ready/saunders", "train_working.h5ad"))
except:
    expression_quantified["train"] = anndata.AnnData(sc.read_mtx("not_ready/saunders/GSE202639_reference_raw_counts.mtx.gz").T, dtype=np.float32)
    expression_quantified["train"].var = pd.read_csv("not_ready/saunders/GSE202639_reference_gene_metadata.csv.gz")
    expression_quantified["train"].obs = pd.read_csv("not_ready/saunders/GSE202639_reference_cell_metadata.csv.gz")
    os.makedirs(finalDataFileFolder, exist_ok = True)
    expression_quantified["train"].write_h5ad(os.path.join("not_ready/saunders", "train_working.h5ad"))

try:
    expression_quantified["test"] = sc.read_h5ad(os.path.join("not_ready/saunders", "test_working.h5ad"))
except:
    expression_quantified["test"] = sc.AnnData(sc.read_mtx("not_ready/saunders/GSE202639_zperturb_full_raw_counts.mtx.gz").T, dtype=np.float32)         
    expression_quantified["test"].var = pd.read_csv("not_ready/saunders/GSE202639_zperturb_full_gene_metadata.csv.gz")
    expression_quantified["test"].obs = pd.read_csv("not_ready/saunders/GSE202639_zperturb_full_cell_metadata.csv.gz")
    os.makedirs(finalDataFileFolder, exist_ok = True)
    expression_quantified["test"].write_h5ad(os.path.join("not_ready/saunders", "test_working.h5ad"))

expression_quantified["train"].obs["perturbation"] = "control"
expression_quantified["train"].obs["is_control"] = True

expression_quantified["test"].obs["perturbation"] = expression_quantified["test"].obs["gene_target"].str.replace(
    "-", ",").str.replace(           # our code expects comma-separated gene lists
    "ctrl,", "").str.replace(        # convert ctrl,blah to blah
    "inj", "control").str.replace(   # injection control; this is not a gene name
    "wnt8", "wnt8a")                 # this gene name has a typo in it. paper says it's wnt8a.
expression_quantified["test"].obs["is_control"] = [g=="ctrl-inj" for g in expression_quantified["test"].obs["gene_target"]]

for t in ("train", "test"):
    expression_quantified[t].obs["is_control_int"] = [float(x) for x in expression_quantified[t].obs["is_control"]]
    expression_quantified[t].obs_names = [str(s) for s in expression_quantified[t].obs_names] 
    expression_quantified[t].var.index = expression_quantified[t].var["gene_short_name"]


# ### Remove low-quality cells and low-expressed genes
# 
# We remove unhashed cells where the timepoint or embryo is unknown. 
# The data are already filtered on depth and mitochondrial content.
print("Removing unhashed cells")
for t in ("train", "test"):
    for f in ("timepoint", "embryo", "cell_type_sub"):
        expression_quantified[t] = expression_quantified[t][pd.notnull(expression_quantified[t].obs[f]),:]
    sc.pp.filter_genes(expression_quantified[t], min_counts=1000, inplace=True)
    print(f"{t} shape:")
    print(expression_quantified[t].shape)

# ### Aggregate
print("Aggregating")
for t in ("train", "test"):
    try:
        X = sc.read_h5ad(os.path.join("not_ready/saunders", f"{t}_aggregated.h5ad"))
    except:
        X = ingestion.aggregate_by_perturbation(
            adata = expression_quantified[t],
            group_by=["perturbation", "tissue", "germ_layer", "cell_type_broad", "cell_type_sub", "embryo", 'timepoint'],
            use_raw = False,
        )
    print(f"done {t}")
    expression_quantified[t] = X
    expression_quantified[t].write_h5ad(os.path.join("not_ready/saunders", f"{t}_aggregated.h5ad"))
    gc.collect()
# ### Normalize
print("Normalizing")
for t in ("train", "test"):
    expression_quantified[t].raw = expression_quantified[t].copy()
    expression_quantified[t].raw.obs = expression_quantified[t].obs
    sc.pp.normalize_total(expression_quantified[t], target_sum=1e4)

# Study perturbation effects
print("Checking perturbation effects")
controls = ["control"]
perturbed_genes = set.union(*[set(p.split(",")) for p in expression_quantified["test"].obs["perturbation"]]).difference(set(controls))
perturbed_and_measured_genes = perturbed_genes.intersection(expression_quantified["test"].var.index)
perturbed_but_not_measured_genes = perturbed_genes.difference(expression_quantified["test"].var.index)
print("These genes were perturbed and measured:")
print(perturbed_and_measured_genes)
print("These genes were perturbed but not measured:")
print(perturbed_but_not_measured_genes)
expression_quantified["test"].uns["perturbed_and_measured_genes"]     = list(perturbed_and_measured_genes)
expression_quantified["test"].uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
expression_quantified["test"].uns["perturbations_overlap"] = True
expression_quantified["test"] = ingestion.describe_perturbation_effect(expression_quantified["test"], "knockout", multiple_genes_hit = True)
status, logFC = ingestion.checkConsistency(expression_quantified["test"], 
                                           perturbationType="knockout", 
                                           group="embryo",
                                           verbose=False)
print(Counter(status))
expression_quantified["test"].obs["consistentW/Perturbation"] = status
expression_quantified["test"].obs["logFC"] = logFC
correlations = ingestion.computeCorrelation(expression_quantified["test"], verbose=True, group="embryo")
expression_quantified["test"].obs["spearmanCorr"] = correlations[0]
expression_quantified["test"].obs[ "pearsonCorr"] = correlations[1]

# Some basic exploration of results
print("Data exploration")
for t in ("train", "test"):
    print(f"Exploring {t}")
    sc.pp.log1p(expression_quantified[t])
    sc.pp.highly_variable_genes(expression_quantified[t], flavor = "seurat_v3", n_top_genes=expression_quantified[t].shape[1])
    with warnings.catch_warnings():
        sc.tl.pca(expression_quantified[t], n_comps=5)
    sc.pp.neighbors(expression_quantified[t])
    sc.tl.umap(expression_quantified[t])
    print(f"Plotting {t}")
    vars_to_show = ["embryo", "timepoint", "perturbation", "cell_type_broad", "cell_type_sub"]
    figs = sc.pl.umap(expression_quantified[t], color = vars_to_show, show = False)
    try:
        os.makedirs(f"perturbations/saunders/{t}", exist_ok=True)
        [fig.figure.savefig(f"perturbations/saunders/{t}/{v}.pdf") for fig,v in zip(figs, vars_to_show)]
    except Exception as e:
        print(f"Plots failed with error {repr(e)}")
    expression_quantified[t].write_h5ad(os.path.join("perturbations/saunders", f"{t}.h5ad"))
