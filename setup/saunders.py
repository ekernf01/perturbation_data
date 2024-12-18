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
# - Aggregate within each embryo and cell type
# - normalize and log-transform 
# - exclude low-expressed genes and select HVG
# - visualize
# 
import warnings
warnings.filterwarnings('ignore')
import gc
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata 
import os
import altair as alt
from collections import Counter

# local
import importlib
import sys
sys.path.append("setup")
import ingestion
importlib.reload(ingestion)

# Universal
geneAnnotationPath = "../accessory_data/gencode.v35.annotation.gtf.gz"       # Downloaded from https://www.gencodegenes.org/human/release_35.html
humanTFPath = "../accessory_data/humanTFs.csv"                               # Downloaded from http://humantfs.ccbr.utoronto.ca/download.php
humanEpiPath = "../accessory_data/epiList.csv"                               # Downloaded from https://epifactors.autosome.org/description 

finalDataFileFolder = "perturbations/saunders"
sc.settings.figdir = finalDataFileFolder

def load_data():
    # ### Combine into anndata to keep everything together
    print("Setting up AnnData objects")
    expression_quantified = {}
    try:
        expression_quantified["train"] = sc.read_h5ad(os.path.join("../not_ready/saunders", "train_working.h5ad"))
        print("h5ad found")
    except:
        expression_quantified["train"] = anndata.AnnData(sc.read_mtx("../not_ready/saunders/GSE202639_reference_raw_counts.mtx.gz").T, dtype=np.float32)
        expression_quantified["train"].var = pd.read_csv("../not_ready/saunders/GSE202639_reference_gene_metadata.csv.gz")
        expression_quantified["train"].obs = pd.read_csv("../not_ready/saunders/GSE202639_reference_cell_metadata.csv.gz")
        os.makedirs(finalDataFileFolder, exist_ok = True)
        expression_quantified["train"].write_h5ad(os.path.join("../not_ready/saunders", "train_working.h5ad"))

    try:
        expression_quantified["test"] = sc.read_h5ad(os.path.join("../not_ready/saunders", "test_working.h5ad"))
        print("h5ad found")
    except:
        expression_quantified["test"] = sc.AnnData(sc.read_mtx("../not_ready/saunders/GSE202639_zperturb_full_raw_counts.mtx.gz").T, dtype=np.float32)         
        expression_quantified["test"].var = pd.read_csv("../not_ready/saunders/GSE202639_zperturb_full_gene_metadata.csv.gz")
        expression_quantified["test"].obs = pd.read_csv("../not_ready/saunders/GSE202639_zperturb_full_cell_metadata.csv.gz")
        os.makedirs(finalDataFileFolder, exist_ok = True)
        expression_quantified["test"].write_h5ad(os.path.join("../not_ready/saunders", "test_working.h5ad"))

    # Add perturbation metadata
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

    # Clean up data type for cell id's
    for t in ("train", "test"):
        expression_quantified[t].obs_names = [str(s) for s in expression_quantified[t].obs_names] 

    # We remove unhashed cells where the timepoint or embryo is unknown. 
    # The data are already filtered on depth and mitochondrial content (Saunders et al. 2023 Method, section "Sequencing, read processing and cell filtering").
    print("Removing unhashed cells")
    for t in ("train", "test"):
        for f in ("timepoint", "embryo", "cell_type_sub"):
            expression_quantified[t] = expression_quantified[t][pd.notnull(expression_quantified[t].obs[f]),:]

    return expression_quantified

def get_aggregated_data(expression_quantified = {"train": None, "test": None}):
    # ### Aggregate
    # The methods we evaluate are designed for 10x data, which is deeper. We aggregate the data within each cell type and embryo to make it more similar to 10x data.
    print("Aggregating")
    try:
        for t in ("train", "test"):
            expression_quantified[t] = sc.read_h5ad(os.path.join("../not_ready/saunders", f"{t}_aggregated.h5ad"))
        print("Aggregated data found.")
    except:
        expression_quantified = load_data()
        for t in ("train", "test"):
            X = ingestion.aggregate_by_perturbation(
                adata = expression_quantified[t],
                group_by=["perturbation", "tissue", "germ_layer", "cell_type_broad", "cell_type_sub", "embryo", 'timepoint'],
                use_raw = False,
            )
            print(f"done {t}")
            expression_quantified[t] = X
            expression_quantified[t].write_h5ad(os.path.join("../not_ready/saunders", f"{t}_aggregated.h5ad"))
        gc.collect()
    return expression_quantified

# ### Clean up gene names.
# important detail: wnt8a appears twice in the tx annotation (bicistronic; 2 ORFs). The two ORFs are immediately adjacent.
# Saunders et al. targeted both ORFs (Table S4). We will merge these into one gene.
def merge_duplicate_genes(adata):
    dupes = adata.var_names.value_counts().to_frame(name="frequency").query("frequency > 1")
    print("Aggregating counts within genes whose name occurs more than once:")
    print(dupes)
    to_remove = []
    for g in dupes.index:
        print(".", end="", flush=True)
        all_occurrences = adata.var_names == g
        first_occurrence = np.where(all_occurrences)[0][0]
        later_occurrences = np.where(all_occurrences)[0][1:]
        to_remove = to_remove + list(later_occurrences)
        adata.X[:, first_occurrence] = adata.X[:, all_occurrences].sum(axis=1)
    adata = adata[:, [i for i in range(adata.n_vars) if i not in to_remove]]
    print("")
    return adata


expression_quantified = get_aggregated_data()
for t in ("train", "test"):
    expression_quantified[t].var_names = [str(g) for g in expression_quantified[t].var["gene_short_name"]]
    expression_quantified[t] = merge_duplicate_genes(expression_quantified[t])

# Keep the same genes in train and test
shared_genes = set(list(expression_quantified["test"].var_names)).intersection(
    set(list(expression_quantified["train"].var_names)))
shared_genes = list(shared_genes)
print(f"Train num genes: {len(expression_quantified['train'].var_names)}")
print(f"Test num genes: {len(expression_quantified['test'].var_names)}")
print(f"Shared genes: {len(shared_genes)}")
for t in ("train", "test"):
    expression_quantified[t] = expression_quantified[t][:, shared_genes]

# ### Normalize
print("Normalizing")
for t in ("train", "test"):
    try:
        expression_quantified[t].X = expression_quantified[t].X.to_csr()
    except AttributeError:
        pass
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
expression_quantified["train"].obs["expression_level_after_perturbation"] = np.nan # This will be ignored anyway
expression_quantified["train"].obs["perturbation_type"] = "knockout" # Since predictions will be compared to KO's in the test set

# Some basic exploration of results
print("Data exploration")
for t in ("train", "test"):
    print(f"Exploring {t}")
    # Fix up a few key metadata fields
    sc.pp.calculate_qc_metrics(expression_quantified[t], inplace=True)
    expression_quantified[t].obs["timepoint"] = expression_quantified[t].obs["timepoint"].astype("str").astype(float)
    expression_quantified[t].obs["cell_type"] = expression_quantified[t].obs["cell_type_broad"]
    expression_quantified[t].obs["cell_count"] = expression_quantified[t].obs["count"]
    sc.pp.log1p(expression_quantified[t])
    sc.pp.highly_variable_genes(expression_quantified[t], flavor = "seurat_v3", n_top_genes=expression_quantified[t].shape[1])
    with warnings.catch_warnings():
        sc.tl.pca(expression_quantified[t], n_comps=100)
    sc.pp.neighbors(expression_quantified[t])
    sc.tl.umap(expression_quantified[t])
    expression_quantified[t].write_h5ad(os.path.join("../perturbations/saunders", f"{t}.h5ad"))

expression_quantified = dict()
for t in ("train", "test"):
    print(f"Plotting {t}")
    expression_quantified[t] = sc.read_h5ad(os.path.join("../perturbations/saunders", f"{t}.h5ad"))
    vars_to_show = ["embryo", "timepoint", "perturbation", "cell_type", "cell_type_broad", "cell_type_sub", "cell_count", 'total_counts', 'log1p_total_counts']
    for v in vars_to_show:
        fig = sc.pl.umap(expression_quantified[t], color = v, show = False)
        try:
            os.makedirs(f"../perturbations/saunders/{t}", exist_ok=True)
            fig.figure.savefig(f"../perturbations/saunders/{t}/{v}.pdf")
        except Exception as e:
            print(f"Plots failed with error {repr(e)}")

# Study effect of perturbation differently than above (due to it being very shallow data)
all_comparisons = dict()
for perturbation in expression_quantified["test"].uns["perturbed_and_measured_genes"]:
    wobblybobs = expression_quantified["test"].obs.copy()[["cell_type", "perturbation"]]
    wobblybobs["expression"] = expression_quantified["test"][:, perturbation].X.toarray()
    wobblybobs = wobblybobs.groupby(["cell_type", "perturbation"]).agg("mean").reset_index()
    all_comparisons[perturbation] = wobblybobs.query(f'perturbation=="{perturbation}"').merge(
        wobblybobs.copy().query('perturbation == "control"')[["expression", "cell_type"]], 
        left_on = "cell_type",
        right_on = "cell_type", 
        suffixes = ("KO", "control")
    )
wobblybobs = pd.concat(all_comparisons).reset_index()
wobblybobs["difference"] = wobblybobs["expressionKO"] - wobblybobs["expressioncontrol"] 
alt.data_transformers.disable_max_rows()
KO_effect = alt.Chart(data=wobblybobs).mark_circle(size = 300).encode(
    x = "expressioncontrol",
    y = "difference", 
    color = alt.Color("cell_type", scale = alt.Scale(scheme='category20')),
).properties(
    width=100,
    height=100
).facet(
    facet = "perturbation",
    columns = 4
).resolve_scale(
    x='independent',
    y='independent'
)
KO_effect.save("../perturbations/saunders/test/KO_transcript_level.svg")
