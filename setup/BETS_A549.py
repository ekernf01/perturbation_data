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

for k in sample_list.keys():
    z = np.zeros((len(sample_list[k]), probe_to_gene.shape[0]))
    expression_quantified[k] = anndata.AnnData(
        X = z.copy(), 
        layers = {"detection":z.copy()},
        var = probe_to_gene,
        obs = pd.DataFrame(index = sample_list[k]),
    )
    for sample_position,sample_name in enumerate(sample_list[k]):
        x = pd.read_csv(f"../not_ready/fantom4/fantom4_data/{raw_data_folders[k]}/{sample_name}", sep = "\t")
        x = probe_to_gene.merge(x, left_on = "ProbeId", right_on="ProbeID", how = "left")
        assert all( x["Target"].values == expression_quantified[k].var["Target"].values )
        if k=="test":
            assert "Detection" in x.columns[-1]
            assert "Signal" in x.columns[-2]
            expression_quantified[k][sample_name, ].X = x.iloc[:,-2].values
            expression_quantified[k].layers["detection"][sample_position, :] = x.iloc[:,-1].values
        if k=="train":
            assert "Signal" in x.columns[-1]
            expression_quantified[k][sample_position, ].X = x.iloc[:,-1].values

    # Examine mean-variance relationship
    x = expression_quantified[k].X.mean(axis=0)
    y = expression_quantified[k].X.std(axis=0)
    plt.figure()
    fig = sns.scatterplot(x=x,y=y)
    plt.title("Mean-variance relationship")
    fig.figure.savefig(f"../perturbations/fantom4/{k}/mean_var.pdf", bbox_inches='tight')

# Label with timestamps
expression_quantified["train"].obs["timepoint"] = [float(s.split("_")[1].replace("h.txt", ""))
                                                   for s in expression_quantified["train"].obs.index]
expression_quantified["test"].obs["timepoint"] = 48 # See supp info section "perturbation experiments" pages 13-14

# Label with celltypes
expression_quantified["train"].obs["cell_type"] = "THP-1"
expression_quantified["test"].obs["cell_type"] = "THP-1"

# Label lot/replicate in case of batch effects
expression_quantified["train"].obs["sample"] = [s.split("_")[0] for s in expression_quantified["train"].obs.index]
expression_quantified["train"].obs["lot"] = "unknown"
expression_quantified["test"].obs["sample"] = [s.split("_")[1] for s in expression_quantified["test"].obs.index]
expression_quantified["test"].obs["lot"] = [s.split("_")[2].replace(".txt", "") for s in expression_quantified["test"].obs.index]

# Label with perturbed genes
expression_quantified["train"].obs["perturbation"] = "control"
expression_quantified["train"].obs["is_control"] = True
expression_quantified["test"].obs["perturbation"] = (
    [s.split("_")[0]
     .replace("NC", "control")                 # NC means a negative control: see supp tables 7 and 9 in the FANTOM4 supp info.
     .replace("BMI1", "PCGF4")                 # Same gene, alternate name used by this array
     .replace("SREBP1", "SREBF1")              # Same gene, alternate name used by this array
    for s in expression_quantified["test"].obs.index]
)
expression_quantified["test"].obs["is_control"] = [g=="control" for g in expression_quantified["test"].obs["perturbation"]]
for k in ("train", "test"):
    expression_quantified[k].obs["is_control_int"] = [float(x) for x in expression_quantified[k].obs["is_control"]]
    expression_quantified[k].obs_names = [str(s) for s in expression_quantified[k].obs_names] 

# Remove 14 features that are only available in the timeseries data
complete_cases = np.isnan(expression_quantified["test"].X).sum(0) <= 0 
expression_quantified[ "test"] = expression_quantified[ "test"][:,complete_cases]
expression_quantified["train"] = expression_quantified["train"][:,complete_cases]

# Switch to gene-level quantification 
for k in ("train", "test"):
    expression_quantified[k].var.index = [ 
        expression_quantified[k].var.loc[i, "Symbol"] 
        if pd.notnull(expression_quantified[k].var.loc[i, "Symbol"]) 
        else expression_quantified[k].var.loc[i, "Transcript"]
        for i in expression_quantified[k].var.index
    ]
    new_adata = anndata.AnnData(
        X = np.zeros(
                (
                    expression_quantified[k].n_obs, 
                    len(expression_quantified[k].var.index.unique())
                )
            ),
            obs = expression_quantified[k].obs,
            var = pd.DataFrame(
                index = expression_quantified[k].var.index.unique()
            )
        )
    for g in expression_quantified[k].var.index.unique():
        new_adata[:,g].X = expression_quantified[k][:,g].X.sum(axis = 1)
    expression_quantified[k] = new_adata.copy()
    del new_adata

# ### Quantile normalization followed by natural log
# The 10th highest value in sample i is normalized to the mean of the 10th highest values from all samples.
#
# Trying to do this in the simplest way possible, we use 4 steps.
#
# - Estimate the desired order statistics as a sorted vector
# - Make each sample uniform on [0, 1]
# - Make each sample uniform on [0, nvars-1]
# - Use each sample as an index to access the desired order statistics in the right order
# 
desired_order_statistics = expression_quantified["train"][0,:].X.copy()*0
for k in ("train", "test"):
    for sample_position,sample_name in enumerate(sample_list[k]):
        desired_order_statistics = desired_order_statistics + np.sort(expression_quantified[k][sample_name, :].X)
desired_order_statistics = desired_order_statistics / np.sum([ad.n_obs for ad in expression_quantified.values()])    
desired_order_statistics = np.log(np.squeeze(desired_order_statistics))
print("Normalizing")
for k in ("train", "test"):
    print(k)
    expression_quantified[k].raw = expression_quantified[k].copy()
    expression_quantified[k].raw.obs = expression_quantified[k].obs
    expression_quantified[k].X = sklearn.preprocessing.quantile_transform(expression_quantified[k].X, axis = 1)

    expression_quantified[k].X = np.round(expression_quantified[k].X*(expression_quantified[k].n_vars-1))
    for s in expression_quantified[k].obs_names: 
        expression_quantified[k][s, :].X = desired_order_statistics[[int(i) for i in expression_quantified[k][s, :].X.flatten()]]

    # Examine mean-variance relationship
    x = expression_quantified[k].raw.X.std(axis=0)
    y = expression_quantified[k].X.std(axis=0)
    plt.figure()
    fig = sns.scatterplot(x=x,y=y)
    plt.title("Mean-variance relationship after quantile transformation + natural log")
    fig.figure.savefig(f"../perturbations/fantom4/{k}/transformed_mean_var.pdf", bbox_inches='tight')

# Study perturbation effects
print("Checking perturbation effects")
controls = ["control"]
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
                                           perturbationType="knockdown", 
                                           verbose=False)
print(Counter(status))
expression_quantified["test"].obs["consistentW/Perturbation"] = status
expression_quantified["test"].obs["logFC"] = logFC
correlations = ingestion.computeCorrelation(expression_quantified["test"], verbose=True)
expression_quantified["test"].obs["spearmanCorr"] = correlations[0]
expression_quantified["test"].obs[ "pearsonCorr"] = correlations[1]

# Some basic exploration of results
print("Data exploration")
expression_quantified["both"] = anndata.concat(expression_quantified)
for t in ("train", "test", "both"):
    print(f"Exploring {t}")
    sc.pp.highly_variable_genes(expression_quantified[t], flavor = "seurat_v3", n_top_genes=expression_quantified[t].shape[1])
    with warnings.catch_warnings():
        sc.tl.pca(expression_quantified[t], n_comps=10)
    sc.pp.neighbors(expression_quantified[t])
    sc.tl.louvain(expression_quantified[t])    
    sc.tl.umap(expression_quantified[t])

# Regress out suspected nuisance effects
#
# Training data: time-point 0 has mean 0
# Test data: each cluster has mean 0
# 
expression_quantified[f"train_uncorrected"] = expression_quantified["train"].copy()
expression_quantified[ f"test_uncorrected"] = expression_quantified[ "test"].copy()
sc.pp.regress_out(expression_quantified["test"], "louvain")
for g in expression_quantified["train"].var.index:
    mean_t0 = expression_quantified["train"][expression_quantified["train"].obs["timepoint"]==0,g].X.mean()
    v = expression_quantified["train"][:,g].X - mean_t0
    assert all(pd.notnull(v)), g
    expression_quantified["train"][:,g].X = v

for t in ("train", "test"):
    with warnings.catch_warnings():
        sc.tl.pca(expression_quantified[t], n_comps=10)
    sc.pp.neighbors(expression_quantified[t])
    sc.tl.louvain(expression_quantified[t])
    sc.tl.umap(expression_quantified[t])

for t in ("train", "test", "both", "test_uncorrected", "train_uncorrected"):
    print(f"Plotting {t}")
    vars_to_show = ["timepoint", "perturbation", "louvain", "sample", "lot"]
    for v in vars_to_show:
        fig = sc.pl.umap(expression_quantified[t], color = v, show = False, legend_loc='on data')
        try:
            os.makedirs(f"../perturbations/fantom4/{t}", exist_ok=True)
            fig.figure.savefig(f"../perturbations/fantom4/{t}/{v}.pdf", bbox_inches='tight')
        except Exception as e:
            print(f"Plots failed with error {repr(e)}")
    clusters = expression_quantified[t].obs[["louvain", "perturbation"]].value_counts().reset_index()
    clusters = clusters.groupby("louvain")["perturbation"].agg(func = lambda x: " ".join(np.sort(x)))
    clusters.to_csv(f"../perturbations/fantom4/{t}/clusters.csv")
    expression_quantified[t].write_h5ad(os.path.join("../perturbations/fantom4", f"{t}.h5ad"))

load_perturbations.check_perturbation_dataset("fantom4")