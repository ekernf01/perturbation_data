#!/usr/bin/env python
# coding: utf-8

# ### Simulated data
#
# This notebook generates data from an autoregressive linear model under 
# various assumptions about effect sizes, noise level, sample size, network structure, and time-scale.

import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
import pandas as pd
import os
import altair as alt
import shutil 
import ingestion

# Assume we run from Eric's whole-project-all-repos directory or location of this script.
try:
    os.chdir("perturbation_data/setup")
except FileNotFoundError:
    pass

# Access our code
import load_perturbations
import load_networks
import ggrn_backend3.api as ggrn_autoregressive

# Access our data collections
load_networks.set_grn_location(
    '../network_collection/networks'
)
load_perturbations.set_data_path(
    '../perturbation_data/perturbations'
)
DEFAULT_HUMAN_TFs = pd.read_csv("../accessory_data/humanTFs.csv")
DEFAULT_HUMAN_TFs = DEFAULT_HUMAN_TFs.loc[DEFAULT_HUMAN_TFs["Is TF?"]=="Yes", "HGNC symbol"]

# Would this work if we had everything right?
#
# - steady state assumption is right
# - no noise
# - large effects
# - correct network structure 
# - a couple hundred perturbations

# If we use the gTEX network structure to generate the data, can we tell that it's the best network?
# This can replace experiment 1.??.??
expression_quantified, R,G,Q,F, latent_dimension = ggrn_autoregressive.simulate_autoregressive(F = , )
expression_quantified.var.index = expression_quantified.var["gene_name"]
expression_quantified.obs["perturbation"]   = 
expression_quantified.obs["is_control"]     = expression_quantified.obs["perturbation"] == "ctrl"
expression_quantified.obs["is_control_int"] = expression_quantified.obs["is_control"].astype("int")
expression_quantified.obs["spearmanCorr"] = 0
expression_quantified.uns["perturbations_overlap"] = False
expression_quantified = ingestion.describe_perturbation_effect(
    expression_quantified, 
    perturbation_type="knockout", 
    multiple_genes_hit=False,
)
sc.pp.highly_variable_genes(expression_quantified, flavor="seurat_v3", n_top_genes=expression_quantified.var.shape[0])
perturbed_genes = set(list(expression_quantified.obs['perturbation'].unique())).difference({"ctrl"})
perturbed_and_measured_genes = perturbed_genes.intersection(expression_quantified.var.index)
perturbed_but_not_measured_genes = perturbed_genes.difference(expression_quantified.var.index)
print("These genes were perturbed and measured:")
print(perturbed_and_measured_genes)
print("These genes were perturbed but not measured:")
print(perturbed_but_not_measured_genes)
expression_quantified.uns["perturbed_and_measured_genes"] = list(perturbed_and_measured_genes)
expression_quantified.uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
expression_quantified.raw = expression_quantified.copy()
finalDataFileFolder = "../simulation1"
os.makedirs(finalDataFileFolder, exist_ok = True)
expression_quantified.write_h5ad(os.path.join(finalDataFileFolder, "test.h5ad"))