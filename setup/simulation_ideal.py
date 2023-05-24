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
import numpy as np
import os
import altair as alt
import shutil 
import ingestion

# Access our code
import load_perturbations
import load_networks
import ggrn_backend3.api as ggrn_autoregressive

# Access our data collections
load_networks.set_grn_location(
    '../../network_collection/networks'
)
load_perturbations.set_data_path(
    '../../perturbation_data/perturbations'
)
DEFAULT_HUMAN_TFs = pd.read_csv("../../accessory_data/humanTFs.csv")
DEFAULT_HUMAN_TFs = DEFAULT_HUMAN_TFs.loc[DEFAULT_HUMAN_TFs["Is TF?"]=="Yes", "HGNC symbol"]

# Would this work if we had everything right?
#
# - steady state assumption is right
# - no noise
# - large effects
# - correct network structure 
# - a couple hundred perturbations

# If we use the CellOracle network structure to generate the data, can we beat the mean + median in the basic experiments?
celloracle_network = load_networks.load_grn_all_subnetworks("celloracle_human")
celloracle_network = load_networks.pivotNetworkLongToWide(celloracle_network)
celloracle_network.index = celloracle_network["gene_short_name"]
# Make it square
celloracle_network = celloracle_network.loc[celloracle_network.index.isin(celloracle_network.columns), :]
celloracle_network = celloracle_network.loc[:, celloracle_network.columns.isin(celloracle_network.index)]
celloracle_network = celloracle_network.loc[celloracle_network.index, celloracle_network.index]
# To make the steady-state exist and dominate the data, set the max eigenvalue to 1 and initialize to leading eigenvector.
effect_size = 0.1
F = np.array(celloracle_network)*effect_size
eigenstuff = np.linalg.eig(F)
max_index = np.argmax(eigenstuff[0])
F = F / np.abs(eigenstuff[0][2])
X0 = eigenstuff[1][:, 2]
assert all(1==np.round(F.dot(X0) / X0)), "Dominant eigenpair for F should be 1, X0."
X0 = np.array([X0,X0,X0,X0])

# Create both steady-state and single-step data
for num_steps in [1, 100]:
    expression_quantified, R,G,Q,F, latent_dimension = ggrn_autoregressive.simulate_autoregressive(F = F, num_steps = num_steps, initial_state=X0)

    # When checking these data, the benchmarking and ggrn framework will typically assume it's on 
    # the scale of logged gene expression data, so certainly no values above 15.
    expression_quantified.uns["skip_log_check"] = True

    # Fix metadata
    expression_quantified.obs.loc[expression_quantified.obs["perturbation"]=="control",  "is_control"] = True
    expression_quantified.var.index = expression_quantified.var["gene_name"] = celloracle_network.index
    not_control = ~expression_quantified.obs["is_control"]
    expression_quantified.obs.loc[not_control, "perturbation"] = [
        celloracle_network.index[int(g)] 
        for g in expression_quantified.obs.loc[not_control, "perturbation"]
    ]

    expression_quantified.obs["is_control_int"] = expression_quantified.obs["is_control"].astype("int")
    expression_quantified.uns["perturbations_overlap"] = False
    expression_quantified.var["highly_variable_rank"] = range(expression_quantified.n_vars)
    perturbed_genes = set(list(expression_quantified.obs['perturbation'].unique())).difference({"control"})
    perturbed_and_measured_genes     = perturbed_genes.intersection(expression_quantified.var.index)
    perturbed_but_not_measured_genes = perturbed_genes.  difference(expression_quantified.var.index)
    print("These genes were perturbed and measured:")
    print(perturbed_and_measured_genes)
    print("These genes were perturbed but not measured:")
    print(perturbed_but_not_measured_genes)
    expression_quantified.uns["perturbed_and_measured_genes"]     = list(perturbed_and_measured_genes)
    expression_quantified.uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
    expression_quantified.raw = expression_quantified.copy()
    expression_quantified = ingestion.describe_perturbation_effect(
        expression_quantified, 
        perturbation_type="overexpression", 
        multiple_genes_hit=False,
    )
    # Save results
    finalDataFileFolder = f"../perturbations/simulation_{num_steps}_steps"
    os.makedirs(finalDataFileFolder, exist_ok = True)
    expression_quantified.write_h5ad(os.path.join(finalDataFileFolder, "test.h5ad"))