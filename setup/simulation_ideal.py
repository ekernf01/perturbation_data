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
# - all factors perturbed in either train or test data

for true_network in [
        "celloracle_human"      ,
        "gtex_rna"            ,
        "MARA_FANTOM4"          ,
        "cellnet_human_Hg1332"  ,
        "cellnet_human_Hugene"  ,
    ]:
    # If we generate the data via a network structure available to GGRN, can we beat the mean + median in the basic experiments?
    network_edges = load_networks.load_grn_all_subnetworks(true_network)
    network_edges = load_networks.pivotNetworkLongToWide(network_edges)
    network_edges.index = network_edges["gene_short_name"]

    # Make it square and TFs-only (much faster; code scales poorly with dimension because we perturb all genes)
    network_edges = network_edges[network_edges["gene_short_name"].isin(DEFAULT_HUMAN_TFs)]
    network_edges = network_edges.loc[network_edges.index.isin(network_edges.columns), :]
    network_edges = network_edges.loc[:, network_edges.columns.isin(network_edges.index)]
    network_edges = network_edges.loc[network_edges.index, network_edges.index]
    # To make the steady-state exist under all perts, set the max eigenvalue plus effect size to <<1.
    effect_size = 0.1
    F = np.array(network_edges)*effect_size
    eigenstuff = np.linalg.eig(F)
    max_index = np.argmax(np.abs(eigenstuff[0]))
    F = 0.01*F / np.abs(eigenstuff[0][2])
    X0 = 0*eigenstuff[1][:, 2] # Initialize to 0 or leading eigenvector
    X0 = np.array([X0,X0])

    # Create both steady-state and single-step data
    for noise_sd in [0, 0.001]:
        for num_steps in [1, 10]:
            finalDataFileFolder = f"../perturbations/simulation_TrueNetwork={true_network}_S={num_steps}_NoiseSD={noise_sd}"
            print(finalDataFileFolder)
            expression_quantified, R,G,Q,F, latent_dimension = ggrn_autoregressive.simulate_autoregressive(
                F = F, 
                num_steps = num_steps, 
                initial_state=X0, 
                expression_level_after_perturbation = 1
            )
            # Censor initial state
            expression_quantified = expression_quantified[expression_quantified.obs["time"]>0,:] 
            expression_quantified.X = expression_quantified.X + noise_sd*np.random.standard_normal(expression_quantified.X.shape)
            # When checking these data, the benchmarking and ggrn framework will typically assume it's on 
            # the scale of logged gene expression data, so certainly no values above 15. We need to skip this check.
            expression_quantified.uns["skip_log_check"] = True

            # Fix metadata
            expression_quantified.obs.loc[expression_quantified.obs["perturbation"]=="control",  "is_control"] = True
            expression_quantified.var.index = expression_quantified.var["gene_name"] = network_edges.index
            not_control = ~expression_quantified.obs["is_control"]
            expression_quantified.obs.loc[not_control, "perturbation"] = [
                network_edges.index[int(g)] 
                for g in expression_quantified.obs.loc[not_control, "perturbation"]
            ]

            expression_quantified.obs["is_control_int"] = expression_quantified.obs["is_control"].astype("int")
            expression_quantified.uns["perturbations_overlap"] = False
            expression_quantified.var["highly_variable_rank"] = range(expression_quantified.n_vars)
            perturbed_genes = set(list(expression_quantified.obs['perturbation'].unique())).difference({"control"})
            perturbed_and_measured_genes     = perturbed_genes.intersection(expression_quantified.var.index)
            perturbed_but_not_measured_genes = perturbed_genes.  difference(expression_quantified.var.index)
            expression_quantified.uns["perturbed_and_measured_genes"]     = list(perturbed_and_measured_genes)
            expression_quantified.uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
            expression_quantified.raw = expression_quantified.copy()
            expression_quantified = ingestion.describe_perturbation_effect(
                expression_quantified, 
                perturbation_type="overexpression", 
                multiple_genes_hit=False,
            )
            # Save results
            os.makedirs(finalDataFileFolder, exist_ok = True)
            expression_quantified.write_h5ad(os.path.join(finalDataFileFolder, "test.h5ad"))