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
import ingestion
import matplotlib.pyplot as plt

# Access our code
import pereggrn_perturbations
import pereggrn_networks
import ggrn_backend3.api as ggrn_autoregressive

# Access our data collections
pereggrn_networks.set_grn_location(
    '../../network_collection/networks'
)
pereggrn_perturbations.set_data_path(
    '../../perturbation_data/perturbations'
)
DEFAULT_HUMAN_TFs = pd.read_csv("../../accessory_data/humanTFs.csv")
DEFAULT_HUMAN_TFs = DEFAULT_HUMAN_TFs.loc[DEFAULT_HUMAN_TFs["Is TF?"]=="Yes", "HGNC symbol"]

# Would our benchmarks work if we had everything right?
#
# - steady state assumption is right, or matched controls are available
# - no noise
# - large effects
# - correct network structure 
# - all factors perturbed in either train or test data

for true_network in [
        "MARA_FANTOM4"          ,
        "gtex_rna"              ,
        "celloracle_human"      ,
        "cellnet_human_Hg1332"  ,
        "cellnet_human_Hugene"  ,
    ]:
    # If we generate the data via a network structure available to GGRN, can we beat the mean + median in the basic experiments?
    network_edges = pereggrn_networks.pereggrn_grn_all_subnetworks(true_network)
    network_edges = pereggrn_networks.pivotNetworkLongToWide(network_edges)
    network_edges.index = network_edges["gene_short_name"]

    # Make it square, symmetric, and TFs-only.
    # # TFs-only is much faster; the code scales poorly with dimension because we perturb all genes.
    network_edges = network_edges[network_edges["gene_short_name"].isin(DEFAULT_HUMAN_TFs)]
    network_edges = network_edges.loc[network_edges.index.isin(network_edges.columns), :]
    network_edges = network_edges.loc[:, network_edges.columns.isin(network_edges.index)]
    network_edges = network_edges.loc[network_edges.index, network_edges.index]
    gene_names = network_edges.index
    network_edges = np.array(network_edges)>0
    effect_sizes = 1.0 
    F = network_edges*effect_sizes

    # To make the steady-state exist under all perts, set the max eigenvalue plus effect size to <<1.
    eigenstuff = np.linalg.eig(F)
    max_index = np.argmax(np.abs(eigenstuff[0]))
    max_eigenvalue = np.abs(eigenstuff[0][2]) 
    F = 0.5*F / np.max([1, max_eigenvalue])

    # Create both steady-state and single-step data
    for noise_sd in [0, 0.001]:
        for num_steps in [1, 10, 50, 100]:
            finalDataFileFolder = f"../perturbations/simulation_TrueNetwork={true_network}_S={num_steps}_NoiseSD={noise_sd}"
            print(finalDataFileFolder)
            expression_quantified, R,G,Q,F, latent_dimension = ggrn_autoregressive.simulate_autoregressive(
                F = F, 
                num_steps = num_steps, 
                expression_level_after_perturbation = 1, 
                initial_state = np.random.random(F.shape),
                matched_control_is_integer = False, 
                residual_connections= False,
            )
            expression_quantified.X = expression_quantified.X + noise_sd*np.random.standard_normal(expression_quantified.X.shape)
            # When checking these data, the benchmarking and ggrn framework will typically assume it's on 
            # the scale of logged counts per million gene expression data, so certainly no values above 15. We need to skip this check.
            expression_quantified.uns["skip_log_check"] = True

            # Fix metadata
            expression_quantified.obs.loc[expression_quantified.obs["perturbation"]=="control",  "is_control"] = True
            expression_quantified.var.index = expression_quantified.var["gene_name"] = gene_names
            not_control = ~expression_quantified.obs["is_control"]
            expression_quantified.obs.loc[not_control, "perturbation"] = [
                gene_names[int(g)] 
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
            print(f"max expression: {expression_quantified.X.max()}")
            print(f"mean expression: {expression_quantified.X.mean()}")
            # Save results
            os.makedirs(finalDataFileFolder, exist_ok = True)
            expression_quantified.write_h5ad(os.path.join(finalDataFileFolder, "test.h5ad"))