#!/usr/bin/env python
# coding: utf-8

# ### Simulated data
#
# This code attempts to run a realistic simulation by using predictions from
# previously fitted models. 

import scanpy as sc
import anndata
import os
ridgecvpredict = sc.read_h5ad("../../perturbation_benchmarking/experiments/1.0_1/outputs/predictions/5.h5ad")
ridgecvfitted = sc.read_h5ad("../../perturbation_benchmarking/experiments/1.0_1/outputs/fitted_values/5.h5ad")
ridgecvpredict.obs["previously_used_for"] = "test"
ridgecvfitted.obs["previously_used_for"] = "train"

expression_quantified = anndata.concat([ridgecvfitted, ridgecvpredict])
expression_quantified.var = ridgecvfitted.var
expression_quantified.obs["perturbation_type"] = "overexpression"
expression_quantified.obs["is_control"] = expression_quantified.obs["perturbation"].isin(["CAG-rtTA35-IH", "Emerald", "Control"])
perturbed_genes = set(list(expression_quantified.obs['perturbation'].unique())).difference({"control"})
perturbed_and_measured_genes = perturbed_genes.intersection(expression_quantified.var.index)
perturbed_but_not_measured_genes = perturbed_genes.difference(expression_quantified.var.index)
print("These genes were perturbed and measured:")
print(perturbed_and_measured_genes)
print("These genes were perturbed but not measured:")
print(perturbed_but_not_measured_genes)
expression_quantified.uns["perturbed_and_measured_genes"] = list(perturbed_and_measured_genes)
expression_quantified.uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
expression_quantified.raw = expression_quantified.copy()
out = "../perturbations/simulation_realistic0"
os.makedirs(out, exist_ok=True)
expression_quantified.write_h5ad(os.path.join(out, "test.h5ad"))

import load_perturbations
load_perturbations.set_data_path('../../perturbation_data/perturbations')
load_perturbations.check_perturbation_dataset("simulation_realistic0")