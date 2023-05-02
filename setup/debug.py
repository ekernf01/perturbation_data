import anndata
import pandas as pd
import numpy as np
import os
import scanpy as sc
import ingestion
TFs = pd.read_csv("../../accessory_data/humanTFs.csv")["HGNC symbol"]
num_genes = 50
num_ctrl = 100
num_perturb = 400
assert num_genes < num_perturb, "This is assumed to have p << n. Otherwise, modify the 'perturbation' column construction."
assert round(num_perturb/num_genes)==num_perturb/num_genes, "This is assumed to have round n/p. Otherwise, modify the 'perturbation' column construction."
expression_quantified = anndata.AnnData(
    X = np.random.random((num_perturb + num_ctrl,num_genes)), 
    obs = pd.DataFrame({
        "index": [f"o{i}" for i in range(num_perturb + num_ctrl)], 
        "cell_type":"all",
        "is_control":     [True   for _ in range(num_ctrl)] + [False  for _ in range(num_perturb)],
        "is_control_int": [1      for _ in range(num_ctrl)] + [0      for _ in range(num_perturb)],
        "perturbation":   ["ctrl" for _ in range(num_ctrl)] + list(np.repeat(TFs[0:num_genes], num_perturb/num_genes)),
        "spearmanCorr": 0,
    }),
    var = pd.DataFrame(
        index = TFs[0:num_genes], 
    )
)
expression_quantified.raw = expression_quantified.copy()
controls = ["ctrl"]
perturbed_genes = set.union(*[set(p.split(",")) for p in expression_quantified.obs["perturbation"]]).difference(set(controls))
perturbed_and_measured_genes = perturbed_genes.intersection(expression_quantified.var.index)
perturbed_but_not_measured_genes = perturbed_genes.difference(expression_quantified.var.index)
expression_quantified.uns["perturbed_and_measured_genes"]     = list(perturbed_and_measured_genes)
expression_quantified.uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
expression_quantified.uns["perturbations_overlap"] = False

expression_quantified = ingestion.describe_perturbation_effect(expression_quantified, perturbation_type="knockout")
sc.pp.highly_variable_genes(expression_quantified, flavor = "seurat_v3", n_top_genes=expression_quantified.shape[1])

os.makedirs("../perturbations/debug", exist_ok = True)
expression_quantified.write_h5ad(os.path.join("../perturbations/debug", f"test.h5ad"))
