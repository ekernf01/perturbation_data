# This script makes a small and entirely unnatural dataset suitable for
# unit-testing of DCD-FG with results expected to be exactly correct.
import os
import pandas as pd
import numpy as np
import scipy
import torch
import anndata
import networkx as nx

# Our modules
import sys
sys.path.append("../setup")
import ingestion 
import ggrn.api as ggrn

# In case I run this from a commonly used but wrong working directory
try:
    os.chdir("perturbation_data/setup")
except:
    pass

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%6.4f" % x))

# Simple simulation code
def forward(x, n):
    global adjMat, weight
    x = np.matmul(x, adjMat * weight) + n
    return np.squeeze(np.asarray(x))

# Unlike the prediction methods we added to DCD-FG, this one iterates all the way to steady state. 
def simulateKO(control_expression: np.ndarray, 
               noise: np.ndarray, 
               KO_gene_idx: int=0, 
               KO_gene_value: float = 0, 
               maxiter: int=100):
    x = control_expression
    for i in range(100):
        xold = x.copy()
        x[KO_gene_idx] = KO_gene_value
        x = forward(x, noise) 
        if np.linalg.norm(xold - x) < 1e-12:
            break
    x[KO_gene_idx] = KO_gene_value
    return x

# Create a DAG
G = nx.DiGraph()
G.add_edges_from([(0,1), (1,2), 
                  (2,3), (3,4)])
print(f"G is a DAG? {nx.is_directed_acyclic_graph(G)}")

adjMat = nx.adjacency_matrix(G, range(len(G.nodes))).todense()
adjMat = torch.from_numpy(adjMat)
print("Adjacency Matrix")
print(adjMat.numpy())

weight = adjMat
print("Weight")
print(weight.numpy())

numInst    = 3000
trainShape = (numInst, adjMat.shape[0])
cntrl      = np.zeros(trainShape)

# Create a baseline control expression
source     = np.array([5, 0, 0, 0, 0])
print("\nBias expression level at SS (run simulations to completion 3 times):")
bias = np.ones(5) * 5
bias = simulateKO(bias, source, KO_gene_idx=0, KO_gene_value=source[0])
bias = simulateKO(bias, source, KO_gene_idx=0, KO_gene_value=source[0])
bias = simulateKO(bias, source, KO_gene_idx=0, KO_gene_value=source[0])

# Perturb each gene separately
cntrl += bias
fold=5
for f in range(fold):
    for i in range(numInst//fold*f, numInst//fold*(f+1)):
        cntrl[i] = simulateKO(cntrl[i], source, KO_gene_idx=f, KO_gene_value=np.linspace(1e-2, bias[f], numInst//fold+1)[i-numInst//fold*f])
        
# Add a control expression
cntrl = np.vstack([cntrl, bias])

# Create an anndata and add all the required features
adata = anndata.AnnData(X=cntrl, 
                        dtype=np.float64,
                        obs=pd.DataFrame([f"Sample{i}" for i in range(cntrl.shape[0])], columns=["sample"]),
                        var=pd.DataFrame([f"Gene{i}"   for i in range(cntrl.shape[1])], columns=["gene"]))
adata.obs["perturbation"] = (["Gene0"] * (numInst//fold) + 
                             ["Gene1"] * (numInst//fold) + 
                             ["Gene2"] * (numInst//fold) + 
                             ["Gene3"] * (numInst//fold) +
                             ["Gene4"] * (numInst//fold) +
                             ["Control"])
adata.obs["is_control"]   = adata.obs["perturbation"] == "Control"
adata.obs["is_control_int"]   = [1 if i else 0 for i in adata.obs["is_control"]]
adata.var.index = adata.var.gene
adata.var.index.name = "geneName"
adata.obs.index = [str(i) for i in range(numInst+1)]
adata.obs["consistentW/Perturbation"] = True
adata.obs["logFC"] = -999
adata.obs["spearmanCorr"] = -999
adata.raw = adata
adata.X = adata.raw.X.copy()
adata.X = scipy.sparse.csr_matrix(adata.X)
perturbed_and_measured_genes = adata.var.index
perturbed_but_not_measured_genes = list()
print("These genes were perturbed but not measured:")
print(perturbed_but_not_measured_genes)
adata.uns["perturbed_and_measured_genes"] = list(set(adata[~adata.obs.is_control].obs.perturbation))
adata.uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
adata = ingestion.describe_perturbation_effect(adata, "knockdown", multiple_genes_hit = True)
adata.uns["weight"] = weight.numpy()
adata.uns["weight_mask"] = adjMat.numpy()
adata.uns["bias"]   = bias
adata.var["highly_variable_rank"] = [1,2,3,4,5]
os.makedirs("../perturbations/dcdfg_test", exist_ok=True)
adata.write_h5ad("../perturbations/dcdfg_test/test.h5ad")