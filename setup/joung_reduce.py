import scanpy as sc
import numpy as np
import pandas as pd
import anndata
from scipy.sparse import csr_matrix
# local
import importlib
import sys
sys.path.append("setup")
import ingestion
import global_effects
importlib.reload(ingestion)
importlib.reload(global_effects)

# We ran this on the file from:
# https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217460/suppl/GSE217460%5F210322%5FTFAtlas.h5ad.gz
# Beware it is 160GB when unzipped.
expression_quantified = sc.read_h5ad("../not_ready/joung/GSE217460_210322_TFAtlas.h5ad",    backed = "r")

orfs = expression_quantified.obs["TF"].unique()
aggs = []
for i,orf in enumerate(orfs):
    try:
        agg = sc.read_h5ad(f"../not_ready/{orf}.h5ad")
    except:
        chunk = expression_quantified[expression_quantified.obs["TF"]==orf, :].to_memory()
        chunk.raw = anndata.AnnData(X = np.exp(chunk.X)-1, var = chunk.var, obs = chunk.obs)
        for i,cell in enumerate(chunk.obs_names):
            chunk.raw.X[i, :] = chunk.raw[cell, :].X*chunk.obs["n_counts"][cell]/1e4
        S_genes_hum = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", 
                    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "CENPU", "HELLS", "RFC2", 
                    "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", 
                    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", 
                    "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
        G2M_genes_hum = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
                    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "PIMREG", 
                    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", 
                    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "JPT1", "CDC20", "TTK",
                    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", 
                    "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", 
                    "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]
        sc.tl.score_genes_cell_cycle(chunk, s_genes=S_genes_hum, g2m_genes=G2M_genes_hum)
        chunk.obs["is_control"] = False
        agg = ingestion.aggregate_by_perturbation(chunk, group_by = ["TF", "batch", "phase"])
        agg.write_h5ad(f"../not_ready/{orf}.h5ad")    
    aggs.append(agg)

finished = anndata.AnnData(
    X   = np.concat([agg.X for agg in aggs]), 
    obs = pd.concat([agg.obs for agg in aggs]), 
    var = pd.concat([agg.var for agg in aggs]), 
)
finished.write_h5ad("../not_ready/GSE217460_210322_TFAtlas_aggregated.h5ad")