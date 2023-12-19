# This script simulates scRNA versions of the Nakatake data.
import scanpy as sc
import os 

# local
import importlib
import sys
sys.path.append("setup")
import ingestion
importlib.reload(ingestion)

nakatake = sc.read_h5ad("perturbations/nakatake/test.h5ad")
os.makedirs("perturbations/nakatake_simulated_scrna", exist_ok=True)
ingestion.simulate_single_cells(nakatake, num_cells=50, counts_per_cell=10000).write_h5ad("perturbations/nakatake_simulated_scrna/test.h5ad")
