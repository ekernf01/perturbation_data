#!/usr/bin/env python
# coding: utf-8

# ### Early Perturb-seq example: UPR in K562 cells
# 
# This notebook prepares an early Perturb-seq example from Adamson et. al. The number of perturbations
# is rather small compared to some more recently available work (82 distinct genes), but GEARS gets 
# favorable results on these data. Because of that, we are motivated to test various methods. 
# We use the GEARS preprocessing, which is to select 5k variable genes and log1p-transform.
# 
import warnings
warnings.filterwarnings('ignore')
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata 
import os
import altair as alt
from collections import Counter
import shutil 

finalDataFileFolder = "../perturbations/adamson"
os.makedirs(finalDataFileFolder, exist_ok = True)
sc.settings.figdir = finalDataFileFolder

from gears import PertData, GEARS
pert_data = PertData('./data')
pert_data.load(data_name = 'adamson')
expression = pert_data.adata
expression.obs["perturbation"]   = expression.obs["condition"].str.replace("+ctrl", "")
expression.obs["is_control"]     = expression.obs["perturbation"] == "ctrl"
expression.obs["is_control_int"] = expression.obs["is_control"].astype("int")
expression.write_h5ad("../perturbations/adamson/test.h5ad")
shutil.rmtree("./data")
