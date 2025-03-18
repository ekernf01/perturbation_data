#!/usr/bin/env python
# coding: utf-8

# We downsample the definitive endoderm data for fast software tests.
import scanpy as sc
import numpy as np
import os
os.makedirs("perturbations/definitive_endoderm_downsampled", exist_ok=True)
for t in ["train", "test"]:
    ad = sc.read_h5ad(f"perturbations/definitive_endoderm/{t}.h5ad")
    ad[np.random.choice(ad.obs.index, 500, replace=False), :].write_h5ad(os.path.join(f"perturbations/definitive_endoderm_downsampled/{t}.h5ad"))