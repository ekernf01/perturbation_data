import os
import pandas as pd
import scanpy as sc
import anndata

def load_perturbation_metadata():
    return pd.read_csv(os.path.join(os.environ["PERTURBATION_PATH"], "perturbations.csv"))

def load_perturbation(dataset_name: str, training_data_only: bool = False):
    """Load a perturbation dataset. 

    Args:
        dataset_name (str): Taken from the metadata rownames.
        training_data_only (bool, optional): Not implemented yet. Once implemented, if true, return separate training data with no perturbations (e.g. timecourse). Defaults to False.

    Returns:
        anndata.AnnData: Perturbation data in a uniform format as described by `check_perturbation_dataset` or the README. 
    """
    return sc.read_h5ad(os.path.join(os.environ["PERTURBATION_PATH"], dataset_name, "test.h5ad"))

def check_perturbation_dataset(dataset_name: str = None, ad: anndata.AnnData = None):
    """Enforce expectations on a perturbation dataset.

    Args:
        h5ad_file (str): Path to file containing perturbation data.
        ad (anndata.AnnData): AnnData object containing perturbation data.
    """
    if ad is None and dataset_name is None:
        raise ValueError("Provide exactly one of ad and dataset_name")
    if not ad is None and not dataset_name is None:
        raise ValueError("Provide exactly one of ad and dataset_name")
    if ad is None and dataset_name is not None:
        ad = load_perturbation(dataset_name)
    
    # Names of genes perturbed
    assert "perturbation" in set(ad.obs.columns), "No 'perturbation' column"
    
    # Level of those genes after perturbation
    assert "expression_level_after_perturbation" in set(ad.obs.columns), "No 'expression_level_after_perturbation' column"
    
    # Overexpression / knockout / knockdown
    assert "perturbation_type" in set(ad.obs.columns), "No 'perturbation_type' column"    
    assert all(
        [pt in {"overexpression", "knockout", "knockdown"} 
        for pt in ad.obs["perturbation_type"]]
    ),  "Invalid 'perturbation_type' column"

    # Boolean column is_control with both T and F
    assert "is_control"   in set(ad.obs.columns), "No 'is_control' column"
    assert bool==ad.obs["is_control"].dtype, "non-boolean 'is_control' column"
    assert       ad.obs["is_control"].any(), "no controls found"
    assert not   ad.obs["is_control"].all(), "only controls found"

    # A set `"perturbed_but_not_measured_genes"` in `uns`, containing all genes or pathways that are perturbed but not measured.
    # A set `"perturbed_and_measured_genes"` in `uns`, containing all genes or pathways that are perturbed and also measured.
    assert all( [    g in ad.var_names for g in ad.uns["perturbed_and_measured_genes"]] ),     "perturbed_and_measured_genes"    " not all measured"
    assert all( [not g in ad.var_names for g in ad.uns["perturbed_but_not_measured_genes"]] ), "perturbed_and_not_measured_genes sometimes measured"
    
    # This is more complicated when each perturbation can affect multiple genes.
    has_multiple_genes_hit = "perturbations_overlap" in ad.uns.keys() and ad.uns["perturbations_overlap"]
    if has_multiple_genes_hit:
        all_genes_hit = set.union(*[set(p.split(",")) for p in ad.obs["perturbation"]])      
    else:
        all_genes_hit = set(ad.obs["perturbation"]) 
    assert all( [g     in all_genes_hit for g in ad.uns["perturbed_and_measured_genes"]] ),     "perturbed_and_measured_genes"  " not perturbed"
    assert all( [g     in all_genes_hit for g in ad.uns["perturbed_but_not_measured_genes"]] ), "perturbed_and_not_measured_genes not perturbed"
    
    # Expression in `.X` should be normalized and natural-log-transformed. 
    assert ad.X.max() < 15, "Expression values too big -- did you log them?" #exp(15) is about 3 million -- too big to be a transcript count.
            
    # Raw data should be present in `raw`.
    assert ad.raw is not None, "raw data are missing"

    return True
