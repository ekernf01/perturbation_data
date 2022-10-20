from typing import Iterable
import numpy as np 
import pandas as pd 
import h5py
import scanpy as sc
import anndata
import typing 

def simplify_categorical(x:pd.DataFrame, column: str, max_categories: int = 20, filler: str = "other", new_column: str = None):
    """Mark less frequent categories as other. Accepts and returns a dataframe."""
    assert type(x) is pd.DataFrame
    assert column in x.columns
    if new_column is None:
        new_column = column + "_simplified"
    counts = pd.DataFrame(x[column].value_counts())
    counts[new_column] = filler 
    counts.iloc[0:max_categories, 1] = counts.index[0:max_categories] 
    simplified = pd.merge(x, counts, how = "left", left_on = column, right_index = True)
    return simplified

def convert_ens_to_symbol(ensembl_ids, gtf, strip_version = False):
    """Convert ensembl gene id's (incomprehensible) into Entrez gene symbols (e.g. GAPDH)
    
    Args:
    
    - gtf: path to a GTF file with transcript annotations, e.g. Gencode V35.
    - ensemble_ids: iterable with inputs.
    - strip_version: ensembl ID's can be like 'ENSG01234.5' or like 'ENSG01234'. The '.5' is the version, i.e. the number of times this id has changed. Sometimes you want to strip this off (strip_version = True). More on ensembl ID's:
        
        https://useast.ensembl.org/Help/Faq?id=488#:~:text=An%20Ensembl%20stable%20ID%20consists,(version).&text=The%20second%20part%20is%20a,(object%20type)(identifier).
    
    """
    gene_identifiers = pd.read_csv(gtf, 
                                   sep = "\t", 
                                   comment = "#", 
                                   header=None)
    gene_identifiers = gene_identifiers.loc[:, [8]]
    gene_identifiers = gene_identifiers.drop_duplicates()
    gene_identifiers["ensembl_id"] = [[g for g in s.split(";") if "gene_id " in g][0] for s in gene_identifiers[8]]
    gene_identifiers["ensembl_id"] = gene_identifiers["ensembl_id"].str.replace('"|gene_id| ', '')
    if strip_version:
        gene_identifiers["ensembl_id"] = [i.split(".")[0] for i in gene_identifiers["ensembl_id"]]
    gene_identifiers["symbol"] = [[g for g in s.split(";") if "gene_name" in g] for s in gene_identifiers[8]]
    gene_identifiers["symbol"] = [l[0] if len(l)>0 else None for l in gene_identifiers["symbol"]]
    gene_identifiers["symbol"] = gene_identifiers["symbol"].str.replace('"|gene_name| ', '')
    gene_identifiers = gene_identifiers[["ensembl_id", "symbol"]]
    gene_identifiers.drop_duplicates(inplace = True)
    gene_identifiers.set_index("ensembl_id", inplace = True)
    gene_identifiers.drop_duplicates(inplace = True) #same symbol, different ens id
    
    return [gene_identifiers.loc[eg, "symbol"] 
            if eg in gene_identifiers.index and gene_identifiers.loc[eg, "symbol"] is not None else eg 
            for eg in ensembl_ids]


def read_cmap(expression_file, gene_metadata, instance_metadata):
    """
    Read a dataset in CMAP's HDF-based gctx format, returning an AnnData object.
    """
    # expr
    hf = h5py.File(expression_file, 'r')
    expression_quantified = np.array(hf.get('/0/DATA/0/matrix'))
    gene_id = pd.DataFrame(np.array(hf.get(name = "/0/META/ROW")["id"]), columns = ["pr_gene_id"]).astype(int)
    sample_id = pd.DataFrame(np.array(hf.get(name = "/0/META/COL")["id"]), columns = ["sample_id"])
    sample_id["sample_id"] = sample_id["sample_id"].str.decode("utf8")
    expression_quantified = sc.AnnData(expression_quantified, 
                                 var = gene_id.copy(),
                                 obs = sample_id.copy())
    # sample meta
    sample_metadata = pd.read_csv(instance_metadata, sep = "\t")
    expression_quantified.obs = pd.merge(
        expression_quantified.obs, 
        sample_metadata,
        how = "left", 
        on = "sample_id",
    )
    expression_quantified.obs_names = expression_quantified.obs["sample_id"]
    # gene meta
    more_gene_metadata = pd.read_csv(gene_metadata, sep = "\t")
    more_gene_metadata["pr_gene_id"] = more_gene_metadata["pr_gene_id"].astype(int)
    expression_quantified.var = pd.merge(
        expression_quantified.var, 
        more_gene_metadata,
        how = "left", 
        on = "pr_gene_id",
    )
    expression_quantified.var_names = expression_quantified.var["pr_gene_symbol"]
    return expression_quantified

def describe_perturbation_effect(adata: anndata.AnnData, perturbation_type) -> anndata.AnnData:
    """ Add details on perturbation effect on targeted genes

    Args:
        adata (anndata.AnnData): A perturbation dataset
        perturbation_type (typing.Union): one of {"overexpression", "knockout", "knockdown"}, or if mixed, an iterable of length equal to n_samples.

    Raises:
        ValueError: Triggered by invalid perturbation types.

    Returns:
        anndata.AnnData: adata with columns filled in for 'expression_level_after_perturbation' and 'perturbation_type'
    """
    multiple_genes_hit = "perturbations_overlap" in adata.uns.keys() and adata.uns["perturbations_overlap"]
    adata.obs["expression_level_after_perturbation"] = np.nan
    adata.obs["perturbation_type"] = perturbation_type
    def do_one_gene(p, g):
        if g in adata.uns["perturbed_and_measured_genes"]:
            if adata.obs.loc[p, "perturbation_type"] in {"overexpression", "knockdown"}:
                return adata[p, g].X[0,0]
            elif adata.obs.loc[p, "perturbation_type"] in {"knockout"}:
                return 0
            else:
                raise ValueError("perturbation_type must be in 'overexpression', 'knockdown', 'knockout'")
        else:
            return np.nan

    for p in adata.obs.query("~is_control").index:
        gene_or_genes = adata.obs.loc[  p, "perturbation"]
        if multiple_genes_hit:
            elap = []
            for g in gene_or_genes.split(","):
                elap.append(do_one_gene(p, g))
        else:
            elap = do_one_gene(p, gene_or_genes)
        adata.obs.loc[ p, "expression_level_after_perturbation"] = elap

    return adata