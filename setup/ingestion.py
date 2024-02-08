import numpy as np 
import pandas as pd 
import h5py
import scanpy as sc
import anndata
import gc
from joblib import Parallel, delayed, cpu_count, dump

# For QC
import os
import itertools as it
import scipy
from scipy.stats import spearmanr, pearsonr, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

def try_toarray(x):
    try:
        x = x.toarray()
    except Exception:
        pass
    return x

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
    gene_identifiers["ensembl_id"] = gene_identifiers["ensembl_id"].str.replace('"|gene_id| ', '', regex = True)
    if strip_version:
        gene_identifiers["ensembl_id"] = [i.split(".")[0] for i in gene_identifiers["ensembl_id"]]
    gene_identifiers["symbol"] = [[g for g in s.split(";") if "gene_name" in g] for s in gene_identifiers[8]]
    gene_identifiers["symbol"] = [l[0] if len(l)>0 else None for l in gene_identifiers["symbol"]]
    gene_identifiers["symbol"] = gene_identifiers["symbol"].str.replace('"|gene_name| ', '', regex = True)
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


def describe_perturbation_effect(adata: anndata.AnnData, perturbation_type, multiple_genes_hit: bool = None) -> anndata.AnnData:
    """ Add details about perturbation's effect on the targeted genes

    Args:
        adata (anndata.AnnData): A perturbation dataset
        perturbation_type (typing.Union): one of {"overexpression", "knockout", "knockdown"}, or if mixed, an iterable of length equal to n_samples.
        multiple_genes_hit: Set to True if there observations with multiple genes perturbed.
    Raises:
        ValueError: Triggered by invalid perturbation types.

    Returns:
        anndata.AnnData: adata with columns filled in for 'expression_level_after_perturbation' and 'perturbation_type'
    """
    if multiple_genes_hit is None:
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
                elap.append(str(do_one_gene(p, g)))
            elap = ",".join(elap)
        else:
            elap = [do_one_gene(p, gene_or_genes)]
        adata.obs.loc[ p, "expression_level_after_perturbation"] = elap

    return adata


def deseq2_size_factors(counts_df):
    """
    Calculate DESeq size factors
    median of ratio to reference sample (geometric mean of all samples)
    
    https://github.com/broadinstitute/pyqtl/blob/master/qtl/norm.py
    References:
     [1] Anders & Huber, 2010
     [2] R functions:
          DESeq::estimateSizeFactorsForMatrix
    """
    idx = np.all(counts_df>0, axis=1)
    tmp_df = np.log(counts_df[idx])
    s = np.exp(np.median(tmp_df.T - np.mean(tmp_df, axis=1), axis=1))
    return s


def deseq2Normalization(counts_df):
    """
    Equivalent to DESeq2:::counts.DESeqDataSet; counts(x, normalized=T)
    """
    return counts_df / deseq2_size_factors(counts_df)



def visualizeLogFC(fc, pval = None, show_plots = False):
    validLogFC = fc[fc != -999]
    rangeMin = np.floor(np.min(validLogFC))
    rangeMax = np.ceil (np.max(validLogFC))
    plt.figure(figsize=(4,2.5))
    if pval is not None:
        pval = pval[pval != -999]
        plt.scatter(validLogFC, -np.log10(pval))
        plt.ylabel("-Log10 p-value")
    else:
        plt.hist(validLogFC, 
                bins=np.linspace(rangeMin, 
                                rangeMax, 
                                int((rangeMax-rangeMin)*3+1)), 
                label="Per Trial")
        plt.ylabel("Count")

    plt.axvline(0, 0, 1, color='red', label="No Change")
    plt.xlabel("Log2 Fold Change (perturbed/control)")
    plt.legend()
    if show_plots:
        plt.show()

def checkConsistency(adata: anndata.AnnData, 
                     perturbationType: str="overexpression", 
                     group: str=None,
                     verbose: bool=False, 
                     do_return_pval = False, 
                     show_plots = False):
    """ Check whether the gene that was perturbed is actually 
    measured to be higher (if overexpressed) or lower (if knocked
    down) or nearly zero (if knocked out).
    If an observation is a control or if the perturbed gene is not measured, 'N/A' is labeled. 
    If a perturbagen's expression is higher or lower than the median control (matching 
    the direction of intended perturbation), it is labeled True. Otherwise, False. 
    
    Args:
        adata (anndata.AnnData): the object to operate on. adata.X is expected to be normalized but not log-transformed. 
            It is expected to be a dense array, not a sparse e.g. scipy CSR. 
        perturbation_type (str): one of {"overexpression", "knockout", "knockdown"}
        group (str, default None): a column in adata.obs to indicate sub-group of
                                   the treatment and the control.
        verbose (bool): show a swarmplot noting the difference between the control
                        and the treatment, if the perturbation direction and expression
                        level are disconcordant.
    """
        
    
    assert perturbationType in ["overexpression", "knockout", "knockdown"]
    
    normX        = try_toarray(adata.X.copy())
    controlIndex = np.where(adata.obs.is_control)[0]
    control      = normX[controlIndex, :]
    logFC        = np.full((adata.n_obs), -999.0)
    pval         = np.full((adata.n_obs), -999.0)
    consistencyStatus = np.full((adata.n_obs), "Yes")
    
    for row, perturbagen in sorted(enumerate(adata.obs["perturbation"]), key=lambda x: x[1]):
        if (adata.obs.is_control[row] or perturbagen not in adata.var.index):
            consistencyStatus[row] = "NA"
            continue
        loc = np.where(adata.var.index == perturbagen)[0]
        # assert loc.shape[0] == 1 # This gene is only measured by one feature.
        
        if group is not None:        # Only compare treatment to within-group controls (to limit variability)
            assert group in adata.obs.columns
            control = normX[adata.obs.is_control & (adata.obs[group] == adata.obs[group][row]), :]
        logFC[row] = np.log2(np.median(try_toarray(normX[row, loc])) / np.median(try_toarray(control[:, loc])))   
        has_same_perturbation = perturbagen == adata.obs["perturbation"]
        pval[row] = ttest_ind(
            np.log2(try_toarray(normX[has_same_perturbation, :][:, loc])).flatten(), 
            np.log2(try_toarray(control[:, loc])).flatten(), 
            equal_var=True,
        ).pvalue
        if perturbationType == "overexpression" and np.median(normX[row, loc]) > np.median(control[:, loc]):
            continue
        if perturbationType == "knockdown"      and np.median(normX[row, loc]) < np.median(control[:, loc]):
            continue
        if perturbationType == "knockout"       and abs(np.median(normX[row, loc])) < 1e-3:
            continue
        consistencyStatus[row] = "No"  
        
        if verbose:
            plt.figure(figsize=(4,1))
            g = sns.swarmplot(control[:, loc].flatten(), orient='h', label="control")
            g.axvline(np.median(try_toarray(normX[row, loc])), 0, 1, color='red', label="treatment", lw=1)
            g.legend(loc='lower right', bbox_to_anchor=(1.45, 0), ncol=1)
            plt.title(f"{perturbagen} {perturbationType}")
            plt.show()

    # The fold changes are computed via the following logic.  
    # NaN -> treatment = 0, control = 0
    # posInf -> treatment > 0, control = 0 
    # negInf -> treatment = 0, control > 0 (knocked out)
    consistencyStatus[np.isnan(logFC)] = "NA"
    logFC[np.isnan(logFC)] = 0
    logFC[np.isposinf(logFC)] = np.nanmedian(logFC[(logFC > 0) & (logFC != -999) & np.isfinite(logFC)])
    logFC[np.isneginf(logFC)] = np.nanmedian(logFC[(logFC < 0) & (logFC != -999) & np.isfinite(logFC)])
    
    if do_return_pval:
        visualizeLogFC(logFC, pval, show_plots = show_plots)
        return consistencyStatus, logFC, pval
    else:
        visualizeLogFC(logFC, show_plots = show_plots)
        return consistencyStatus, logFC



def computeCorrelation(adata: anndata.AnnData, 
                       verbose: bool=False, 
                       group: str=None):
    """
    Compute the correlation between biological replicates on scale of log fold change. For each 
    set of perturbation, the final correlation score is the median of 
    correlation between all pair-wise combinations of perturbation expression
    and control expression. Both Spearman and Pearson correlation are
    computed.

    This assume the existence of "is_control" in adata.obs. 
    """
    
    def computelogFC(t1, t2, c1, c2):
        logFC1, logFC2 = np.log2(r1 / c1), np.log2(r2 / c2)
        validGeneEntry = np.isfinite(logFC1) & np.isfinite(logFC2)
        logFC1 = logFC1[validGeneEntry]
        logFC2 = logFC2[validGeneEntry]
        if len(logFC1)>1:
            return spearmanr(logFC1, logFC2)[0], pearsonr (logFC1, logFC2)[0]
        else:
            return -999,-999
    
    
    normX        = adata.X.copy()
    spearmanList = np.full(adata.n_obs, fill_value=-999, dtype=np.float64)
    pearsonList  = np.full(adata.n_obs, fill_value=-999, dtype=np.float64)
    controlExpr  = normX[adata.obs.is_control, :]
    
    tempc = 0
    for perturbagen in sorted(set(adata[~adata.obs.is_control].obs.perturbation)):
        
        # All perturbation expressions
        replicaRow  = np.where(adata.obs.perturbation == perturbagen)[0]
        if replicaRow.shape[0] == 1:        # skip perturbation w/o replication
            continue
        
        # if verbose:                         # print how many replicas each perturbagen has
        #     print(replicaRow, perturbagen)
            
        temp1, temp2 = list(), list()
        for (row1, row2) in it.combinations(replicaRow, 2):
            r1, r2 = normX[row1, :], normX[row2, :]
            if group:
                assert group in adata.obs.columns
                g1, g2 = adata.obs[group][row1], adata.obs[group][row2]
                c1 = np.median(try_toarray(normX[(adata.obs.is_control) & (adata.obs[group] == g1), :]), axis=0)
                c2 = np.median(try_toarray(normX[(adata.obs.is_control) & (adata.obs[group] == g2), :]), axis=0)
                s, p = computelogFC(r1, r2, c1, c2)
            else:
                c1 = np.median(controlExpr.copy(), axis=0)
                s, p = computelogFC(r1, r2, c1, c1)
            temp1.append(s)
            temp2.append(p)                
        
        # Record the median values
        spearmanList[replicaRow] = np.median(temp1)
        pearsonList [replicaRow] = np.median(temp2)
            
            
    # Everything below this is to produce plots
    if verbose:
        corrVal = np.array([spearmanList, pearsonList]).flatten()[:, np.newaxis]
        corrStr = np.array(["Spearman"] * len(spearmanList) + 
                           ["Pearson" ] * len(pearsonList)).flatten()[:, np.newaxis]
        
        assert "consistentW/Perturbation" in adata.obs
        assert "logFC" in adata.obs

        status  = np.tile(adata.obs['consistentW/Perturbation'], 2)[:, np.newaxis]
        logFC   = np.tile(adata.obs['logFC']                   , 2)[:, np.newaxis]
        corrDF  = pd.DataFrame(np.hstack([corrVal, corrStr, logFC, status]), columns=["Value", "Name", "logFC", "ConsistentW/Perturbation"])
        corrDF[['Value']] = corrDF[['Value']].apply(pd.to_numeric)
        corrDF[['logFC']] = corrDF[['logFC']].apply(pd.to_numeric)
        delCol  = ((corrDF["Value"] == -999) | 
                   (corrDF["ConsistentW/Perturbation"] == "NA") | 
                   (corrDF["logFC"] == -999))
        corrDF  = corrDF.loc[~delCol, :]

        fig, axes = plt.subplots(2,3, figsize=(10,7))
        axes[0,0].set_xlabel("Spearman Correlation")
        axes[0,0].set_ylabel("Pearson Correlation")
        axes[0,0].scatter(x=corrDF[corrDF["Name"] == "Spearman"].iloc[:,0], 
                        y=corrDF[corrDF["Name"] == "Pearson" ].iloc[:,0], s=1)
        
        gs = axes[0, 2].get_gridspec()
        axes[0, 1].remove()
        axes[0, 2].remove()
        axes[1, 1].remove()
        axes[1, 2].remove()
        axlong = fig.add_subplot(gs[:, -1])
        try:
            sns.violinplot(data=corrDF, 
                        x="Value", 
                        y="Name", 
                        hue="ConsistentW/Perturbation", 
                        split=False,
                        cut=0,
                        scale="count",
                        ax=axlong)
            axlong.set_xlabel("Correlation Scores")
            
            sns.scatterplot(data=corrDF[corrDF["Name"] == "Spearman"],
                            x="Value",
                            y="logFC",
                            hue="ConsistentW/Perturbation", 
                            # style="ConsistentW/Perturbation", 
                            ax=axes[1,0])
            axes[1,0].set_xlabel("Spearman Correlation")
            plt.show()
        except Exception: #input is empty
            pass
    return spearmanList, pearsonList

def aggregate_by_perturbation(adata: anndata.AnnData, group_by: list, use_raw = True):
    """ Compute pseudo-bulk expression by adding raw counts.

    Args:
        adata (anndata.AnnData): Object with raw counts in adata.raw.X
        group_by (list of st): names of categorical columns in adata.obs to group by. Typically starts with "perturbation". 

    Returns:
        anndata.AnnData: Pseudo-bulk expression
    """
    if "is_control" not in group_by:
        group_by.append("is_control")
    if "perturbation" not in group_by:
        print("group_by should normally contain 'perturbation'. If any groups contain more than one perturbation, output metadata will be oversimplified.")
    assert all([g in adata.obs.columns for g in group_by]), "Each element of group_by must be in adata.obs.columns, and 'is_control' is also required."
    assert "group_index" not in group_by, "Apologies: adata.obs['group_index'] is reserved for internal use. Please rename this column."
    # Group the cells
    print("grouping", flush = True)
    # Avoid weird object or categorical dtypes
    for o in group_by:
        if o != "is_control":
            adata.obs[o] = adata.obs[o].astype("str")
    groups = adata.obs.groupby(group_by)
    groups = groups.size().reset_index().rename(columns={0:'count'})
    groups["group_index"] = groups.index
    print(f"Found {len(groups.index)} groups")
    # This merge yields a fast mapping from cells (input) to groups (output), but for speed,
    # we want a fast mapping from each group (input) to the cells in it (output).
    print("mapping groups to cells", flush = True)
    adata.obs = pd.merge(adata.obs, groups, how = "left")
    cells_by_group = {g:[] for g in groups["group_index"]}
    for g in adata.obs["group_index"]:
        assert g in cells_by_group, f"Unexpected group {g} found. Please report this error."
    for i,c in enumerate(adata.obs.index):
        try:
            cells_by_group[adata.obs.loc[c, "group_index"]].append(i)
        except Exception:
            print(adata.obs.loc[c,:].T)
            raise KeyError(f"Cell {(i,c)} has a bad group assignment or bad metadata (see print output).")
    # Finally, sum raw counts per group
    print("summing", flush = True)
    def do_one(i, g, X):
        return X[cells_by_group[g], :].sum(axis = 0) 
    rows = Parallel(n_jobs=cpu_count()-1, verbose = 1, backend="loky")(
        delayed(do_one)(i,g, X = adata.raw.X if use_raw else adata.X)
        for i,g in enumerate(groups["group_index"].unique())
    )
    print("collecting", flush = True)
    newX = scipy.sparse.lil_matrix((len(groups["group_index"].unique()), np.max(rows[0].shape)))
    for i,g in enumerate(groups["group_index"].unique()):
       newX[i,:] = rows[i]
    newX = newX.tocsr()
    newAdata    = sc.AnnData(
        newX, 
        var=adata.var.copy(),
        obs=groups,
    )
    newAdata.obs['is_control_int'] = [int(bool(x)) for x in newAdata.obs["is_control"]]
    gc.collect()
    return newAdata


def simulate_single_cells(adata: anndata.AnnData, num_cells: int, counts_per_cell: int):
    """Split bulk RNA samples into a simulated homogeneous population of cells.

    Args:
        adata (anndata.AnnData): bulk RNA perturbation data
        num_cells (int): how many cells to split each sample into
        counts_per_cell (int): how many reads or UMIs to simulate per cell

    Returns:
        anndata.AnnData: perturbation data with the same variables and num_cells times as many samples
    """
    assert (adata.X < 15).all(), "Input data are expected to be normalized and log1p-transformed"
    adata.X = np.exp(adata.X) - 1
    # Scale each sample to counts_per_cell total UMIs
    # This can be a non-integer because it will just be a Poisson rate param.
    for i in adata.obs_names:
        adata[i, :].X = adata[i,:].X * counts_per_cell / adata[i,:].X.sum()
    # Generate Poisson samples    
    new_adata = anndata.AnnData(
        X   = np.vstack([np.random.poisson(adata.X)   for _ in range(num_cells)]), 
        obs = pd.concat([adata.obs for _ in range(num_cells)]), 
        var = adata.var
    )
    new_adata.obs_names_make_unique()
    new_adata.X = scipy.sparse.csr_matrix(new_adata.X)
    new_adata.raw = new_adata.copy()
    new_adata.uns = adata.uns.copy()
    sc.pp.log1p(new_adata)
    # Fix the ELAP
    new_adata = describe_perturbation_effect(new_adata, perturbation_type="overexpression")
    return new_adata

def checkPerturbationEffectMetricCorrelation(adata: anndata.AnnData, metrics): 
    """Compute correlation between different measures of global effect size"""
    for (n1, n2) in it.combinations(metrics, r=2):
        assert n1 in adata.obs
        assert n2 in adata.obs
        perturbagen = adata.obs.perturbation 
        keepRow = ((adata.obs[n1] != -999) & 
                   (adata.obs[n2] != -999) & 
                   (adata.obs.perturbation.duplicated()) & 
                   (~adata.obs.is_control))
        metric1 = adata[keepRow].obs[n1]
        metric2 = adata[keepRow].obs[n2]

        plt.figure(figsize=(2,2))
        plt.scatter(metric1, 
                    metric2, 
                    s=1, 
                    label=f"{spearmanr(metric1, metric2)[0]:.3f}")
        plt.xlabel(n1)
        plt.ylabel(n2)
        plt.legend()
        plt.show()
    
    

def visualizePerturbationEffect(adata, metrics, TFDict, EpiDict):
    """Visualize effect size versus type of perturbation, e.g. TF versus non-TF"""
    # Everything is artificially shifted right by one 
    # since a non-visible bar (with height=0) is
    # inserted at the beginning. This is done to coerce the
    # legend color for TF/Chromatin Modifier to be 'orange' 
    keepRow      = (~adata.obs.perturbation.duplicated() & (~adata.obs.is_control)) 
    perturbagens = adata[keepRow].obs.perturbation
    colorDict    = {'No': '#1f77b4', 'Yes': '#ff7f0e'}
    perturbagenStatus = np.array(["Yes" if p in TFDict or p in EpiDict else "No" for p in perturbagens])
    perturbagenColor  = np.array([colorDict[p] for p in perturbagenStatus])

    for idx, m in enumerate(metrics):
        plt.figure(figsize=(25,3))
        metricValue = adata[keepRow].obs[m]
        ordering    = np.argsort(metricValue)[::-1]
        if 'mutualInfo' in m:
            index = index[::-1]
        plt.bar(np.arange(len(perturbagens)+1), 
                [0] + list(metricValue[ordering]), 
                width=0.6, 
                linewidth=0, 
                color=[colorDict['Yes']] + list(perturbagenColor[ordering]), 
                log=True, 
                label="TF/Chromatin Modifier")
        plt.bar(0, 
                0, 
                color=[colorDict['No']],
                label="Other Genes")
        plt.legend()
        plt.ylabel(f"{m} Value")
        otherGene = np.array([(i+1,p) for i,p in enumerate(perturbagens[ordering]) 
                              if perturbagenStatus[ordering][i] == 'No'])
        otherGene = np.insert(otherGene, 0, [0, ""], axis=0)
        plt.xticks(otherGene[:,0].astype(int), otherGene[:,1], rotation=90, fontsize=10)
        plt.xlim([0, len(perturbagens)+1])
        plt.show()


def visualizePerturbationMetadata(
    adata: anndata.AnnData, 
    x: str, 
    y: str, 
    style=None, 
    hue=None, 
    markers=None, 
    xlim=None, 
    s=30
):
    """Plot characteristics of each perturbation, e.g. correlation between replicates or global effect size."""
    if xlim is None:
        span = adata.obs[x].max() - adata.obs[x].min()
        xlim = [adata.obs[x].min()-span/10, adata.obs[x].max()+span/10]
    validMat = (adata.obs[x] != -999) & (adata.obs[y] != -999) & (~adata.obs.is_control)
    print(f"{len(validMat)} number of points are plotted")
    plt.figure(figsize=(8, 5))
    g =sns.scatterplot(data=adata.obs[validMat], 
                       x=x,
                       y=y,
                       style=style, 
                       hue=hue, 
                       markers=markers,
                       palette=sns.color_palette("coolwarm", as_cmap=True), 
                       legend='brief', 
                       s=s)
    plt.axhline(0, 0, 1, linestyle='-.', color='brown')
    g.legend(loc='lower right', bbox_to_anchor=(1.4, 0), ncol=1)
    plt.ylabel(f"{y} of Perturbed Gene")
    plt.xlim(xlim)
    plt.show()

