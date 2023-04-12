from typing import Iterable
import numpy as np 
import pandas as pd 
import h5py
import scanpy as sc
import anndata
import typing 
import gc
from joblib import Parallel, delayed, cpu_count, dump

# For QC
import os, sys
import itertools as it
import scipy
from scipy.stats import spearmanr, pearsonr, rankdata, f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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



def checkConsistency(adata: anndata.AnnData, 
                     perturbationType: str="overexpression", 
                     group: str=None,
                     verbose: bool=False, 
                     do_return_pval = False, 
                     show_plots = False):
    """ Check whether the gene that was perturbed is actually 
    measured to be higher (if overexpressed) or lower (if knocked
    down) or nearly zero (if knocked out).
    If a perturbagen is a control or is not measured, 'N/A' is labeled. 
    If a perturbagen's expression is higher or lower than all 
    control groups (matching the direction of intended perturbation),
    True is labeled; otherwise, False is labeled.
    
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
    def visualizeLogFC(fc, pval = None):
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
        
    
    assert perturbationType in ["overexpression", "knockout", "knockdown"]
    
    normX        = adata.X.copy()
    controlIndex = np.where(adata.obs.is_control)[0]
    control      = normX[controlIndex, :]
    logFC        = np.full((adata.n_obs), -999.0)
    pval         = np.full((adata.n_obs), -999.0)
    consistencyStatus = np.full((adata.n_obs), "Yes")
    
    for row, perturbagen in sorted(enumerate(adata.obs.perturbation), key=lambda x: x[1]):
        if (adata.obs.is_control[row] or perturbagen not in adata.var.index):
            consistencyStatus[row] = "NA"
            continue
        loc = np.where(adata.var.index == perturbagen)[0]
        assert loc.shape[0] == 1
        
        if group:        # Only compare treatment to within-group controls (to limit variability)
            assert group in adata.obs.columns
            control = normX[adata.obs.is_control & (adata.obs[group] == adata.obs[group][row]), :]
        logFC[row] = np.log2(try_toarray(normX[row, loc[0]]) / np.median(try_toarray(control[:, loc])))   
        has_same_perturbation = perturbagen == adata.obs.perturbation
        pval[row] = ttest_ind(
            np.log2(try_toarray(normX[has_same_perturbation, loc[0]])), 
            np.log2(try_toarray(control[:, loc[0]])), 
            equal_var=True,
        ).pvalue
        if perturbationType == "overexpression" and normX[row, loc] > np.median(control[:, loc]):
            continue
        if perturbationType == "knockdown"      and normX[row, loc] < np.median(control[:, loc]):
            continue
        if perturbationType == "knockout"       and abs(normX[row, loc]) < 1e-3:
            continue
        consistencyStatus[row] = "No"  
        
        if verbose:
            plt.figure(figsize=(4,1))
            g = sns.swarmplot(control[:, loc].flatten(), orient='h', label="control")
            g.axvline(normX[row, loc], 0, 1, color='red', label="treatment", lw=1)
            g.legend(loc='lower right', bbox_to_anchor=(1.45, 0), ncol=1)
            plt.title(f"{perturbagen} {perturbationType}")
            plt.show()
            
    # NaN -> treatment = 0, control = 0
    # posInf -> treatment > 0, control = 0 
    # negInf -> treatment = 0, control > 0 (knocked out)
    consistencyStatus[np.isnan(logFC)] = "No"
    logFC[np.isnan(logFC)] = 0
    logFC[np.isposinf(logFC)] = np.nanmedian(logFC[(logFC > 0) & (logFC != -999) & np.isfinite(logFC)])
    logFC[np.isneginf(logFC)] = np.nanmedian(logFC[(logFC < 0) & (logFC != -999) & np.isfinite(logFC)])
    
    if do_return_pval:
        visualizeLogFC(logFC, pval)
        return consistencyStatus, logFC, pval
    else:
        visualizeLogFC(logFC)
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

        fig, axes = plt.subplots(2,3, figsize=(10,7), width_ratios=[0.4, 0.1, 0.5])
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
    # Make sure we have all the metadata we will need
    output_metadata_fields = list(set(group_by).union({"perturbation", 'is_control'}))
    if "perturbation" not in group_by:
        print("group_by should normally contain 'perturbation'. If any groups contain more than one perturbation, output metadata will be oversimplified.")
    assert all([g in adata.obs.columns for g in output_metadata_fields]), "Each element of group_by must be in adata.obs.columns, and 'perturbation' and 'is_control' are also required."
    assert "group_index" not in group_by, "Apologies: adata.obs['group_index'] is reserved for internal use. Please rename this column."
    # Group the cells
    print("grouping", flush = True)
    for o in group_by:
        adata.obs[o] = adata.obs[o].astype("str")
    groups = adata.obs.groupby(output_metadata_fields)
    groups = groups.size().reset_index().rename(columns={0:'count'})
    groups["group_index"] = groups.index
    print(f"Found {len(groups.index)} groups")
    print("Number of groups: " + str(groups.shape[0]))
    # This merge yields a fast mapping from cells (input) to groups (output), but for speed,
    # we want a fast mapping from each group (input) to the cells in it (output).
    print("mapping groups to cells", flush = True)
    adata.obs = pd.merge(adata.obs, groups, how = "left")
    cells_by_group = {g:[] for g in groups["group_index"]}
    assert all(g in cells_by_group for g in adata.obs["group_index"]), "Unexpected group found. Please report this error."
    for c in adata.obs.index:
        try:
            cells_by_group[adata.obs.loc[c, "group_index"]].append(c)
        except Exception:
            print(adata.obs.loc[c,:].T)
            raise KeyError(f"Cell {c} has a bad group assignment or bad metadata (see print output).")
    # Finally, sum raw counts per group
    print("summing", flush = True)
    def do_one(i,g,X):
        just_ones = [1 for _ in cells_by_group[g]]
        just_zero = [0 for _ in cells_by_group[g]]
        indicator = scipy.sparse.csr_matrix((just_ones, (just_zero, cells_by_group[g])), shape = (1, adata.n_obs))
        return indicator.dot(X)
    # I use 8x CPU count because this code make poor use of available cores and the memmap makes 
    # extra threads nearly free memory-wise. 
    results = Parallel(n_jobs=8*cpu_count(), verbose = 1, backend="loky")(
        delayed(do_one)(i,g, X = adata.raw.X if use_raw else adata.X)
        for i,g in enumerate(groups["group_index"].unique())
    )
    # Put list of results in a matrix
    print("reshaping")
    newX = scipy.sparse.lil_matrix((len(groups["group_index"].unique()), adata.n_vars))
    for i,g in enumerate(groups["group_index"].unique()):
        newX[i,:] = results[i]
        results[i] = []
    newX = newX.tocsr()
    newAdata    = sc.AnnData(
        newX, 
        var=adata.var.copy(),
        obs=groups,
    )
    newAdata.obs['is_control_int'] = [int(x) for x in newAdata.obs["is_control"]]
    gc.collect()
    return newAdata

# ============================= separator ============================= #
#                                                                       #
#     Here is the START of code that computes global effect size        #
#                                                                       #
# ===================================================================== #


def thresholdByFoldChange(treatment: np.ndarray, control: np.ndarray):
    """ Compute the log fold change between the treatment group and
    the control group. If all log(FC) are positive and the smallest
    one is at least log2, or if all are negative and the largest
    one is at least -log2, we claim the change is significant. 
    Otherwise, we say it is not significant. Also, if some log(FC)
    are positive and some are negative, we say that there is no change """
    fcThreshold = list()
    for idx in range(treatment.shape[1]):
        fc = [np.log2(t/c) for (t,c) 
              in it.product(treatment[:,idx], 
                            control  [:,idx])]
        fc = np.array(fc)
        if all(fc > 0):
            fcThreshold.append(np.min(fc))
        elif all(fc < 0):
            fcThreshold.append(np.max(fc))
        else:
            fcThreshold.append(0)        
#         if ( (all(fc > 0) and np.min(fc) >  np.log(2)) or
#              (all(fc < 0) and np.max(fc) < -np.log(2)) ):
#             fcThreshold.append(True)
#         else:
#             fcThreshold.append(False)
    return fcThreshold


def thresholdByFDR(treatment: np.ndarray, control: np.ndarray, verbose: bool=False):
    """ Compute the false discovery rate for ANOVA between the
    treatment group and the control group. First compute the p-value
    under the standard ANOVA procedure (assuming variances are equal
    across groups). Then, adjust for FDR with Benjamini-Hochberg 
    procedure. Finally, threshold at 0.05.
    """
    samples = [[treatment[:, idx], 
                control  [:, idx]] for idx in range(control.shape[1])]
    pVals   = np.array([f_oneway(*s).pvalue for s in samples])
    pVals[np.isnan(pVals)] = 1
    pAdjusted = multipletests(pVals, method='fdr_bh')[1]
    if verbose:
        plt.figure(figsize=(3,3))
        plt.hist(pVals, bins=np.linspace(-0, 1, 102), alpha=0.5)
        plt.hist(pAdjusted, bins=np.linspace(-0, 1, 102), alpha=0.5)
        plt.show()
    return pAdjusted < 0.05
    

def calcMI(treatment: np.ndarray, control: np.ndarray, bins: int=100, verbose: bool=False):
    """ Compute the mutual information between
    the treatment and control. The expression values
    of the control (before perturbation) and the 
    treatment (after perturbation) are binned and 
    a discrete joint probability mass function 
    is computed. """
    miList = list()
    
    for (i,j) in it.product(treatment, control):        
        c_xy = np.histogram2d(np.log2(i), np.log2(j), bins)[0]
        miList.append(mutual_info_score(None, None, contingency=c_xy))
        if verbose:
            sns.heatmap(np.log2(c_xy))
            plt.show()
    return np.median(miList)
    

def computeFoldChangeStats(treatment: np.ndarray, control: np.ndarray):
    """ Compute the log fold change between the treatment group and
    the control group across all genes. """
    logFCStat = np.abs(np.nanmedian(np.log2(treatment / control), axis=0))
    logFCStat = logFCStat[np.isfinite(logFCStat)]
    
    return (np.mean(logFCStat), 
            np.median(logFCStat), 
            np.linalg.norm(logFCStat, ord=2))

    
def readFile(perturbs: list[str], variables: list, names: str, filename: str):
    """ Variables are 
    ['deg', 'mi', 'mean', 'norm2', 'median'] 
    Because running hundreds of thousands of ANOVA
    takes time, the results are saved in a file, 
    enabling resumption at any time. """
    if os.path.exists(filename):
        df = pd.read_csv(filename, header=0)
        for idx, v in enumerate(variables):
            variables[idx] = df[names[idx]]
    else:
        for idx, v in enumerate(variables):
            variables[idx] = np.full(len(perturbs), -1, dtype=np.float64)
    return variables
    

def saveToFile(variables: list, names: str, filename: str):
    """ Save current progress to disk periodically """
    arr = np.hstack([v[:, np.newaxis] for v in variables])
    df  = pd.DataFrame(arr, columns=names)
    df.to_csv(filename, index=False)    
    

def quantifyEffect(
    adata, 
    fname: str,                
    group: str=None, 
    diffExprFC=True, 
    prefix: str="",
    withDEG: bool=True,
    withMI: bool=True
) -> tuple[np.ndarray]:
    """ Compute the metrics that evaluate the global transcriptomic effect size 
    of each perturbation
    
    adata (anndata.AnnData): expression matrix + metadata
    fname (str): path to the file that stores the metrics
    group (str): whether to group treatment/control that 
                 come from the same biological specimen
    diffExprFC (bool): when counting the number of differentially
                       expressed genes, whether to threshold based
                       on log fold change or not.
    """
    try:
        if group:
            assert group in adata.obs.columns
    except:
        print(f"The specified column {group} does not exist in adata.obs")

    perturbs = sorted(set(adata[~adata.obs.is_control].obs.perturbation))
    normX    = adata.X.copy()
    
    columnToKeep = list(set(range(normX.shape[1])) - 
                        set(np.where(normX == 0)[1]))
    print(len(columnToKeep))
    
    control = normX[adata.obs.is_control, :]
    names   = ["deg", "mi", "mean", "norm2", "median"]
    metric  = [list() for i in range(len(names))]
    category= set(adata.obs[group]) if group else ["all"]
    
    # Load what's already computed, if any
    (deg, mi, mean, norm2, median) = readFile(perturbs, metric, names, fname)
    
    for idx, p in enumerate(perturbs):
        
        # Retrieve all perturbation related to a given TF
        rows = np.where(adata.obs.perturbation == p)[0]
        treatment = normX[adata.obs.perturbation == p, :]
        
        # If missing values, re-compute
        if -1 not in [deg[idx], mean[idx], norm2[idx], median[idx], mi[idx]]:
            print(idx, p, deg[idx], mean[idx], median[idx], norm2[idx], mi[idx])
            continue
        
        # If stratify by group, record each group and take the median across groups.
        fcThreshold, pVThreshold, outStat, mutualInfo = [], [], [], []
        for g in category:
            t = (normX[(adata.obs.perturbation == p) &
                       (adata.obs[group] == g), :] 
                 if group else treatment)
            c = (normX[(adata.obs.is_control) & 
                       (adata.obs[group] == g), :] 
                 if group else control)

            # have to use all c to increase the power of statistical tests
            if withDEG:
                fcThreshold.append(thresholdByFoldChange (t, c))
                pVThreshold.append(thresholdByFDR        (t, c, verbose=False))
                
            # Use median of control to skip over many computation
            c = np.median(c, axis=0)[np.newaxis,:]
            if withMI:
                mutualInfo .append(calcMI                (t[:,columnToKeep], 
                                                          c[:,columnToKeep], 
                                                          100))
            outStat    .append(computeFoldChangeStats(t, c))

            
        if withDEG:
            # If log fold change across samples are consistent (in direction) and meet the threshold
            fcThreshold = np.array(fcThreshold)
            fcThreshold = [fcThreshold[:, col] for col in range(fcThreshold.shape[1])]
            temp = [True if ( (all(pVal > 0) or all(pVal < 0)) and
                              (np.min(np.abs(pVal)) > np.log2(2)) ) 
                    else False
                    for pVal 
                    in fcThreshold]
            fcThreshold = np.array(temp)
            pVThreshold = np.multiply.reduce(pVThreshold, axis=0)

        # Record a given perturbagen's effective.
        if withDEG:
            deg[idx] = sum(fcThreshold & pVThreshold) if diffExprFC else sum(pVThreshold)
        else:
            deg[idx] = -999
        if withMI:
            mi[idx] = np.median(mutualInfo)
        else:
            mi[idx] = -999

        mean  [idx] = np.median([o[0] for o in outStat])
        median[idx] = np.median([o[1] for o in outStat])
        norm2 [idx] = np.median([o[2] for o in outStat])
        print(idx, p, deg[idx], mean[idx], median[idx], norm2[idx], mi[idx])

        if (idx + 1) % 30 == 0 or idx == len(perturbs) - 1:           
            saveToFile([deg, mi, mean, norm2, median], names, fname)
        
    metrics = [deg, mi, mean, norm2, median]
    names   = ['DEG', 'MI', 'logFCMean', 'logFCNorm2', 'logFCMedian']
    
    for mCount in range(5):
        adata.obs[f'{prefix}{names[mCount]}'] = np.full(adata.n_obs, -999, dtype=np.float64)
        for pCount, p in enumerate(perturbs):
            rows = np.where(adata.obs.perturbation == p)[0]
            adata.obs[f'{prefix}{names[mCount]}'][rows] = metrics[mCount][pCount]


# ============================= separator ============================= #
#                                                                       #
#              Here is the END of code that computes bigness            #
#                                                                       #
# ===================================================================== #


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

