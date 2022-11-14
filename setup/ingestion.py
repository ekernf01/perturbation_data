from typing import Iterable
import numpy as np 
import pandas as pd 
import h5py
import scanpy as sc
import anndata
import typing 

# For QC
import os, sys
import itertools as it
from scipy.stats import spearmanr, pearsonr, rankdata, f_oneway
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns


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


def tmmNormalization(matrix, trim_fold_change=0.3, trim_abs_expr=0.05):
    """
    Trimmed mean of M-values normalization
    from https://github.com/ficusss/PyGMNormalize/blob/master/pygmnormalize/normalizations.py
    """
    matrix_np = np.array(matrix)                      # better speed of calculating
    np.seterr(divide='ignore', invalid='ignore')      # for divide on zeros in log2
    
    # Calculation log2(tmm_factor)
    def log2Tmm(index_vec):
        # select the necessary vectors
        curr_vec = matrix_np[:, index_vec]
        ref_vec  = matrix_np[:, index_ref]
        
        # total number molecules in cells
        total_curr_vec = np.sum(curr_vec)
        total_ref_vec = np.sum(ref_vec)
        
        # select significant genes
        check_inf = (~np.isinf(matr_a[:, index_vec])) & (~np.isinf(matr_m[:, index_vec]))
        ranks = rankdata(matr_a[:, index_vec][check_inf], method='ordinal')
        bool_a = (ranks > len(ranks) * trim_abs_expr) & (ranks < len(ranks) * (1 - trim_abs_expr))
        ranks = rankdata(matr_m[:, index_vec][check_inf], method='ordinal')
        bool_m = (ranks > len(ranks) * trim_fold_change) & (ranks < len(ranks) * (1 - trim_fold_change))
        curr_vec = curr_vec[check_inf]
        ref_vec = ref_vec[check_inf]
        bool_curr_vec = curr_vec > 0
        bool_ref = ref_vec > 0
        bool_result = bool_curr_vec & bool_ref & bool_a & bool_m
        
        # calculation of required values
        w_vec = 1 / ((total_curr_vec - curr_vec[bool_result]) / (total_curr_vec * curr_vec[bool_result]) + 
                     (total_ref_vec - ref_vec[bool_result]) / (total_ref_vec * ref_vec[bool_result]))
        m_vec = np.log2(curr_vec[bool_result] / total_curr_vec) - np.log2(ref_vec[bool_result] / total_ref_vec)
        
        # calculation log2(tmm_factor)
        w_sum = np.sum(w_vec)
        if np.isclose(w_sum, 0) or np.isinf(w_sum):
            print("Unexpected sum of weights for vector {}: '{}'".format(index_vec, w_sum), file=sys.stderr)
            return 0
            pass
        
        return np.sum(w_vec * m_vec) / w_sum
    
    f75 = np.percentile(matrix_np[np.any(matrix_np > 0, axis=1)], 75, axis=0)
    index_ref = np.argmin(abs(f75 - np.mean(f75)))

    # find matrix A and M described expression levels of genes
    matr_norm = matrix_np / np.sum(matrix_np, axis=0)
    matr_a = np.log2(matr_norm * matr_norm[:, index_ref].reshape(matr_norm.shape[0], 1)) / 2
    matr_m = np.log2(matr_norm / matr_norm[:, index_ref].reshape(matr_norm.shape[0], 1))
    
    # calculation tmm_factor and normalization of input data
    tmm_factor = 2 ** np.array([log2Tmm(i) for i in range(matrix_np.shape[1])])
    return matrix / tmm_factor


def checkConsistency(adata: anndata.AnnData, 
                     perturbationType: str="overexpression", 
                     group: str=None,
                     verbose: bool=False):
    """ Check whether the gene that was perturbed is actually 
    measured to be higher (if overexpressed) or lower (if knocked
    down) or nearly zero (if knocked off).
    If a perturbagen is a control or is not measured, 'N/A' is labeled. 
    If a perturbagen's expression is higher or lower than all 
    control groups (matching the direction of intended perturbation),
    True is labeled; otherwise, False is labeled.
    
    Args:
        adata (anndata.AnnData): the object to operate on.
        perturbation_type (str): one of {"overexpression", "knockout", "knockdown"}
        group (str, default None): a column in adata.obs to indicate sub-group of
                                   the treatment and the control.
        verbose (bool): show a swarmplot noting the difference between the control
                        and the treatment, if the perturbation direction and expression
                        level are disconcordant.
    """
    def visualizeLogFC(fc):
        validLogFC = fc[fc != -999]
        rangeMin = np.floor(np.min(validLogFC))
        rangeMax = np.ceil (np.max(validLogFC))
        plt.figure(figsize=(4,2.5))
        plt.hist(validLogFC, 
                 bins=np.linspace(rangeMin, 
                                  rangeMax, 
                                  int((rangeMax-rangeMin)*3+1)), 
                 label="Per Trial")
        plt.axvline(0, 0, 1, color='red', label="No Change")
        plt.xlabel("Log2 Fold Change (perturbed/control)")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        
    
    assert perturbationType in ["overexpression", "knockout", "knockdown"]
    
    tmmNormX     = tmmNormalization(adata.X.T).T
    controlIndex = np.where(adata.obs.is_control)[0]
    control      = tmmNormX[controlIndex, :]
    logFC        = np.full((adata.n_obs), -999.0)
    consistencyStatus = np.full((adata.n_obs), "Yes")
    
    for row, perturbagen in sorted(enumerate(adata.obs.perturbation), key=lambda x: x[1]):
        if (adata.obs.is_control[row] or perturbagen not in adata.var.index):
            consistencyStatus[row] = "NA"
            continue
        loc = np.where(adata.var.index == perturbagen)[0]
        assert loc.shape[0] == 1
        
        if group:        # Only compare treatment to within-group controls (to limit variability)
            assert group in adata.obs.columns
            control = tmmNormX[adata.obs.is_control & (adata.obs[group] == adata.obs[group][row]), :]
            
        logFC[row] = np.log2(tmmNormX[row, loc[0]] / np.median(control[:, loc]))
        
        if perturbationType == "overexpression" and tmmNormX[row, loc] > np.median(control[:, loc]):
            continue
        if perturbationType == "knockdown"      and tmmNormX[row, loc] < np.median(control[:, loc]):
            continue
        if perturbationType == "knockout"       and abs(tmmNormX[row, loc]) < 1e-3:
            continue
        consistencyStatus[row] = "No"  
        
        if verbose:
            plt.figure(figsize=(4,1))
            g = sns.swarmplot(control[:, loc].flatten(), orient='h', label="control")
            g.axvline(tmmNormX[row, loc], 0, 1, color='red', label="treatment", lw=1)
            g.legend()
            plt.title(f"{perturbagen} {perturbationType}")
            plt.show()
            
    visualizeLogFC(logFC)
    return consistencyStatus, logFC


def computeCorrelation(adata: anndata.AnnData, 
                       verbose: bool=False, 
                       group: str=None):
    """
    Assume the existence of **is_control** in adata.obs. Compute the
    correlation between biological replica on scale of log fold change. For each 
    set of perturbation, the final correlation score is the median of 
    correlation between all pair-wisecombination of perturbation expression
    and control expression. Both Spearman and Pearson correlation are
    computed.
    """
    
    def computelogFC(t1, t2, c1, c2):
        logFC1, logFC2 = np.log2(r1 / c1), np.log2(r2 / c2)
        validGeneEntry = np.isfinite(logFC1) & np.isfinite(logFC2)
        logFC1 = logFC1[validGeneEntry]
        logFC2 = logFC2[validGeneEntry]
        return spearmanr(logFC1, logFC2)[0], pearsonr (logFC1, logFC2)[0]
    
    
    tmmNormX = tmmNormalization(adata.X.T).T
    spearmanList = np.full(adata.n_obs, fill_value=-999, dtype=np.float64)
    pearsonList  = np.full(adata.n_obs, fill_value=-999, dtype=np.float64)
    controlExpr  = tmmNormX[adata.obs.is_control, :]
    
    for perturbagen in set(adata[~adata.obs.is_control].obs.perturbation):
        
        # All perturbation expressions
        replicaRow  = np.where(adata.obs.perturbation == perturbagen)[0]
        if replicaRow.shape[0] == 1:        # skip perturbation w/o replication
            continue
        
        temp1, temp2 = list(), list()
        for (row1, row2) in it.combinations(replicaRow, 2):
            r1, r2 = tmmNormX[row1, :], tmmNormX[row2, :]
            if group:
                assert group in adata.obs.columns
                g1, g2 = adata.obs[group][row1], adata.obs[group][row2]
                c1 = np.median(tmmNormX[(adata.obs.is_control) & (adata.obs[group] == g1), :], axis=0)
                c2 = np.median(tmmNormX[(adata.obs.is_control) & (adata.obs[group] == g2), :], axis=0)
                s, p = computelogFC(r1, r2, c1, c2)
            else:
                c1 = np.median(controlExpr.copy(), axis=0)
                s, p = computelogFC(r1, r2, c1, c1)
            temp1.append(s)
            temp2.append(p)                
        
        # Record the median values
        spearmanList[replicaRow] = np.median(temp1)
        pearsonList [replicaRow] = np.median(temp2)
            
    if verbose:
        corrVal = np.array([spearmanList, pearsonList]).flatten()[:, np.newaxis]
        corrStr = np.array(["Spearman"] * len(spearmanList) + 
                           ["Pearson" ] * len(pearsonList)).flatten()[:, np.newaxis]
        
        assert "consistentW/Perturbation" in adata.obs
        assert "logFC" in adata.obs

        status  = np.tile(adata.obs['consistentW/Perturbation'], 2)[:, np.newaxis]
        logFC   = np.tile(adata.obs['logFC']                   , 2)[:, np.newaxis]
        corrDF  = pd.DataFrame(np.hstack([corrVal, corrStr, logFC, status]), 
                               columns=["Value", "Name", "logFC", "ConsistentW/Perturbation"])
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
        
    return spearmanList, pearsonList


# ============================= separator ============================= #
#                                                                       #
#            Here is the START of code that computes bigness            #
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


def computeFoldChangeStats(treatment: np.ndarray, control: np.ndarray):
    """ Compute the log fold change between the treatment group and
    the control group. If all log(FC) are positive and the smallest
    one is at least log2, or if all are negative and the largest
    one is at least -log2, we claim the change is significant. 
    Otherwise, we say it is not significant. Also, if some log(FC)
    are positive and some are negative, we say that there is no change """
    logFCStat = list()
    for idx in range(treatment.shape[1]):
        fc = [np.log2(t/c) for (t,c) 
              in it.product(treatment[:,idx], 
                            control  [:,idx])]
        logFCStat.append(abs(np.median(fc)))
    
    return (np.mean(logFCStat), 
            np.median(logFCStat), 
            np.linalg.norm(logFCStat, ord=2))
    

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
    

def quantifyEffect(adata, fname: str, group: str=None, diffExprFC=True) -> float:
    """ Compute the metrics that evaluate the biggness 
    of perturbation
    
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
    tmmNormX = tmmNormalization(adata.X.T).T
    control = tmmNormX[adata.obs.is_control, :]
    names   = ["deg", "mi", "mean", "norm2", "median"]
    metric  = [list() for i in range(len(names))]
    category= set(adata.obs[group]) if group else ["all"]
    
    # Load what's already computed, if any
    (deg, mi, mean, norm2, median) = readFile(perturbs, metric, names, fname)
    
    for idx, p in enumerate(perturbs):
        
        # Retrieve all perturbation related to a given TF
        rows = np.where(adata.obs.perturbation == p)[0]
        treatment = tmmNormX[adata.obs.perturbation == p, :]
        
        # If missing values, re-compute
        if -1 not in [deg[idx], mean[idx], norm2[idx], median[idx], mi[idx]]:
            continue
        
        # If stratify by group, record each group and take the median across groups.
        fcThreshold, pVThreshold, outStat, mutualInfo = [], [], [], []
        for g in category:
            t = (tmmNormX[(adata.obs.perturbation == p) &
                          (adata.obs[group] == g), :] 
                 if group else treatment)
            c = (tmmNormX[(adata.obs.is_control) & 
                          (adata.obs[group] == g), :] 
                 if group else control)
            fcThreshold.append(thresholdByFoldChange (t, c))
            pVThreshold.append(thresholdByFDR        (t, c, verbose=False))
            outStat    .append(computeFoldChangeStats(t, c))
            mutualInfo .append(calcMI                (t, c, 100))
        
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

        deg   [idx] = sum(fcThreshold & pVThreshold) if diffExprFC else sum(pVThreshold)
        mean  [idx] = np.median([o[0] for o in outStat])
        median[idx] = np.median([o[1] for o in outStat])
        norm2 [idx] = np.median([o[2] for o in outStat])
        mi    [idx] = np.median(mutualInfo)
        print(idx, p, deg[idx], mean[idx], median[idx], norm2[idx], mi[idx])

        if (idx + 1) % 10 == 0 or idx == len(perturbs) - 1:           
            saveToFile([deg, mi, mean, norm2, median], names, fname)
        
    return (np.array(deg),
            np.array(mi),
            np.array(mean),
            np.array(norm2),
            np.array(median))


# ============================= separator ============================= #
#                                                                       #
#              Here is the END of code that computes bigness            #
#                                                                       #
# ===================================================================== #
