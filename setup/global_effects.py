from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd
import os
import itertools as it
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns


def try_toarray(x):
    try:
        x = x.toarray()
    except Exception:
        pass
    return x

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
        plt.hist(    pVals, bins=np.linspace(-0, 1, 102), alpha=0.5)
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
    

def computeFoldChangeStats(treatment: np.ndarray, control: np.ndarray, pseudocount = 0):
    """ Compute the log fold change between the treatment group and
    the control group across all genes. """
    logFCStat = np.abs(np.nanmedian(np.log2( (treatment + pseudocount) / (control + pseudocount) ), axis=0))
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
    arr = np.hstack([np.array(v)[:, np.newaxis] for v in variables])
    df  = pd.DataFrame(arr, columns=names)
    df.to_csv(filename, index=False)    
    

def quantifyEffect(
    adata, 
    fname: str,                
    group: str=None, 
    diffExprFC=True, 
    prefix: str="",
    withDEG: bool=True,
    withMI: bool=True, 
    pseudocount = 0
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
    pseudocount (float): pseudocount to use when computing logfc from normalized expression values
    """
    try:
        if group:
            assert group in adata.obs.columns
    except:
        print(f"The specified column {group} does not exist in adata.obs")

    perturbs = sorted(set(adata[~adata.obs.is_control].obs.perturbation))
    normX    = try_toarray(adata.X.copy())
    
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
            # print(idx, p, deg[idx], mean[idx], median[idx], norm2[idx], mi[idx])
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
            outStat    .append(computeFoldChangeStats(t, c, pseudocount))

            
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
        # print(idx, p, deg[idx], mean[idx], median[idx], norm2[idx], mi[idx])

        if (idx + 1) % 30 == 0 or idx == len(perturbs) - 1:           
            saveToFile([deg, mi, mean, norm2, median], names, fname)
        
    metrics = [deg, mi, mean, norm2, median]
    names   = ['DEG', 'MI', 'logFCMean', 'logFCNorm2', 'logFCMedian']
    
    for mCount in range(5):
        adata.obs[f'{prefix}{names[mCount]}'] = np.full(adata.n_obs, -999, dtype=np.float64)
        for pCount, p in enumerate(perturbs):
            rows = adata.obs.index[adata.obs.perturbation == p]
            adata.obs.loc[rows, f'{prefix}{names[mCount]}'] = metrics[mCount][pCount]