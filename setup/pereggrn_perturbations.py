# altered for paul dataset
"""
This module validates the format of perturbation datasets by doing the following:

- Setting and retrieving paths for perturbation data via environment variables
- Loading perturbation datasets and their metadata from specified directories
- Validating the integrity and format of perturbation data using custom criteria, 
    such as checking the presence and format of required files, ensuring consistency 
    between datasets, and confirming the correctness of dataset contents.

The module relies on the pandas, scanpy, anndata, numpy, and os libraries to perform these tasks. 
It is designed to work with specific data structures expected in biological datasets, particularly those involving gene perturbations.

Usage:
- Before loading or validating datasets, set the data path using `set_data_path()`
- Load dataset metadata with `load_perturbation_metadata()`
- Validate datasets using `check_perturbation_dataset()` to ensure they meet expected formats and criteria

Exceptions are raised for missing files, incorrect paths, or data inconsistencies, aiding in troubleshooting and ensuring data reliability.
"""

import os            # Import the os module to interact with the operating system
import pandas as pd  # Import the pandas library for data manipulation and analysis
import scanpy as sc  # Import the scanpy library for analyzing single-cell sequencing data
import anndata       # Import the anndata library to handle annotated data matrices in biology
import numpy as np   # Import the numpy library for numerical operations

def get_data_path():
    """
    Retrieve the path to the perturbation data from an environment variable.

    Returns:
        str: The path stored in the 'PERTURBATION_PATH' environment variable.
    """
    return os.environ['PERTURBATION_PATH'] # Return the path stored in the 'PERTURBATION_PATH' environment variable

def set_data_path(path: str):
    """
    Set the path for perturbation data in an environment variable and check for the existence of required files.

    Args:
        path (str): Path to the directory containing perturbation datasets.

    Raises:
        FileNotFoundError: If the required files are not found at the specified path.
    """
    # Check if the 'perturbations.csv' file exists at the specified path -- not needed for paul?
    if not os.path.isfile(os.path.join(path, "perturbations.csv")): 
       raise FileNotFoundError("perturbations.csv should be a file in the folder whose name is provided to this function.")
    # Check if the 'test.h5ad' file exists within the 'nakatake' sub-directory of the specified path
    if not os.path.isfile(os.path.join(path, "paul", "test.h5ad")):
        raise FileNotFoundError("There should be an AnnData file at <your_input>/nakatake/test.h5ad (and others like it for the other datasets).")
    os.environ['PERTURBATION_PATH'] = path # Set the 'PERTURBATION_PATH' environment variable to the validated path
    return

def load_perturbation_metadata():
    """
    Load metadata about perturbations from a CSV file at the designated path.

    Returns:
        pandas.DataFrame: DataFrame containing perturbation metadata.

    Raises:
        KeyError: If the required environment variable is not set.
    """
    # Attempt to load and return the CSV file containing perturbation metadata
    try:
        return pd.read_csv(os.path.join(get_data_path(), "perturbations.csv"))
    # Raise a KeyError if the required environment variable is not set
    except KeyError as e:
        raise(KeyError("Before using the data you must call set_data_path('path/to/collection') to point to the perturbation data collection."))

def load_perturbation(dataset_name: str, is_timeseries: bool = False):
    """
    Load a perturbation dataset from an AnnData file.

    Args:
        dataset_name (str): Name of the dataset, taken from the metadata row names.
        is_timeseries (bool, optional): If True, loads the training data without perturbation (usually a timecourse). Defaults to False.

    Returns:
        anndata.AnnData: Perturbation data in a uniform format.

    Raises:
        KeyError: If the dataset cannot be found at the specified path.
    """
    t = "train" if is_timeseries else "test" # Determine the dataset type based on whether it is part of a time series
    # Attempt to load and return the AnnData file for the specified dataset
    try:
        return sc.read_h5ad(os.path.join(get_data_path(), dataset_name, f"{t}.h5ad"))
    # Raise a KeyError if the dataset cannot be found at the specified path
    except KeyError as e:
        raise(KeyError("Dataset not found at this path. Before using the data you must call set_data_path('path/to/collection') to point to the perturbation data collection."))

# Function to validate the format and integrity of a loaded perturbation dataset
def check_perturbation_dataset(dataset_name: str = None, ad: anndata.AnnData = None, is_timeseries = False, do_full = False, is_perturbation = True):
    """
    Validate the format and integrity of a loaded perturbation dataset.

    Args:
        dataset_name (str, optional): Name of the dataset. Provide exactly one of `dataset_name` or `ad`.
        ad (anndata.AnnData, optional): AnnData object containing perturbation data. Provide exactly one of `dataset_name` or `ad`.
        is_timeseries (bool, optional): If True, performs checks specific to time series data. Defaults to False.
        do_full (bool, optional): If True, performs a full validation, including more expensive checks. Defaults to False.
        is_perturbation (bool, optional): If True, treats the data as a perturbation dataset with additional metadata. Defaults to True.

    Returns:
        bool: True if the input data are correctly formatted.

    Raises:
        ValueError: If both or neither `dataset_name` and `ad` are provided.
        AssertionError: If various expected conditions are not met.
    """
    # Ensure that exactly one of 'ad' or 'dataset_name' is provided
    if ad is None and dataset_name is None:
        raise ValueError("Provide exactly one of ad and dataset_name")
    if not ad is None and not dataset_name is None:
        raise ValueError("Provide exactly one of ad and dataset_name")
    if ad is None and dataset_name is not None: 
        # A tiny bit of recursion helps us check a dataset with separate train and test folds. 
        # The base-case: AnnData input.
        try:
            # Assert gene names are consistent between training and test data.
            assert all(load_perturbation(dataset_name, is_timeseries = True).var_names == load_perturbation(dataset_name, is_timeseries = False).var_names), "Gene names do not match between train and test data."
            # Ensure that both datasets have timeseries info using recursion
            check_perturbation_dataset(ad=load_perturbation(dataset_name, is_timeseries = True), is_timeseries = True, is_perturbation = False)
            check_perturbation_dataset(ad=load_perturbation(dataset_name, is_timeseries = False), is_timeseries = True, is_perturbation = True)
        except FileNotFoundError:
            # It's allowed to have only perturbation data. If only test data is found, check its validity.
            check_perturbation_dataset(ad=load_perturbation(dataset_name, is_timeseries = False), is_timeseries = False, is_perturbation = True)
        return
    
    # We will later select a variable number of genes based on this ranking. 
    print("Checking gene metadata...", flush = True)
    # Check gene metadata for ranking and ensure all necessary columns are present.
    assert "highly_variable_rank" in set(ad.var.columns), "Genes must be ranked in .var['highly_variable_rank']"
    assert all(~ad.var["highly_variable_rank"].isnull()), "Gene rankings should not be missing for any genes."
    assert all(ad.var["highly_variable_rank"]>=0), "Gene rankings must be positive integers"

    # Time   
    if is_timeseries: # Additional checks if the dataset is a time series
        print("Checking celltype and timepoint labels...", flush = True)
        assert "timepoint" in set(ad.obs.columns), "Time-series data must have a numeric 'timepoint' column"
        assert "cell_type" in set(ad.obs.columns), "Time-series data must have a string 'cell_type' column"

    # Names of genes perturbed
    print("Checking perturbation labels...", flush = True)
    # Validate perturbation labels and their corresponding expression levels
    assert "perturbation" in set(ad.obs.columns), "No 'perturbation' column"
    
    # Level of those genes after perturbation
    assert "expression_level_after_perturbation" in set(ad.obs.columns), "No 'expression_level_after_perturbation' column"
    iter = 0
    for i in ad.obs.index: # Iterating through the index of observations in the AnnData object
        iter = iter + 1
        p = ad.obs.loc[i, "perturbation"] # Retrieve perturbation details for the current observation
        elap = ad.obs.loc[i, "expression_level_after_perturbation"] # Retrieve expression levels after perturbation
        n_levels = len(str(elap).split(",")) # Count the number of expression levels reported
        n_perts =  len(str(p   ).split(",")) # Count the number of perturbations applied
        # Check that the number of perturbations matches the number of reported expression levels
        assert n_levels==n_perts, f"Too many or too few expression_level_after_perturbation entries in sample {i}: {p} has {n_perts} and {elap} has {n_levels}"
        # Further validate expression levels for non-knockout perturbations if full check is enabled or within the first 1000 iterations.
        if (ad.obs.loc[i, "perturbation_type"] != "knockout") and (do_full or iter < 1000):
            for x,g in zip(str(elap).split(","), str(p   ).split(",")):
                if g in ad.var_names:
                    # Check that the reported post-perturbation expression closely matches the data in the matrix
                    assert np.abs(float(x) - float(ad[i,g].X[0,0])) < 0.0001, f"For observation {i}, post-perturbation expression is given in .obs as {x} but the value in .X is {ad[i,g].X[0,0]}."

    # Check if the boolean control status ("is_control") is correctly labeled
    print("Checking control labels...", flush = True)
    assert "is_control"   in set(ad.obs.columns), "No 'is_control' column" # Ensure 'is_control' column exists
    assert bool==ad.obs["is_control"].dtype, "non-boolean 'is_control' column" # Ensure it's a boolean type
    assert       ad.obs["is_control"].any(), "no controls found" # Ensure there is at least one control sample

    # Validate perturbation types
    if is_perturbation:
        # Overexpression / knockout / knockdown
        assert "perturbation_type" in set(ad.obs.columns), "No 'perturbation_type' column"    
        assert all(
            [pt in {"overexpression", "knockout", "knockdown"} 
            for pt in ad.obs["perturbation_type"]]
        ),  "Invalid 'perturbation_type' column"

        assert not ad.obs["is_control"].all(), "only controls found in test data"

        # if it says it's (not) measured, make sure it's (not) measured.
        print("Checking which genes are measured...", flush = True) # Check gene measurement status
        assert all( [    g in ad.var_names for g in ad.uns["perturbed_and_measured_genes"]] ),     "perturbed_and_measured_genes"    " not all measured"
        assert all( [not g in ad.var_names for g in ad.uns["perturbed_but_not_measured_genes"]] ), "perturbed_and_not_measured_genes sometimes measured"
    
        # If it says it's perturbed, make sure it's perturbed. 
        has_multiple_genes_hit = "perturbations_overlap" in ad.uns.keys() and ad.uns["perturbations_overlap"]
        if has_multiple_genes_hit:
            all_genes_hit = set.union(*[set(p.split(",")) for p in ad.obs["perturbation"]])      
        else:
            all_genes_hit = set(ad.obs["perturbation"]) 
        assert all( [g     in all_genes_hit for g in ad.uns["perturbed_and_measured_genes"]] ),     "perturbed_and_measured_genes"  " not perturbed"
        assert all( [g     in all_genes_hit for g in ad.uns["perturbed_but_not_measured_genes"]] ), "perturbed_and_not_measured_genes not perturbed"
    
    # Expression in `.X` should be normalized and natural-log-transformed. 
    print("Checking for log-transform and raw data...", flush = True)
    if "skip_log_check" in ad.uns.keys() and ad.uns["skip_log_check"]:
        pass
    else:
        assert ad.X.max() < 15, "Expression values too big -- did you log them?" #exp(15) is about 3 million -- too big to be a transcript count.
            
    # Raw data should be present in `raw`.
    assert ad.raw is not None, "raw data are missing"
    print("... done.")
    return True # Indicate that all checks passed
