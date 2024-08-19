
This is a collection of uniformly formatted perturbation datasets, accompanied by the code used to acquire and clean the data. This part of our [benchmarking project](https://github.com/ekernf01/perturbation_benchmarking). These data all measure the transcriptome, and every test dataset includes measurements after a genetic knockout or knockdown or overexpression experiment. These datasets are all from human cells. The main differences among these datasets are the cell type, data type (e.g. microarray vs scRNA-seq), and scale (dozens to thousands of perturbations). 

### Installation and usage

If you are seeing this on Github: the expression data are too big to put on GitHub, but they are on Zenodo (DOI: 10.5281/zenodo.10363068). 

If you are seeing this on Zenodo, you can download the data to any permanent location and access it via our [data loader package](https://github.com/ekernf01/pereggrn_perturbations). 

### Format details 

Metadata listing data sources, number of perturbations, and other summaries are stored in `perturbations/perturbations.csv`. 

Each network is stored as a pair of [AnnData](https://anndata.readthedocs.io/en/latest/index.html) objects in `train.h5ad` and `test.h5ad`. Some may lack separate training data, having only `test.h5ad`. When there is a separate training dataset, it will usually come from the same biological system or set of cell types as the test data, but it will lack perturbations and it will have time-series measurements. Some datasets also have a `screen.csv`, which summarizes gross phenotypic outcomes across many genes.

### Preprocessing

The code we used to preprocess the data is in a subfolder `setup`. It is missing from our initial Zenodo upload, but it is available on Github and we intend to include it in future Zenodo data releases. It relies on data in a subfolder `not_ready`, which we have omitted from Zenodo in order to reduce the download size. If you want to repeat or alter preprocessing of a dataset, all data sources are listed in `perturbations/perturbations.csv`. 

### Adding new datasets

Every AnnData object in the collection conforms to certain expectations. To add new datasets or alter ingestion of a current dataset, you must:

- add and fill a row starting with your dataset's name in `perturbations.csv`.
- save an AnnData object in `perturbation_data/perturbations/<dataset_name>/test.h5ad`.
- ensure that the result passes the assertions done by `pereggrn_perturbations.check_perturbation_dataset()`.

This README may be out of date (the checker function is authoritative), but here are some key requirements current as of 2024 Aug 16.

- There must be a file `test.h5ad` containing perturbation transcriptomics data, and optionally a file `train.h5ad` containing time-series data from the same system. Another optional file is `screen.csv` containing gross phenotypic outcomes (not the full transcriptome) of genetic perturbations.
- If `screen.csv` is present, then reading it via `pd.read_csv()` must yield a dataframe with a column `perturbation` containing the names of the genes perturbed. 
- Each h5ad must have a column `"perturbation"` in `.obs` with dtype `str`, indicating what gene or genes are perturbed (e.g. `"FOXN1"` or `"FOXN1,PAX9"`).
- Each must have a boolean column in `.obs` called `"is_control"`, with the obvious meaning.
- Each must have a column `gene_rank` in `.var` containing a positive integer rank for each gene (lower values are more interesting; we might select e.g. the bottom 1,000 genes). 
- Each must have an iterable `"perturbed_but_not_measured_genes"` in `uns`, containing names of all genes that are perturbed but not measured.
- Each must have an iterable `"perturbed_and_measured_genes"` in `uns`, containing all genes that are perturbed and also measured.
- Expression in `.X` should be normalized and log-transformed. 
- Raw data should be present in `raw`. This is currently only needed for GeneFormer.
- `train` and `test` must have the same genes and the same `gene_rank`.
 
Aside from these format requirements, we also analyze perturbation effectiveness, correlation across replicates, normalization, typical scRNA QC (e.g. empty droplets), and possible confounders (donor, batch, cell cycle, treatment) with a special focus on heterogeneity among control samples. We sometimes regress out some variation or remove some samples/cells. You can look at the notebooks or scripts in `setup` for examples. 