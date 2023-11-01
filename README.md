
This is a collection of uniformly formatted perturbation datasets, accompanied by the code used to acquire and clean the data. This part of our [benchmarking project](https://github.com/ekernf01/perturbation_benchmarking).

### Installation and usage

The expression data themselves are too big to put on GitHub, but they are on Zenodo (DOI: 10.5281/zenodo.8071809). Simply download them to any permanent location and point our [data loader package](https://github.com/ekernf01/load_perturbations) to the "perturbations" subfolder using `load_perturbations.set_data_path("path/to/perturbation_data/perturbations")`. There is no R API but you could probably use this collection from R without too much hassle; see format details below.

### About the datasets 

The two main commonalities among these data: they all measure the transcriptome, and every test dataset includes measurements after a genetic knockout or an overexpression experiment. The main differences among these datasets are:

- What lab or project did they come from?
- What organism and cell type are they in?
- How many perturbations are covered?

Metadata answering those questions are stored in `perturbations/perturbations.csv`. 

### Format details 

Each network is stored as a pair of [AnnData](https://anndata.readthedocs.io/en/latest/index.html) objects in `train.h5ad` and `test.h5ad`. Some may lack separate training data, having only `test.h5ad`. When there is a separate training dataset, it will usually come from the same biological system or set of cell types as the test data, but it will lack perturbations and it may have more interesting temporal structure than the test data. 

### Adding new datasets and setting clear expectations

Every AnnData object in the collection conforms to certain expectations. To add new datasets or alter ingestion of a current dataset, you must:

- add a row starting with your dataset's name in `perturbations.csv`.
- save an AnnData object in `perturbation_data/perturbations/<dataset_name>/test.h5ad`.
- ensure that the result passes the assertions done by `load_perturbations.check_perturbation_dataset()`.

You can look at the notebooks in `setup` for examples. This README may be out of date, but here are some key requirements.

- A file `test.h5ad` containing perturbation transcriptomics data, and optionally a file `train.h5ad` containing time-series data from the same system
- Each h5ad must have a column `"perturbation"` in `.obs` with dtype `str`, indicating what gene or genes are perturbed (e.g. `"FOXN1"` or `"FOXN1,PAX9"`).
- Each must have a boolean column in `.obs` called `"is_control"`, with the obvious meaning
- Each must have a column `gene_rank` in `.var` containing a positive integer rank for each gene (lower values are more interesting). 
- Each must have an iterable `"perturbed_but_not_measured_genes"` in `uns`, containing names of all genes that are perturbed but not measured.
- Each must have an iterable `"perturbed_and_measured_genes"` in `uns`, containing all genes that are perturbed and also measured.
- Expression in `.X` should be normalized and log-transformed. 
- Raw data should be present in `raw`. This is currently only needed for GeneFormer.
- `train` and `test` must have the same genes and the same ranking.
 
