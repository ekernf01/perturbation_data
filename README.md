
This is a collection of uniformly formatted perturbation datasets, accompanied by the code used to acquire and clean the data. This part of our [benchmarking project](https://github.com/ekernf01/perturbation_benchmarking).

### Usage

For a Python API, consult the [companion package](https://github.com/ekernf01/load_perturbations). There is no R API but you could probably use this collection from R; see format details below.

### Installation

The networks themselves are too big to put on GitHub. But they are on Patrick's AWS at `s3://cahanlab/eric.kernfeld/eric_laptop/research/projects/perturbation_prediction/cell_type_knowledge_transfer/perturbation_data/`. 

Recommended installation: clone this repo, then find the actual datasets on AWS and place them manually in the `perturbations` directory. 

Eventually, once it's public: probably a zenodo link.

### About the datasets 

The two main commonalities among these data: they all measure the transcriptome, and every test dataset includes measurements after a genetic knockout or an overexpression experiment. The main differences among these datasets are:

- What lab or project did they come from?
- What organism are they in?
- Do they have time-series labels?
- Do they have splicing-based RNA velocity data?
- Are they bulk RNA or single-cell?

Metadata answering those questions are stored in `perturbations/perturbations.csv`. 

### Format details 

Each network is stored as a pair of [AnnData](https://anndata.readthedocs.io/en/latest/index.html) objects in `train.h5ad` and `test.h5ad`. Some may lack separate training data, having only `test.h5ad`. You will need to create your own train/test split by splitting the test data. When there is a separate training dataset, it will usually come from the same biological system or set of cell types as the test data, but it will lack perturbations and it may have more interesting temporal structure than the test data. 

### Adding new datasets and setting clear expectations

Every AnnData object in the collection conforms to certain expectations. This list is in progress and this README may be out of date, but you can use at the function `load_perturbations.check_perturbation_dataset()` in the loader package for authoritative details. To add new datasets or alter ingestion of a current dataset, look at the notebooks in `setup.py` for examples, and ensure that the result ultimately passes the assertions done by `check_perturbation_dataset()`. Here are some key requirements.

- Metadata columns `timepoint` and `cell_type` for time-series datasets (required by [PRESCIENT](https://cgs.csail.mit.edu/prescient/file_formats/))
- A column `"perturbation"` in `.obs`, dtype `str`, indicating what gene or genes are perturbed (e.g. `"FOXN1"` or `"FOXN1,PAX9"`).
- A boolean column in `.obs` called `"is_control"`, with the obvious meaning
- An iterable `"perturbed_but_not_measured_genes"` in `uns`, containing all genes that are perturbed but not measured.
- An iterable `"perturbed_and_measured_genes"` in `uns`, containing all genes that are perturbed and also measured.
- Expression in `.X` should be normalized and log-transformed. 
- Raw data should be present in `raw`.
 
