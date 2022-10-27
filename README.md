### Using the perturbations

This is a collection of uniformly formatted perturbation datasets. We offer R and Python code to quickly read and write from this collection. **The R code doesn't exist yet but the Python does.** 

### Installation

This collection is not yet set up for deployment to non-Eric users. Main obstacles: 

- The Python code is not pip-installable or conda-installable. But it's in this repo, and you can point `sys.path.append` to it. (see example usage below.)
- The networks themselves are too big to put on GitHub. But they are on Patrick's AWS at `s3://cahanlab/eric.kernfeld/eric_laptop/research/projects/perturbation_prediction/cell_type_knowledge_transfer/perturbation_data/`. 

Recommended installation: clone this repo, then find the actual datasets on AWS and place them manually in the `perturbations` directory. 

### Environments

Currently, python code is run from a conda environment called `cell_type_grn_transfer`. See the benchmarking folder for more details. R code does not yet have a controlled environment.

In R:

```
source("R/load_perturbations.R")
# Set this to point to the "perturbations" folder adjacent to this README. 
options(PERTURBATION_PATH = "perturbations")
# What datasets are available?
View(load_perturbation_metadata())
# Grab one
nakatake_et_al = load_perturbation("nakatake") 
```

In Python:

```
# Set this to point to the "load_perturbations" folder inside the "perturbations" folder adjacent to this README. 
sys.path.append('path/to/load_perturbations/') 
import load_perturbations
# Set this to point to the "perturbations" folder adjacent to this README. 
os.environ["PERTURBATION_PATH"] = "perturbations"
# What datasets are available?
load_perturbations.load_perturbation_metadata()
# Grab one
nakatake_et_al = load_perturbations.load_perturbation("nakatake") 
```

### Layout

- `load_perturbations`: Code to access these networks
- `perturbations`: perturbations stored in h5ad (AnnData) format
- `not_ready`: Datasets that we may in the future process into regular format
- `setup`: code and documentation describing how we assembled and formatted this collection.

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

Every AnnData object conforms to certain expectations. This list is in progress and the README may be out of date; look at the function `check_perturbation_dataset` in `setup/ingestion.py` for authoritative details.

- A column `"perturbation"` in `.obs`, dtype `str`, indicating what gene or geness are perturbed (e.g. `"FOXN1"` or `"FOXN1,PAX9"`).
- A boolean column in `.obs` called `"is_control"`, with the obvious meaning
- An iterable `"perturbed_but_not_measured_genes"` in `uns`, containing all genes or pathways that are perturbed but not measured.
- An iterable `"perturbed_and_measured_genes"` in `uns`, containing all genes or pathways that are perturbed and also measured.
- Expression in `.X` should be normalized and log-transformed. 
- Raw data should be present in `raw`.
 
