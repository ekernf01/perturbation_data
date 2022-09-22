### Environments

Currently, python code is run from a conda environment called `cell_type_grn_transfer`. See the benchmarking folder for more details. R code does not yet have a controlled environment.

### Using the perturbations

This is a collection of uniformly formatted perturbation datasets. We offer R and Python code to quickly read and write from this collection. **This doesn't work yet but I hope to set it up soon!**

In R:

```
source("R/load_perturbations.R")
# Set this to point to the "perturbations" folder adjacent to this README. 
options(PERTURBATION_PATH = "perturbations")
# What datasets are available?
View(load_perturbation_metadata())
# Grab one
nakatake_et_al = load_perturbation("psc_overexpression") 
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
nakatake_et_al = load_perturbations.load_perturbation("psc_overexpression") 
```

### Layout

- `load_perturbations`: Code to access these networks
- `perturbations`: perturbations stored in h5ad (AnnData) format
- `not_ready`: Datasets that we may in the future process into regular format
- `setup`: code and documentation describing how we assembled and formatted this collection.

### About the datasets 

The main commonality among these data: they all measure the transcriptome, and every test dataset includes measurements after a genetic knockout or an overexpression experiment. The main differences among these datasets are:

- What lab or project did they come from?
- What organism are they in?
- Do they have time-series labels?
- Do they have splicing-based RNA velocity data?
- Are they bulk RNA or single-cell?

Metadata answering those questions are stored in `perturbations/perturbations.csv`. 

### Format details 

Each network is stored as a pair of [AnnData](https://anndata.readthedocs.io/en/latest/index.html) objects in `train.h5ad` and `test.h5ad`. Some may lack separate training data, having only `test.h5ad`. You will need to create your own train/test split by splitting the test data. When there is a separate training dataset, it will usually come from the same biological system or set of cell types as the test data, but it will lack perturbations and it may have more interesting temporal structure than the test data. 

Every AnnData object contains:

- a column "perturbation" in `.obs`, indicating what gene or signaling pathway is perturbed.
- The column "perturbation" must contain at least one entry "Control" indicating the control samples. OR maybe I will require a binary column in `.obs` containing "is_control". This is undecided.
- A set "perturbed_but_not_measured_genes" in `uns`, containing all genes or pathways that are perturbed but not measured.
- **???? This list of expectations is in progress.**
- Expression in `.X` is expected to be on a log scale. 

