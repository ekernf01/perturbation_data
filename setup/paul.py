# ### Mouse hematopoiesis 
# 
# This notebook ingests a dataset from a paper about myeloid progenitor transcriptional heterogeneity ([Paul et al 2015](https://pubmed.ncbi.nlm.nih.gov/26627738/)). We reformat the data, conduct exploratory analysis, and annotate cell types. 

# Get required libraries
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pereggrn_perturbations as dc # the data checker (dc)
import scipy.sparse
import re
import os
from scipy.stats import rankdata as rank
from sklearn.neighbors import KNeighborsRegressor
import ggrn.api as ggrn

# Load the main dataframe
df = pd.read_csv('../not_ready/paul/GSE72857_umitab.txt', sep='\t')

# Load the experimental design table
exp_design = pd.read_csv('../not_ready/paul/GSE72857_experimental_design.txt', sep='\t', skiprows=19)

# ##### Isolating the Wildtype and Perturbations
# 
# The data is currently stored in two files:
# - GSE72857_umitab.txt which has the genes and sample/cell names based on well ID.
# - GSE72857_experimental_design which has metadata about each sample (based on well ID).
# 
# Our next step will be merging the main dataframe with the experimental design table using the well IDs to have the data for wildtype and perturbations.

# Transpose the main dataframe for merging
df_t = df.T
df_t.index.name = 'Well_ID'

# Merge the main dataframe with the experimental design table using the well IDs
# merged_df = df_t.merge(exp_design[['Well_ID', 'Batch_desc']], left_index=True, right_on='Well_ID', how='left')
merged_df = df_t.merge(exp_design, left_index=True, right_on='Well_ID', how='left')
merged_df.set_index('Well_ID', inplace=True)

# ##### Creating the AnnData Structures
# 
# Before transposing the matrix, the data appears to have gene names as row indices and sample/cell names as column headers.
# The values represent expression levels for each gene in each sample/cell.
# 
# Do the following to convert the txt file to h5ad and add the necessary metadata:

# Extract gene names and expression data
merged_gene_columns = [col for col in merged_df.select_dtypes(include=[np.number]).columns if col not in exp_design]
merged_numeric_data = merged_df[merged_gene_columns]
merged_gene_names = merged_numeric_data.columns.values
merged_cell_names = merged_numeric_data.index.values

# Create AnnData object
adata_merged = sc.AnnData(X=merged_numeric_data.values.astype(float))
adata_merged.X = scipy.sparse.csr_matrix(adata_merged.X)
adata_merged.X = adata_merged.X.tocsr()
adata_merged.var_names = merged_gene_names
adata_merged.obs_names = merged_cell_names

# Add metadata to obs
adata_merged.obs['Batch_desc'] = merged_df['Batch_desc'].values
adata_merged.obs['Seq_batch_ID'] = merged_df['Seq_batch_ID'].values
adata_merged.obs['Amp_batch_ID'] = merged_df['Amp_batch_ID'].values
adata_merged.obs['well_coordinates'] = merged_df['well_coordinates'].values
adata_merged.obs['Mouse_ID'] = merged_df['Mouse_ID'].values
adata_merged.obs['Plate_ID'] = merged_df['Plate_ID'].values
adata_merged.obs['Pool_barcode'] = merged_df['Pool_barcode'].values
adata_merged.obs['Cell_barcode'] = merged_df['Cell_barcode'].values
adata_merged.obs['Number_of_cells'] = merged_df['Number_of_cells'].values
adata_merged.obs['CD34_measurement'] = merged_df['CD34_measurement'].values
adata_merged.obs['FcgR3_measurement'] = merged_df['FcgR3_measurement'].values
adata_merged.obs["well_row"]    = adata_merged.obs["well_coordinates"].str.extract(r"([A-Z])\d+")[0]
adata_merged.obs["well_column"] = adata_merged.obs["well_coordinates"].str.extract(r"([0-9]+)")[0]

# #### Labeling by Cell Type
# 
# To label by cell types, we'll have to process the data, perform PCA & clustering, and label based on the clusters and our knowledge of the sequence and relations of cell types in human hematopoiesis.

# ##### Exclude cells with RNA count <1000, then normalize and apply log transformation
# Calculate the total counts per cell
adata_merged.obs['total_counts'] = adata_merged.X.sum(axis=1)
adata_merged.obs['log10_total_counts'] = np.log10(adata_merged.obs['total_counts'])
# Filter out cells with fewer than 1,000 RNA counts
print(f"Total cells before filtering: {adata_merged.n_obs}")
adata_merged = adata_merged[adata_merged.obs['total_counts'] >= 1000, :].copy()

# Verify filtering step
print(f"Total cells after filtering: {adata_merged.n_obs}")

# Normalize each cell by the total counts, then multiply by a scaling factor (e.g., 10,000)
adata_merged.X = adata_merged.X / adata_merged.obs['total_counts'].values[:, None] * 10000

# Perform log transformation
adata_merged.X = adata_merged.X.tocsr()
adata_merged.raw = adata_merged.copy()
adata_merged.X = np.log1p(adata_merged.X)  # This is equivalent to np.log(adata_merged.X + 1)
adata_merged.X = adata_merged.X.tocsr()

# Verify normalization and log transformation
print(f"Data after normalization and log transformation (top left corner):\n{adata_merged.X.A[0:5, 0:5]}")

# Check the dimensions of the filtered data
print(f"adata_filtered shape: {adata_merged.shape}")

# Check for NaN values
if np.any(np.isnan(adata_merged.X.A)):
    print("NaN values found in the data.")
else:
    print("No NaN values found in the data.")


# We run a typical exploratory analysis: variable gene selection, PCA, nearest-neighbors, diffusion maps, and modularity-minimizing graph clustering.

sc.pp.highly_variable_genes(adata_merged, n_bins=50, n_top_genes = adata_merged.var.shape[0], flavor = "seurat_v3" )
sc.tl.pca(adata_merged, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata_merged, n_neighbors=4, n_pcs=20, )
sc.tl.umap(adata_merged)
sc.tl.diffmap(adata_merged)
sc.tl.louvain(adata_merged, resolution=0.8)
S_genes_hum = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", 
            "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "CENPU", "HELLS", "RFC2", 
            "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", 
            "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", 
            "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_genes_hum = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
             "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "PIMREG", 
             "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", 
             "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "JPT1", "CDC20", "TTK",
             "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", 
             "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", 
             "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]
sc.tl.score_genes_cell_cycle(
    adata_merged, 
    s_genes   = [g.title() for g in S_genes_hum if g.title() in adata_merged.var_names], 
    g2m_genes = [g.title() for g in G2M_genes_hum if g.title() in adata_merged.var_names]
)
adata_merged.write_h5ad('../not_ready/paul/paul_clustered_but_not_annotated.h5ad')

# PAGA graph construction
sc.tl.paga(adata_merged, groups='louvain')
plt.rcParams["figure.figsize"] = [6, 4.5]
sc.pl.paga(adata_merged, show=False)
sc.tl.draw_graph(adata_merged, init_pos='paga', random_state=123)

# Rank genes based on the high variability information provided by Scanpy
adata_merged.var['highly_variable_rank'] = np.argsort(~adata_merged.var['highly_variable'].values)

# Is control: Infer from 'Batch_desc'
adata_merged.obs['is_control'] = adata_merged.obs['Batch_desc'].apply(lambda x: False if 'KO' in str(x) else True)

# Perturbation: Use 'Batch_desc' to determine perturbations
adata_merged.obs['perturbation'] = adata_merged.obs['Batch_desc'].apply(
    lambda x: 'Cebpa' if 'Cebpa KO' in x else ('Cebpe' if 'Cebpe KO' in x else ('Cebpe control' if 'Cebpe control' in x else ('Cebpa control' if 'Cebpa control' in x else 'None')))
)

# Expression level after perturbation: Set to NaN if control or 0 if not control
adata_merged.obs['expression_level_after_perturbation'] = adata_merged.obs['is_control'].apply(
    lambda x: np.nan if x else 0
).values

# Perturbation type: Always knockout since this is the only possible perturbation in this data
adata_merged.obs['perturbation_type'] = adata_merged.obs['Batch_desc'].apply(
    lambda x: 'knockout'
)

print(adata_merged.obs[["is_control", "perturbation", "Batch_desc"]].value_counts())

# Define perturbed and measured genes
perturbed_but_not_measured_genes = []

# Store these lists in the uns attribute
adata_merged.uns['perturbed_but_not_measured_genes'] = perturbed_but_not_measured_genes

plt.rcParams["figure.figsize"] = [12, 8]
markers = {
    "Erythroids": ["Gata1", "Gfi1b", "Car1", "Car2", "Klf1", "Zfpm1", "Cpox", "Beta-s", "Hbb-b1", "Hba-a2"],
    "Megakaryocytes": ["Pf4", "Itga2b", "Pbx1", "Mef2c", "Fli1"],
    "Granulocytes": ["Elane", "Prtn3", "Mpo", "Cebpe", "Csf1r", "Cebpa", "Lgals1"],
    "Eosinophil": ["Prg2"],
    "Basophil": ["Prss34", "Lmo4"],
    "Monocytes": ["Flt3", "Ifitm1", "Lmo4", "Elane", "Prtn3", "Mpo", "Cebpe", "Csf1r", "Cebpa", "Lgals1"],
    "DC": ["Irf8", "Id2", "Cd74", "H2-Aa"],
    "Lymphoid": ["Gzmb", "Tcb", "Ccl5"], 
    "other_variables": ["log10_total_counts", "phase", "perturbation", "Batch_desc", "cell_type", "timepoint", "dpt_pseudotime"],
}

for cell_type, genes in markers.items():
    present_genes = []
    for gene in genes:
        pattern = re.compile(rf"^{re.escape(gene)}(;.*)?$")
        present_genes.extend([g for g in adata_merged.var_names if pattern.match(g)])
        present_genes.extend([g for g in adata_merged.obs.columns if pattern.match(g)])
    if present_genes:
        print(f"Marker genes for {cell_type} present in the dataset: {present_genes}")
        # Calculate the number of rows needed for 2 plots per row
        n_cols = 2
        n_rows = (len(present_genes) + 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        fig.suptitle(f"{cell_type} Marker Genes", fontsize=16)
        axes = axes.flatten()  # Flatten the axes array for easy iteration
        
        for ax, gene in zip(axes, present_genes):
            sc.pl.draw_graph(adata_merged, color=[gene], use_raw=False, ax=ax, show=False)
            ax.set_title(gene)
        
        # Remove any empty subplots
        for i in range(len(present_genes), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
        plt.savefig(f"../not_ready/paul/paul_{cell_type}.pdf")
    else:
        print(f"No marker genes found for {cell_type} in the dataset.")

plt.figure()
summary_plot = sc.pl.draw_graph(adata_merged, color="Batch_desc", use_raw=False, show=False)
summary_plot.figure.savefig('../not_ready/paul/paul_summary.svg', bbox_inches='tight')
adata_merged.obs["summary"] = "timeseries"
is_knockout_or_matched_control = adata_merged.obs["Batch_desc"].str.contains("Cebp")
adata_merged.obs.loc[is_knockout_or_matched_control, "summary"] = adata_merged.obs.loc[is_knockout_or_matched_control, "Batch_desc"]
print(adata_merged.obs["summary"].value_counts())
summary_plot = sc.pl.draw_graph(
    adata_merged, 
    color="summary", 
    palette = {
        "timeseries": "#008000",  # green
        "Cebpa KO": "#FFA500",    # orange
        "Cebpa control": "#0000FF",  # blue
        "Cebpe KO": "#FFC0CB",    # pink
        "Cebpe control": "#800080",  # purple
    },
    use_raw=False, 
    show=False
)
summary_plot.figure.savefig('../not_ready/paul/paul_summary_simple.svg', bbox_inches='tight')

# Make cluster anottation dictionary - CHANGE based on observations
annotation_filtered = {
    "MEP":[17, 2, 14],
    "Erythroids":[9, 5, 18, 3, 13, 20],  
    "GMP":[16],
    "late_GMP" :[10, 4, 12],   
    "Megakaryocytes":[11, 17],
    "Granulocytes":[6, 0, 15],
    "Monocytes": [8, 1, 19],
    "DC":[7]
}

# Reverse the dictionary format to map cluster to cell types
annotation_filtered_rev = {}
for cell_type, clusters in annotation_filtered.items():
    for cluster in clusters:
        annotation_filtered_rev[str(cluster)] = cell_type
        
annotation_filtered_rev

adata_merged.obs["cell_type"] = [annotation_filtered_rev.get(i, 'NaN') for i in adata_merged.obs.louvain]

# pseudotime inference
adata_train = adata_merged[adata_merged.obs['summary'] == 'timeseries', :].copy()

adata_train.uns['iroot'] = np.flatnonzero(adata_train.obs['cell_type'].isin({'MEP', 'GMP'}))[0]
sc.tl.dpt(adata_train, n_dcs=5)
adata_train = adata_train[np.isfinite(adata_train.obs["dpt_pseudotime"]), :].copy()
projector = KNeighborsRegressor(n_neighbors=10).fit(y = adata_train.obs['dpt_pseudotime'], X = adata_train.obsm['X_pca'])
adata_merged.obs['dpt_pseudotime'] = projector.predict(adata_merged.obsm['X_pca'])
del adata_train

summary_plot = sc.pl.draw_graph(
    adata_merged, 
    color="dpt_pseudotime", 
    use_raw=False, 
    show=False
)
summary_plot.figure.savefig('../not_ready/paul/paul_dpt.svg', bbox_inches='tight')


def max_to_1_but_min_the_same(x):
    return x.min() + (1-x.min())*(rank(x)/rank(x).max())

terminal_states = [
    "Monocytes", "Erythroids", "Megakaryocytes", "Granulocytes", "DC"
]
for ct in terminal_states:
    is_in_ct = adata_merged.obs["cell_type"] == ct
    adata_merged.obs.loc[is_in_ct, "dpt_pseudotime"] = max_to_1_but_min_the_same(adata_merged.obs.loc[is_in_ct, "dpt_pseudotime"])
    
summary_plot = sc.pl.draw_graph(
    adata_merged, 
    color="dpt_pseudotime", 
    use_raw=False, 
    show=False
)
summary_plot.figure.savefig('../not_ready/paul/paul_dpt_rescaled.svg', bbox_inches='tight')

# We are in an awkward situation where downstream code expects known discrete time labels, not pseudotemporal ordering of a single-timepoint measurement of an asynchronous process.
# Therefore, discretize pseudotime.
adata_merged.obs['timepoint'] = [int(t) for t in (adata_merged.obs['dpt_pseudotime']/5).round(1)*10]
summary_plot = sc.pl.draw_graph(
    adata_merged,
    color="timepoint", 
    use_raw=False, 
    show=False
)
summary_plot.figure.savefig('../not_ready/paul/timepoint.svg', bbox_inches='tight')


# Does this play nicely with GGRN's OT matching scheme?
# Downstream code in GGRN can use this by setting matching_method="optimal_transport".
matched = ggrn.match_timeseries(adata_merged[adata_merged.obs['summary']=='timeseries', :], 
                                matching_method="optimal_transport", 
                                matched_control_is_integer=False, 
                                do_look_backwards = True)
has_matched = matched.obs["matched_control"].notnull()
matched.obs["matched_past_celltype"] = ""
matched.obs.loc[has_matched, "matched_past_celltype"] = matched.obs.loc[matched.obs.loc[has_matched, "matched_control"], "cell_type"].values
# Result is ok but still contains some unexpected lineage relationships.
print("Preview of optimal transport-based matching:")
print(matched[matched.obs["matched_past_celltype"]!=matched.obs["cell_type"], :].obs.value_counts(["matched_past_celltype", "cell_type"]))

# A more customized solution based on nearest neighbors and pseudotime.
# Downstream code in GGRN can use this by setting matching_method="user".
def do_pseudotime_matching(adata):
    adata.obs["matched_control"] = np.nan
    def pick_lowest_pseudotime(indices):
        return indices[np.argmin(adata.obs["dpt_pseudotime"].iloc[indices])]
    for i, cellname in enumerate(adata.obs_names):
        try:
            adata.obs.loc[cellname, "matched_control"] = adata.obs_names[pick_lowest_pseudotime(adata.obsp["connectivities"][i,:].nonzero()[1])]
        except ValueError: #no neighbors
            pass
    return adata

matched = do_pseudotime_matching(adata_merged[adata_merged.obs['summary']=='timeseries', :])
has_matched = matched.obs.index[matched.obs["matched_control"].notnull()].values
matched.obs["matched_past_celltype"] = ""
matched.obs.loc[has_matched, "matched_past_celltype"] = matched.obs.loc[matched.obs.loc[has_matched, "matched_control"], "cell_type"].values
print("Preview of neighborhood+pseudotime-based matching:")
print(matched[matched.obs["matched_past_celltype"]!=matched.obs["cell_type"], :].obs.value_counts(["matched_past_celltype", "cell_type"]))

# Final choice: the pseudotime matching is more in line with what we know about the data.
adata_merged.obs["matched_control"] = np.nan
adata_merged.obs.loc[has_matched, "matched_control"] = matched.obs.loc[has_matched, "matched_control"]

# Save the matched embeddings for visualization using the same R code from the evals
embeddings = pd.DataFrame(adata_merged.obsm["X_draw_graph_fa"], index = adata_merged.obs_names)
adata_merged.obs[["train_viz1", "train_viz2"]] = embeddings
adata_merged.obs[["progenitor_viz1", "progenitor_viz2"]] = np.nan
adata_merged.obs.loc[has_matched, ["progenitor_viz1", "progenitor_viz2"]] = embeddings.loc[adata_merged.obs.loc[has_matched, "matched_control"], :].values
adata_merged.obs[["matched_control", "cell_type", "timepoint", 
                  "train_viz1", "train_viz2", "progenitor_viz1", "progenitor_viz2"]].to_csv("../not_ready/paul/paul_matching_sanity_check.csv")


# Visualize results
celltype_plot = sc.pl.draw_graph(
    adata_merged, 
    color='cell_type', 
    palette = {
        "MEP": "#800080",          # purple
        "GMP": "#000000",          # black
        "late_GMP": "#A52A2A",     # brown
        "Monocytes": "#FFA500",    # orange
        "Erythroids": "#FF0000",   # red
        "DC": "#FFFF00",           # yellow
        "Megakaryocytes": "#0000FF", # blue
        "Granulocytes": "#008000", # green    
    }, 
    use_raw=False, 
    show=False,
)
celltype_plot.figure.savefig('../not_ready/paul/paul_cell_type.svg', bbox_inches='tight')

# Split and save the data into train (which contains the wildtype data) and test (which contains the perturbation data and paired controls).

# Create train and test anndata objects
# Filter the data based on Batch_desc
train_adata = adata_merged[~adata_merged.obs['Batch_desc'].str.contains('control|KO', na=False)].copy()
test_adata = adata_merged[adata_merged.obs['Batch_desc'].str.contains('control|KO', na=False)].copy()
print(test_adata.obs['Batch_desc'].value_counts())
cell_type_composition = test_adata.obs[['perturbation', 'cell_type', 'timepoint']].value_counts().reset_index().sort_values(["cell_type", "perturbation"])
cell_type_composition["cell_type_is_abundant"] = cell_type_composition["count"] > 25
print(cell_type_composition)
test_adata.obs = test_adata.obs.merge(cell_type_composition[["perturbation", "cell_type", "timepoint", "cell_type_is_abundant"]], on=["perturbation", "cell_type", "timepoint"], how="left")
test_adata = test_adata[test_adata.obs['cell_type_is_abundant']].copy()
train_adata.uns["perturbed_and_measured_genes"] = [] # No perturbation in the training data

# Since the CEPBA and CEBPE perturbations have separate genetic backgrounds
# and sorting strategies, they each need their own matched control, and 
# the easiest way to avoid mixing this up is to split the data into two. 
train_output_file_path = '../perturbations/paul1/train.h5ad'
os.makedirs(os.path.dirname(train_output_file_path), exist_ok=True)
train_adata.write_h5ad(train_output_file_path)
train_output_file_path = '../perturbations/paul2/train.h5ad'
os.makedirs(os.path.dirname(train_output_file_path), exist_ok=True)
train_adata.write_h5ad(train_output_file_path)

# Save test.h5ad
test1 = test_adata[test_adata.obs["perturbation"].str.contains("Cebpa"), :]
test1.uns['perturbed_and_measured_genes'] = ["Cebpa"]
test1.write_h5ad('../perturbations/paul1/test.h5ad')
test2 = test_adata[test_adata.obs["perturbation"].str.contains("Cebpe"), :]
test2.uns['perturbed_and_measured_genes'] = ["Cebpe"]
test2.write_h5ad('../perturbations/paul2/test.h5ad')


# Finally, ensure that the dataset passes all the required checks in the data checker.

# Set the path to the dataset
dc.set_data_path("../perturbations")

# Check the dataset using the data checker
is_valid = dc.check_perturbation_dataset("paul2")
is_valid = dc.check_perturbation_dataset("paul1")


