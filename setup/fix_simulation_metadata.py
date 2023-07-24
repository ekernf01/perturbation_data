import os
import scanpy as sc

# Fix broken metadata in simulated test sets
parent_folder = "../perturbations"
subfolders = os.listdir(parent_folder)
simulation_folders = [folder for folder in subfolders if folder.startswith("simulation_TrueNetwork")]
for foldername in simulation_folders:
    h5ad_file_path = os.path.join(parent_folder, foldername, "oldtest.h5ad")
    if os.path.exists(h5ad_file_path):
        print(".", end = "", flush = True)
        adata = sc.read_h5ad(h5ad_file_path)
        adata.obs["is_control"] = adata.obs["perturbation"]=="control"
        adata.obs["expression_level_after_perturbation"] = 0.1
        adata.obs["is_treatment"] = adata.obs["perturbation"]!="control"
        adata.obs["is_control_int"] = [1 if c else 0 for c in adata.obs["perturbation"]=="control"]
        not_control = ~adata.obs["is_control"]
        adata.obs["perturbation"] = adata.obs["perturbation"].astype(str)
        adata.obs.loc[not_control, "perturbation"] = [
                adata.var_names[int(g)] 
                for g in adata.obs.loc[not_control, "perturbation"]
        ]
        perturbed_genes = set(list(adata.obs['perturbation'].unique())).difference({"control"})
        perturbed_and_measured_genes     = perturbed_genes.intersection(adata.var.index)
        perturbed_but_not_measured_genes = perturbed_genes.  difference(adata.var.index)
        adata.uns["perturbed_and_measured_genes"]     = list(perturbed_and_measured_genes)
        adata.uns["perturbed_but_not_measured_genes"] = list(perturbed_but_not_measured_genes)
        adata.write_h5ad(os.path.join(parent_folder, foldername, "test.h5ad"))

    else:
        print(f"Error: h5ad file not found in folder: {foldername}")
