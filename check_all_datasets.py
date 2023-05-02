PROJECT_PATH = '/home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/'
import os
import load_perturbations
os.environ["PERTURBATION_PATH"]  =  "perturbations"
# Assume we run from Eric's whole-project-all-repos directory or location of this script.
try:
    os.chdir("perturbation_data")
except FileNotFoundError:
    pass
for dataset_name in load_perturbations.load_perturbation_metadata().query("is_ready=='yes'")["name"][::-1]:
    print("Checking " + dataset_name)
    try:
        load_perturbations.check_perturbation_dataset(dataset_name)
    except Exception as e:
        print(repr(e))
