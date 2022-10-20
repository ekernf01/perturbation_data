PROJECT_PATH = '/home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/'
import importlib
import sys
import os
sys.path.append(os.path.expanduser(os.path.join(PROJECT_PATH, 'perturbations', 'load_perturbations'))) 
import load_perturbations
importlib.reload(load_perturbations)
os.environ["PERTURBATION_PATH"]  = PROJECT_PATH + "perturbations/perturbations"

for dataset_name in load_perturbations.load_perturbation_metadata().query("is_ready=='yes'")["name"]:
    print("Checking " + dataset_name)
    try:
        load_perturbations.check_perturbation_dataset(dataset_name)
    except AssertionError as e:
        print(repr(e))
