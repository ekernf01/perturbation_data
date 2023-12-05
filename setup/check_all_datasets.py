import os
import load_perturbations
load_perturbations.set_data_path("../perturbations")
for dataset_name in load_perturbations.load_perturbation_metadata().query("is_ready=='yes'")["name"][::-1]:
    print("Checking " + dataset_name)
    try:
        load_perturbations.check_perturbation_dataset(dataset_name)
    except Exception as e:
        print(repr(e))
