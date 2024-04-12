import os
import pereggrn_perturbations
pereggrn_perturbations.set_data_path("../perturbations")
for dataset_name in pereggrn_perturbations.load_perturbation_metadata().query("is_ready=='yes'")["name"][::-1]:
    print("Checking " + dataset_name)
    try:
        pereggrn_perturbations.check_perturbation_dataset(dataset_name)
    except Exception as e:
        print(repr(e))
