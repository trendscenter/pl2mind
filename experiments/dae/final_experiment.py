from os import path

yaml_file = path.join(path.abspath(path.dirname(__file__)), "fine_tune.yaml")
name = "DAE Fine Tune"

fileparams = {
    "out_path": "${PYLEARN2_OUTS}"
}

default_hyperparams  = {
    "nvis": 0,
    "dataset_name": "smri",
    "learning_rate": 0.001,
    "min_lr": 0.0001,
    "decay_factor": 1.0005,
    "batch_size": 10,
    "init_momentum": 0.0,
    "final_momentum": 0.5,
    "termination_criterion": {
        "__builder__": "pylearn2.termination_criteria.MonitorBased",
        "channel_name": "\"valid_objective\"",
        "prop_decrease": 0,
        "N": 20
        },
    "data_class": "MRI_Standard",
    }

results_of_interest = [
    "objective"
]