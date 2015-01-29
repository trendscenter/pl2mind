from pl2mind.experiments.dae.dae_experiment import *

yaml_file = path.join(path.abspath(path.dirname(__file__)), "pretrain.yaml")
# Name for your experiment
name = "Second DAE Experiment"
default_hyperparams["nvis"] = default_hyperparams["nhid"]
default_hyperparams["corruptor"] = {
    "__builder__": "pylearn2.corruption.BinomialCorruptor",
    "corruption_level": .3
}