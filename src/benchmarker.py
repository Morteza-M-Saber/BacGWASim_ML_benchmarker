import argparse
import os
import sys

import snakemake
import yaml

parser = argparse.ArgumentParser(prog="mlBenchmarker", description="Description")
parser.add_argument(
    "--snakefile", dest="snakefile", metavar="FILE", help="Path to a snakefile",
)
parser.add_argument(
    "--config", dest="config", metavar="FILE", help="Path to a config file",
)
parser.add_argument(
    "--method",
    dest="method",
    default=None,
    choices=["gwas", "ml"],
    help="method for feature selection",
)
parser.add_argument(
    "--mlModel",
    dest="mlModel",
    default=None,
    choices=["lgbm", "lr", "rf", "xgb", "svc"],
    help="ml model for feature selection",
)
parser.add_argument(
    "--gwasModel",
    dest="gwasModel",
    default=None,
    choices=["gemma", "pyseer"],
    help="gwas model for feature selection",
)
parser.add_argument(
    "--alpha",
    dest="alpha",
    default=None,
    choices=["1", "0.0069"],
    help="alpha value for pyseer enet model (lasso:1,enet:0.0069)",
)
parser.add_argument(
    "--outDir", dest="outDir", metavar="DIR", help="Path to the output directory",
)

args = parser.parse_args()
args = vars(args)


# Loading default config values
with open(args["config"]) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
config["method"] = args["method"]
config["mlModel"] = args["mlModel"]
config["gwasModel"] = args["gwasModel"]
config["alpha"] = args["alpha"]
config["outDir"] = args["outDir"]
snakemake.snakemake(
    snakefile=args["snakefile"], config=config, forceall=True, cores=config["cores"]
)
