import argparse
import os
import sys
from typing import Dict

import snakemake
import yaml

parser = argparse.ArgumentParser(prog="mlBenchmarker", description="Description")
parser.add_argument(
    "--snakefile",
    dest="snakefile",
    default="src/benchmarkerML",
    help="Path to a snakefile",
)
parser.add_argument(
    "--config", 
    dest="config", 
    default="src/configfile.yaml", 
    help="Path to a config file",
)
parser.add_argument(
    "--simulation_file_pathes",
    dest="simulation_file_pathes",
    default=None,
    help="Path to the list of simulations pathes (.vcf for gwas and .pickle for ml), one path per line",
)
parser.add_argument(
    "--causal_variant_file_pathes",
    dest="causal_variant_file_pathes",
    default=None,
    help="Path to the list of causal variant file pathes, one path per line",
)
parser.add_argument(
    "--phenotype_file_pathes",
    dest="phenotype_file_pathes",
    default=None,
    help="Path to the list of phenotype pathes(.phen for gwas and .pickle for ml), one path per line",
)
parser.add_argument(
    "--method",
    dest="method",
    default="ml",
    choices=["gwas", "ml"],
    help="method for feature selection",
)
parser.add_argument(
    "--mlModel",
    dest="mlModel",
    default="lr",
    choices=["lgbm", "lr", "rf", "xgb", "svc"],
    help="ml model for feature selection",
)
parser.add_argument(
    "--gwasModel",
    dest="gwasModel",
    default="gemma",
    choices=["gemma", "pyseer"],
    help="gwas model for feature selection",
)
parser.add_argument(
    "--alpha",
    dest="alpha",
    type=float,
    default=1.0,
    choices=[1, 0.0069],
    help="alpha value for pyseer enet model (lasso:1,enet:0.0069)",
)
parser.add_argument(
    "--output", dest="output", help="Path to the output directory",
)

args = parser.parse_args()
args = vars(args)


# Loading default config values
if args["config"]:
    with open(args["config"]) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
else:
    config: Dict = dict()
if args["phenotype_file_pathes"]:
    config["phenotype_file_pathes"] = args["phenotype_file_pathes"]
if args["causal_variant_file_pathes"]:
    config["causal_variant_file_pathes"] = args["causal_variant_file_pathes"]
if args["simulation_file_pathes"]:
    config["simulation_file_pathes"] = args["simulation_file_pathes"]
if args["method"]:
    config["method"] = args["method"]
if args["mlModel"]:
    config["mlModel"] = args["mlModel"]
if args["gwasModel"]:
    config["gwasModel"] = args["gwasModel"]
if args["alpha"]:
    config["alpha"] = args["alpha"]
if args["output"]:
    config["output"] = args["output"]
snakemake.snakemake(
    snakefile=args["snakefile"], config=config, forceall=True, cores=config["cores"]
)
