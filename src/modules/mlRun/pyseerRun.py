# pyseerRun.py


def get_options():
    import argparse

    description = "Running pyseerlmm on the vcf file"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--vcf", help="Complete path to the vcf file for running GWAS ")
    parser.add_argument(
        "--model", default="enet", help="Model for running GWAS (lmm/enet/rf/blup"
    )
    parser.add_argument("--alpha", default=1, help="alpha (1:lasso,0.0069:elasticNet,")
    parser.add_argument(
        "--phen", help="Complete path to the phenotype file of the samples "
    )
    parser.add_argument(
        "--phenType",
        default="quant",
        help="quant for continuous and cc for case-control",
    )
    parser.add_argument(
        "--cpu", default=3, help="Number of CPUs to be used for multithreading"
    )
    parser.add_argument("--out", help="PySeerLMM output file")

    return parser.parse_args()


options = get_options()


import os
from subprocess import call

import numpy as np
import pandas as pd


def pyseerRun(
    phen,
    phenType,
    vcf,
    cpu,
    output,
    model,
    alpha,
    python="~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python",
    phenoFormatter="/project/6006375/masih/projects/Papers/7thPaper/analysis/modules/mlRun/phenoFormatter.py",
    pyseer="~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/pyseer",
    bcftools="/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/intel2016.4/bcftools/1.9/bin/bcftools",
):
    outDir = os.path.split(output)[0]
    # change sample names from numbers to strings (recommended by pyseer)
    phenotype = phen
    txt = open(os.path.join(outDir, "sampleRename.txt"), "w")
    with open(phenotype, "r") as file:
        line = file.readline()
        while line:
            name = line.split()[1]
            txt.write("%s %s\n" % (name, name + "_"))
            line = file.readline()
    txt.close()
    callString = "%s reheader %s -s %s -o %s" % (
        bcftools,
        vcf,
        os.path.join(outDir, "sampleRename.txt"),
        os.path.join(outDir, "vcfRename.vcf"),
    )
    call(callString, shell=True)
    # Format the phenotype for pyseer
    callString = "%s %s  %s --phenType %s --output %s" % (
        python,
        phenoFormatter,
        phenotype,
        phenType,
        os.path.join(str(outDir), "pyseerPhen"),
    )
    call(callString, shell=True)
    print("phenotype reformatting completed!")
    # Run pyseer
    CallString = (
        "%s --wg %s --alpha %s --phenotypes %s --vcf %s  --cpu %s  > %s 2> %s"
        % (
            pyseer,
            model,
            alpha,
            os.path.join(str(outDir), "pyseerPhen"),
            os.path.join(outDir, "vcfRename.vcf"),
            str(cpu),
            output + "_pyseer",
            output + "_wgScreen",
        )
    )
    call(CallString, shell=True)
    ###format the output,1)rank by lowest lrt_pvalue, 2)normalize beta as effect sizes
    df = pd.read_csv(output + "_pyseer", sep="\t")
    # create the rankings and set it as the index
    ind_ = range(len(list(df["filter-pvalue"])))
    df["ranks"] = ind_
    df.set_index("ranks", inplace=True)
    df["notes"].fillna(0, inplace=True)
    df["notes"].replace(
        [
            "af-filter",
            "bad-chisq",
            "pre-filtering-failed",
            "lrt-filtering-failed",
            "perfectly-separable-data",
            "firth-fail",
            "matrix-inversionerror",
        ],
        1,
        inplace=True,
    )
    df = df[df["notes"] != 1]

    # detected as true positive
    df.sort_values(
        by="filter-pvalue", inplace=True
    )  # sorting variants by lowest 'lrt_pvalue'
    df["variant"] = df["variant"].str.replace("_", ":")
    # create the rankings and set it as the index
    ind_ = range(len(list(df["filter-pvalue"])))
    df["ranks"] = ind_
    df.set_index("ranks", inplace=True)
    # getting the list of filtered items and add them with value of zero
    totalId = []
    with open(os.path.join(outDir, "vcfRename.vcf"), "r") as file:
        line = file.readline()
        while line[0] == "#":
            line = file.readline()
        while line:
            totalId.append(line.split("\t")[2])
            line = file.readline()
    miss = list(set(totalId) - set(df.variant))
    missDf = pd.DataFrame({"variant": miss, "filter-pvalue": [0] * len(miss)})
    df = df.append(missDf, ignore_index=True)
    # create the table to be used by mlEval.py
    fit_imp = np.array(df["filter-pvalue"]) / np.array(df["filter-pvalue"]).sum()
    feature_importance = pd.DataFrame(
        {"feature": df.variant, "normalized_importance": fit_imp,}
    )
    feature_importance.to_csv(output)


pyseerRun(
    options.phen,
    options.phenType,
    options.vcf,
    options.cpu,
    options.out,
    options.model,
    options.alpha,
)
