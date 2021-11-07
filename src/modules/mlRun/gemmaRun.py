import os
import pathlib
from subprocess import call

import numpy as np
import pandas as pd


def get_options():
    import argparse

    description = "Running GEMMA on .vcf and .phen and return featureimportances"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--vcf", help=".vcf file including all dataste ")
    parser.add_argument(
        "--phen", help="Sample phenotypes in GCTA format (case=1,affected=2,unknown=-9)"
    )
    parser.add_argument("--out", help="Feature importances and rankings in .csv ")

    return parser.parse_args()


options = get_options()


def gemma_run(
    vcf,
    phen,
    out,
    gemma="gemma",
    plink="plink",
):
    """

    Run GEMMA to estimate feature importances corrected for stratification

    Parameters
    --------

    vcf : string
        .vcf file including the genomic data

    phen: string
        .phen directory including the phenotype in GCTA format

    out: string
        directory where intermediate files and output will be written

    gemma: string, Default: Gemma from Beluga
        complete path to gemma tool

    plink: string, Default: Plink1.9 from Beluga
        complete path to plink1.9 tool


    Notes
    --------

    - This code is dependent on plink and gemma tools, directory should be provieded above

    """
    ###getting the directory to write the intermediate files
    actDir = pathlib.Path().absolute()
    outDir = "/".join(out.split("/")[:-1])
    # Creating relationship matrix
    ### Add phenotype to vcf file and create binary file using plink to be used by gemma
    callString = (
        "%s --vcf %s --pheno %s --allow-extra-chr --make-bed --allow-no-sex  --out  %s"
        % (plink, vcf, phen, os.path.join(outDir, "plink_binary"))
    )
    call(callString, shell=True)
    print("Plink phenotype attachment completed")
    ###Create relationship matrix usign gemma
    os.chdir(outDir)
    # callString='export LD_LIBRARY_PATH=/home/masih/anaconda3/pkgs/gsl-1.16-0/lib' #Subprocess.call does not work this way, you need to put this in .bashrc file to work
    # call(callString, shell=True)
    callString = "%s  -bfile %s -gk 1 -o CenterRelatadnessMatrix" % (
        gemma,
        "plink_binary",
    )  # Gemma by default creates an 'output' directory in current active directory and writes the outputs there, with outdir you are just adding further directory dow in 'outdir' so it can just be ignored.
    call(callString, shell=True)
    print("Gemma relMatix creation completed")
    ###Running gemma
    gsm = os.path.join("output", "CenterRelatadnessMatrix.cXX.txt")
    callString = "%s -bfile %s -k %s -lmm 4 -o gwas  > %s" % (
        gemma,
        "plink_binary",
        gsm,
        "gemma.ScreenOut",
    )
    call(callString, shell=True)
    ###format the output,1)rank by lowest lrt_pvalue, 2)normalize beta as effect sizes
    os.chdir(actDir)
    gemma_out = os.path.join(outDir, "output", "gwas.assoc.txt")
    df = pd.read_csv(gemma_out, sep="\t")
    df.sort_values(by="p_lrt", inplace=True)
    # create the rankings and set it as the index
    ind_ = np.arange(0, len(list(df.rs)), 1)
    df["ranks"] = ind_
    df.set_index("ranks", inplace=True)
    # estimate normalized fit_imp for all and beta_positive_only markers
    fit_positive = []
    for item in list(df["beta"]):
        if item >= 0:
            fit_positive.append(item)
        else:
            fit_positive.append(0)
    fit_pos = np.array(fit_positive) / np.array(fit_positive).sum()
    fit_all = np.absolute(df.beta) / np.absolute(df.beta).sum()
    feature_importance = pd.DataFrame(
        {
            "feature": df.rs,
            "all_normalized_importance": fit_all,
            "normalized_importance": fit_pos,
            "p_lrt": df.p_lrt,
        }
    )
    feature_importance.to_csv(out)


gemma_run(options.vcf, options.phen, options.out)

open