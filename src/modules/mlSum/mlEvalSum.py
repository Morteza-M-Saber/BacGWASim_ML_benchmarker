# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:00:08 2020

@author: Masih
?? ?? ?????? ????? ?????, ???? ??? ?? ????? ???
"""


def get_options():
    import argparse

    description = "Summarizaing the performance of ML over all the replicates "
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--infile",
        help="ML evaluation results in json format in file (one file per line) ",
    )
    parser.add_argument("--out", help="Summarized evaluation file")

    return parser.parse_args()


options = get_options()

import json

import numpy as np
import pandas as pd


def ml_eval_summarize(infile, out):
    jsons = []
    with open(infile) as file:
        line = file.readline()
        while line:
            jsons.append(line.strip())
            line = file.readline()
    with open(jsons[0]) as f:
        res = json.load(f)
    resDict = {}
    for eff_ in res:
        resDict[eff_] = {"top50": [], "rank": [], "fit_imp": []}
    for reps in jsons:
        with open(reps) as f:
            resNow = json.load(f)
            for eff_ in resNow:
                top50_ = len([item for item in resNow[eff_][0] if item <= 50]) / len(
                    resNow[eff_][0]
                )
                rank_ = np.median(resNow[eff_][0])
                fit_imp_ = np.sum(resNow[eff_][1]) / np.sum(resNow[eff_][2])
                resDict[eff_]["top50"].append(top50_)
                resDict[eff_]["rank"].append(rank_)
                resDict[eff_]["fit_imp"].append(fit_imp_)
    df = pd.DataFrame()
    for eff_ in resDict:
        for metrics_ in resDict[eff_]:
            df[metrics_ + "_OR" + str(eff_)] = resDict[eff_][metrics_]
    df.to_csv(out)


ml_eval_summarize(options.infile, options.out)
