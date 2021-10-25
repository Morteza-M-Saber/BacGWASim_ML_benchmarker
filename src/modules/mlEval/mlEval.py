# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:34:01 2020

@author: Masih
"""


def get_options():
    import argparse

    description = "Evaluating performance of machine learning apporach"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--infile", help="Output of ml gwas test ")
    parser.add_argument("--causalVar", help="path to true causal variants ")
    parser.add_argument("--out", help="Evaluation files")

    return parser.parse_args()


options = get_options()

import json

import pandas as pd


def ml_eval(infile, causalVar, out):

    # 1) read the result of bagging_lgbm output

    df = pd.read_csv(infile)
    # df.sort_values(by='normalized_importance',ascending=False,inplace=True), it is already sorted

    # 2) read the true causal marker data
    CausalLoci = causalVar
    causalD = pd.read_csv(CausalLoci, sep="\t")
    causal = causalD.QTL
    causal_eff = causalD.Effect
    causal_eff_norm = [round(item / sum(causal_eff), 3) for item in causal_eff]
    fit_dict = {}
    for item_ in zip(causal, causal_eff, causal_eff_norm):
        if round(float(item_[1])) not in fit_dict:
            fit_dict[round(float(item_[1]))] = [[item_[0], item_[2]]]
        else:
            fit_dict[round(float(item_[1]))].append([item_[0], item_[2]])
    # 3)extract the ranking and feature importance of causal marker by bagging_lgbm
    eval_dict = {}
    for item_ in fit_dict:
        eval_dict[item_] = [[], [], []]
        for item2_ in fit_dict[item_]:
            if item2_[0] in list(df.feature):
                rank = df[df["feature"] == item2_[0]].index.item()
                fit_imp = round(
                    float(df[df["feature"] == item2_[0]]["normalized_importance"]), 3
                )
                eval_dict[item_][0].append(rank)
                eval_dict[item_][1].append(fit_imp)
                eval_dict[item_][2].append(item2_[1])
            else:
                print("%s not in output of GWAS tool!!" % (item2_[0]))
    # 4)writing the output
    txt = open(str(out), "w")
    txt.write("%s\t%s\t%s\n" % ("EffectSize", "variantRank", "FeatureImportances"))
    for item in eval_dict:
        txt.write(
            "%s\t%s\t%s\n" % (item, str(eval_dict[item][0]), str(eval_dict[item][1]))
        )
    txt.close()
    # 5) writing the output as .json file
    with open(str(out) + ".json", "w") as f:
        json.dump(eval_dict, f)


ml_eval(options.infile, options.causalVar, options.out)
