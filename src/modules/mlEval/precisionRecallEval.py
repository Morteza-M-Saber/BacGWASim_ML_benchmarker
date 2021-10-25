# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:51:53 2020

@author: Masih
?? ?? ?????? ????? ?????, ???? ??? ?? ????? ???
"""


def get_options():
    import argparse

    description = "Evaluating performance of ml approach"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--infile", help="Output of ml gwas test ")
    parser.add_argument("--causalVar", help="path to true causal variants ")
    parser.add_argument("--out", help="Evaluation files")

    return parser.parse_args()


options = get_options()

import numpy as np
import pandas as pd


def pre_recall_eval(infile, causalVar, out):
    df = pd.read_csv(infile)
    # df.sort_values(by='normalized_importance',ascending=False,inplace=True)

    CausalLoci = causalVar
    causalD = pd.read_csv(CausalLoci, sep="\t")
    causal = causalD.QTL

    precision, recall = [], []
    for tops_ in range(16, 500):
        top_select = df.head(tops_)
        tp = len(set(causal).intersection(top_select.feature))
        fp = len(top_select.feature) - tp
        fn = len(set(causal)) - tp
        pre_ = tp / (tp + fp)
        rec_ = tp / (tp + fn)
        precision.append(pre_)
        recall.append(rec_)

    # One-dimensional linear interpolation using interp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    from numpy import interp

    mean_rec = np.linspace(0, 1, 100)
    interp_ = interp(mean_rec, recall, precision)
    interp_[0] = 1
    # calculating area under curve using sklearn auc function
    from sklearn.metrics import auc

    pre_recal_auc = round(auc(mean_rec, interp_), 3)
    sns.lineplot(
        x=mean_rec,
        y=interp_,
        err_style=None,
        palette="tab10",
        linewidth=2.5,
        label="AUC = %0.2f" % pre_recal_auc,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out + ".png", dpi=300)

    # save the interp_ data so could be used later for plotting the averge
    # score across all simulations
    df = pd.DataFrame({"mean_rec": mean_rec, "precisions": interp_})
    df.to_csv(out + ".csv")


pre_recall_eval(options.infile, options.causalVar, options.out)
