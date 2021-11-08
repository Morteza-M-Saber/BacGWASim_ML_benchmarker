import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import interp
from sklearn.metrics import auc


def get_options():

    description = "Evaluating performance of ml approach"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--infile", help="Output of ml gwas test ")
    parser.add_argument("--causalVar", help="path to true causal variants ")
    parser.add_argument("--out", help="Evaluation files")

    return parser.parse_args()


options = get_options()


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
    matplotlib.use("Agg")
    sns.set(style="whitegrid")
    mean_rec = np.linspace(0, 1, 100)
    interp_ = interp(mean_rec, recall, precision)
    interp_[0] = 1
    # calculating area under curve using sklearn auc function

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
    plt.savefig(out[:-4] + ".png", dpi=300)

    # save the interp_ data so could be used later for plotting the averge
    # score across all simulations
    df = pd.DataFrame({"mean_rec": mean_rec, "precisions": interp_})
    df.to_csv(out[:-4] + ".csv")


pre_recall_eval(options.infile, options.causalVar, options.out)
