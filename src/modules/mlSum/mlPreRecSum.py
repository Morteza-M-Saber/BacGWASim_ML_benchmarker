import json
import os.path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc

sys.path.insert(0, "/project/6006375/masih/projects/python_classes")


def get_options():
    import argparse

    description = "Summarizaing the performance of Eval (precision-recall curve) over all the replicates"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--infile",
        help="ML eval (pre-recall curve) results in .csv format in file (one file per line) ",
    )
    parser.add_argument("--out", help="Summarized evaluation file")

    return parser.parse_args()


options = get_options()

# convert cm to inch
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


matplotlib.use("Agg")


def ml_pre_rec_summarize(infile, out):
    csvs = []
    with open(infile) as file:
        line = file.readline()
        while line:
            csvs.append(line.strip())
            line = file.readline()
    mean_rec = pd.read_csv(csvs[0]).mean_rec
    precision = []
    auc_ = []
    for reps in csvs:
        precision.append(pd.read_csv(reps).precisions)
        auc_.append(
            round(auc(pd.read_csv(reps).mean_rec, pd.read_csv(reps).precisions), 3)
        )
    # getting the mean
    mean_pre_ = np.mean(precision, axis=0)
    std_pre_ = np.std(precision, axis=0)
    # mean_tpr_xgb[-1] = 1.0
    mean_auc_ = auc(mean_rec, mean_pre_)
    std_auc_ = np.std(auc_)
    ##plotting the data
    from ploter_class import line_ploter

    df = pd.DataFrame(
        {
            "mean_rec": list(mean_rec),
            "mean_pre": list(mean_pre_),
            "std_pre": list(std_pre_),
        }
    )
    lp = line_ploter()
    lp.line_plot(
        df,
        "mean_rec",
        ["mean_pre"],
        ["std_pre"],
        out=out,
        labs=[[mean_auc_, std_auc_]],
        xlabel="Recall",
        ylabel="Precision",
        figsize=(8, 8),
    )

    ##) writing the output as .json file
    d = {
        "mean_rec": list(mean_rec),
        "mean_pre": list(mean_pre_),
        "std_pre": list(std_pre_),
        "mean_auc": mean_auc_,
        "std_auc": std_auc_,
    }
    with open(str(out).replace(".png", "") + "_pre_rec_summary.json", "w") as f:
        json.dump(d, f)


ml_pre_rec_summarize(options.infile, options.out)
