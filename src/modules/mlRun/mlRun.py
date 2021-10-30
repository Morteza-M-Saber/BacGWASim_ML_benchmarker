# -*- coding: utf-8 -*-
"""
@author: Masih

Run LGBMClassifier feature selection 
?? ?? ?????? ????? ?????, ???? ??? ?? ????? ???.
"""

###@1)importing required libraries and functions
import os.path
import sys

sys.path.insert(0, "/project/6006375/masih/projects/python_classes")

# import libraries
import pandas as pd


def get_options():
    import argparse

    description = "Run ML FeatureSelector on the dataset"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--infile", help="simulation in pandas pickle format")
    parser.add_argument(
        "--mlModel", help="machine learning model used for feature selection"
    )
    parser.add_argument("--phen", help="simulated phenotype in  pandas pickle format")
    parser.add_argument("--out", help="Complete path to output directory ")
    return parser.parse_args()


options = get_options()


def ml_fitSelect(infile, mlModel, phen, outfile):

    ###@2)reading the dataset
    df = pd.read_pickle(infile)
    # dfFrac=df.sample(frac=0.1, replace=False, random_state=1)
    # dfFrac1=dfFrac.sample(frac=0.1, replace=False, axis=1, random_state=1)
    phen = pd.read_pickle(phen)
    # removing samples with missing phenotype labels
    na = phen.index[phen["phenotype"] == -9].tolist()
    df.drop(na, axis=0, inplace=True)
    phen.drop(na, axis=0, inplace=True)
    # renaming the data for ML input
    X = df
    Y = phen.phenotype
    ###@4-1) Feature selection using lgbm featureSelector
    from feature_selector import FeatureSelector

    fs = FeatureSelector(data=X, labels=Y)
    if mlModel == "svc":
        df, eval = fs.identify_fit_importance_svc()
    elif mlModel == "lr":
        df, eval = fs.identify_fit_importance_lr()
    elif mlModel == "rf":
        df, eval = fs.identify_fit_importance_rf()
    elif mlModel == "lgbm":
        df, eval = fs.identify_fit_importance_lgbm()
    elif mlModel == "xgb":
        df, eval = fs.identify_fit_importance_xgb()
    df.sort_values(by="normalized_importance", ascending=False, inplace=True)
    # write the features importances in a csv file
    df.to_csv(outfile)
    eval.to_csv(outfile[:-4] + "CV.csv")
    return (df, eval)


ml_fitSelect(options.infile, options.mlModel, options.phen, options.out)
