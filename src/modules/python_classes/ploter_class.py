# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:45:17 2020

@author: Masih
ای که خواهان تولدی دیگری, نخست مرگ را پذیرا باش
"""
# convert cm to inch
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


class box_ploter:
    def __init__(self):

        pass

    def dot_box_plot(
        self,
        df,
        x,
        y,
        out,
        hue=False,
        ylabel="",
        xlabel="",
        label_median=True,
        showmeans=True,
        showfliers=False,
        figsize=(8, 8),
    ):
        """


        Parameters
        ----------
        df : TYPE
            Dataframe including the dataset.
        x : TYPE
            Categories of x axis of boxplot.
        y : TYPE
            the column of dataframe including information on categories.
        out : TYPE
            complete path to resulting figure.
        hue: STR
            hue column title
        ylabel : TYPE, optional
            label of y axis. The default is ''.
        xlabel : TYPE, optional
            label of x axis. The default is ''.
        label_median : TYPE, optional
            Show the median values of categories above each category. The default is True.
        showmeans : TYPE, optional
            Show the mean of values as rectangle. The default is True.
        showfliers : TYPE, optional
            Show the outliers in boxplot along with dot plot. The default is False.
        figsize: (x,y),optional.
            Size of the plot. The default is (8,8) in centimeters.

        Returns
        -------
        the boxplot figure.

        Example:
        tips = sns.load_dataset("tips")
        box_ploter(df=tips,x="day", y="total_bill",out='boxplot_test.png')

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from pylab import (
            axes,
            boxplot,
            figure,
            legend,
            plot,
            savefig,
            setp,
            show,
            xlim,
            ylim,
        )

        plt.figure(figsize=cm2inch(figsize))
        params = dict(data=df, x=x, y=y, hue=hue, dodge=True)
        sns.stripplot(
            size=8,
            jitter=0.35,
            palette="bright",
            edgecolor="black",
            linewidth=1,
            **params
        )
        p_box = sns.boxplot(
            palette="pastel",
            linewidth=4,
            showmeans=showmeans,
            showfliers=False,
            **params
        )
        if label_median:
            medians = df.groupby([x])[y].median()
            maxes = df.groupby([x])[y].max()
            vertical_offset = df[y].median() * 0.05  # offset from median for display
            for xtick in p_box.get_xticks():
                p_box.text(
                    xtick,
                    maxes[xtick] + vertical_offset,
                    round(medians[xtick], 3),
                    horizontalalignment="center",
                    size="small",
                    color="black",
                    weight="semibold",
                )
        plt.ylabel(ylabel, size=10)
        plt.xlabel("")
        # get legend information from the plot object
        handles, labels = p_box.get_legend_handles_labels()
        # specify just one legend
        plt.legend(handles[3:], labels[3:])
        plt.tight_layout()
        plt.savefig(out, dpi=300)


class line_ploter:
    def __init__(self):

        pass

    def line_plot(
        self,
        df,
        x,
        y,
        err,
        out,
        labs=[[None, None]],
        ylabel="",
        xlabel="",
        fnt_size=10,
        x_rotation=0,
        x_lim=(0, 1),
        y_lim=(0, 1),
        figsize=(8, 8),
    ):
        """


        Parameters
        ----------
        df : pandas dataframe
            Datafarme including the dataset.
        x : the column of df including x values
            DESCRIPTION.
        y : a list including columns of df to be plotted
            [y1,y2,..].
        err : a list including columns of df to be used as error bars.
            This should be in the same order as y values
            standard deviation of y values to produce erro bars.
        out : str
            complete path to output directory.
        labs : list[['AUC','STD'],['AUC2',STD2],...], optional
            DESCRIPTION. The default is [[None,None]].AUC,STD values per each of line plots
            This is specifically designed for the purpose of precision-recall curve of ML GWAS
            benchmarking project and should be changed for other projects later.
        ylabel : TYPE, optional
            DESCRIPTION. The default is ''.
        xlabel : TYPE, optional
            DESCRIPTION. The default is ''.
        fnt_size : TYPE, optional
            DESCRIPTION. The default is 10.
        x_rotation : TYPE, optional
            in case of long x labels to rotate them and make space. The default is 0.
        x_lim : TYPE, optional
            DESCRIPTION. The default is (0,1).
        y_lim : TYPE, optional
            DESCRIPTION. The default is (0,1).
        figsize : TYPE, optional
            figsize in cm. The default is (8, 8).

        Returns
        -------
        The line plots each for

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from pylab import (
            axes,
            boxplot,
            figure,
            legend,
            plot,
            savefig,
            setp,
            show,
            xlim,
            ylim,
        )

        # sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
        sns.set_context("paper")
        fig, ax = plt.subplots(figsize=cm2inch(figsize))
        colors_ = ["r", "b", "g"]
        for index_ in range(len(y)):
            y_ = y[index_]
            err_ = err[index_]
            x_ = x
            sns.lineplot(
                data=df,
                x=x_,
                y=y_,
                color=colors_[index_],
                label=r"AUC = %0.2f $\pm$ %0.2f" % (labs[index_][0], labs[index_][1]),
                lw=2,
                alpha=0.8,
            )
            bound_up = np.minimum(np.array(df[y_]) + np.array(df[err_]), 1)
            bound_low = np.maximum(np.array(df[y_]) - np.array(df[err_]), 0)
            ax.fill_between(
                np.array(df[x_]),
                bound_low,
                bound_up,
                color=colors_[index_],
                alpha=0.1,
                # label=r'$\pm$ 1 std. dev.'
            )
        plt.setp(ax.get_legend().get_texts(), fontsize=fnt_size)
        ax.tick_params(length=8, width=1)
        plt.xticks(size=fnt_size)  # 9-10 points are standard paper size
        plt.xticks(rotation=x_rotation)
        plt.yticks(size=fnt_size)
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        plt.xlabel(xlabel, size=fnt_size)
        plt.ylabel(ylabel, size=fnt_size)
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        plt.tight_layout()

        plt.savefig(out, dpi=300)
