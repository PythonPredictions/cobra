# third party imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_univariate_predictor_quality(df_metric: pd.DataFrame,
                                      dim: tuple=(12, 8),
                                      path: str=None):
    """Plot univariate quality of the predictors

    Parameters
    ----------
    df_metric : pd.DatFrame
        DataFrame containing for each variable the train AUC or RMSE and
        test AUC or RMSE along with a boolean indicating whether or not it is selected
        based on the criteria.
    dim : tuple, optional
        Width and length of the plot.
    path : str, optional
        Path to store the figure.
    """

    if "AUC selection" in df_metric.columns:
        metric = "AUC"
        ascending = False
    elif "RMSE selection" in df_metric.columns:
        metric = "RMSE"
        ascending = True

    df = (df_metric[df_metric["preselection"]]
          .sort_values(by=metric+" selection", ascending=ascending))

    df = pd.melt(df, id_vars=["predictor"],
                 value_vars=[metric+" train", metric+" selection"],
                 var_name="split",
                 value_name=metric)

    # plot data
    with plt.style.context("seaborn-ticks"):
        fig, ax = plt.subplots(figsize=dim)

        ax = sns.barplot(x=metric, y="predictor", hue="split", data=df)
        ax.set_title("Univariate Quality of Predictors")

        # Set pretty axis
        sns.despine(ax=ax, right=True)

        # Remove white lines from the second axis
        ax.grid(False)

        if path is not None:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()


def plot_correlation_matrix(df_corr: pd.DataFrame,
                            dim: tuple=(12, 8),
                            path: str=None):
    """Plot correlation matrix amongst the predictors

    Parameters
    ----------
    df_corr : pd.DataFrame
        Correlation matrix.
    dim : tuple, optional
        Width and length of the plot.
    path : str, optional
        Path to store the figure.
    """
    fig, ax = plt.subplots(figsize=dim)
    ax = sns.heatmap(df_corr, cmap='Blues')
    ax.set_title('Correlation Matrix')

    if path is not None:
        plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_performance_curves(model_performance: pd.DataFrame,
                            dim: tuple=(12, 8),
                            path: str=None,
                            colors: dict={"train": "#0099bf",
                                          "selection": "#ff9500",
                                          "validation": "#8064a2"}):
    """Plot performance curves generated by the forward feature selection
    for the train-selection-validation sets

    Parameters
    ----------
    model_performance : pd.DataFrame
        Contains train-selection-validation performance for each model trained
        in the forward feature selection.
    dim : tuple, optional
        Width and length of the plot.
    path : str, optional
        Path to store the figure.
    """

    model_type = model_performance["model_type"][0]

    if model_type == "classification":
        metric = "AUC"
    elif model_type == "regression":
        metric = "RMSE"

    max_metric = np.round(max(max(model_performance['train_performance']),
                              max(model_performance['selection_performance']),
                              max(model_performance['validation_performance'])), 1)

    with plt.style.context("seaborn-whitegrid"):

        fig, ax = plt.subplots(figsize=dim)

        plt.plot(model_performance['train_performance'], marker=".",
                 markersize=20, linewidth=3, label=metric+" train",
                 color=colors["train"])
        plt.plot(model_performance['selection_performance'], marker=".",
                 markersize=20, linewidth=3, label=metric+" selection",
                 color=colors["selection"])
        plt.plot(model_performance['validation_performance'], marker=".",
                 markersize=20, linewidth=3, label=metric+" validation",
                 color=colors["validation"])

        # Set x- and y-ticks
        ax.set_xticks(np.arange(len(model_performance['last_added_predictor'])))
        ax.set_xticklabels(model_performance['last_added_predictor'].tolist(),
                           rotation=40, ha='right')

        if model_type == "classification":
            ax.set_yticks(np.arange(0.5, max_metric + 0.02, 0.05))
        elif model_type == "regression":
            ax.set_yticks(np.arange(0, max_metric*1.02, np.floor(max_metric/10)))

        # Make pretty
        ax.legend(loc='lower right')
        fig.suptitle('Performance curves forward feature selection',
                     fontsize=20)
        plt.ylabel('Model performance')

        if path is not None:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()


def plot_variable_importance(df_variable_importance: pd.DataFrame,
                             title: str=None,
                             dim: tuple=(12, 8),
                             path: str=None):
    """Plot variable importance of a given model

    Parameters
    ----------
    df_variable_importance : pd.DataFrame
        DataFrame containing columns predictor and importance
    title : str, optional
        Title of the plot.
    dim : tuple, optional
        Width and length of the plot.
    path : str, optional
        Path to store the figure.
    """
    with plt.style.context("seaborn-ticks"):
        fig, ax = plt.subplots(figsize=dim)
        ax = sns.barplot(x="importance", y="predictor",
                         data=df_variable_importance,
                         color="cornflowerblue")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Variable importance")

        # Set Axis - make them pretty
        sns.despine(ax=ax, right=True)

        # Remove white lines from the second axis
        ax.grid(False)

        if path is not None:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()
