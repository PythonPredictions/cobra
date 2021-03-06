# third party imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_univariate_predictor_quality(df_auc: pd.DataFrame,
                                      dim: tuple=(12, 8),
                                      path: str=None):
    """Plot univariate quality of the predictors

    Parameters
    ----------
    df_auc : pd.DatFrame
        Contains for each variable the train auc and selection auc allong with
        a boolean indicating whether or not it is selected based on the
        criteria
    dim : tuple, optional
        tuple with width and lentgh of the plot
    path : str, optional
        path to store the figure
    """

    df = (df_auc[df_auc["preselection"]]
          .sort_values(by='AUC train', ascending=False))

    df = pd.melt(df, id_vars=["predictor"],
                 value_vars=["AUC train", "AUC selection"],
                 var_name="split",
                 value_name="AUC")

    # plot data
    with plt.style.context("seaborn-ticks"):
        fig, ax = plt.subplots(figsize=dim)

        ax = sns.barplot(x="AUC", y="predictor", hue="split", data=df)
        ax.set_title('Univariate Quality of Predictors')

        # Set Axis - make them pretty
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
        Correlation matrix
    dim : tuple, optional
        tuple with width and lentgh of the plot
    path : str, optional
        path to store the figure
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
        contains train-selection-validation performance for each model trained
        in the forward feature selection
    dim : tuple, optional
        tuple with width and lentgh of the plot
    path : str, optional
        path to store the figure
    """
    highest_auc = np.round(max(max(model_performance['train_performance']),
                               max(model_performance['selection_performance']),
                               max(model_performance['validation_performance'])
                               ), 1)

    with plt.style.context("seaborn-whitegrid"):

        fig, ax = plt.subplots(figsize=dim)

        plt.plot(model_performance['train_performance'], marker=".",
                 markersize=20, linewidth=3, label='AUC train',
                 color=colors["train"])
        plt.plot(model_performance['selection_performance'], marker=".",
                 markersize=20, linewidth=3, label='AUC selection',
                 color=colors["selection"])
        plt.plot(model_performance['validation_performance'], marker=".",
                 markersize=20, linewidth=3, label='AUC validation',
                 color=colors["validation"])
        # Set x/yticks
        ax.set_xticks(np.arange(len(model_performance['last_added_predictor'])))
        ax.set_xticklabels(model_performance['last_added_predictor'].tolist(),
                           rotation=40, ha='right')
        ax.set_yticks(np.arange(0.5, highest_auc + 0.02, 0.05))
        #Make Pretty
        ax.legend(loc='lower right')
        fig.suptitle('Performance curves - forward feature selection',
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
        Title of the plot
    dim : tuple, optional
        tuple with width and lentgh of the plot
    path : str, optional
        path to store the figure
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
