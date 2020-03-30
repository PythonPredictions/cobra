# third party imports
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_variable_importance(df_variable_importance: pd.DataFrame,
                             title: str=None,
                             dim: tuple=(12, 8)):
    """Plot variable importance of a given model

    Parameters
    ----------
    df_variable_importance : pd.DataFrame
        DataFrame containing columns predictor and importance
    title : str, optional
        Title of the plot
    dim : tuple, optional
        tuple with width and lentgh of the plot
    """

    # plot data
    fig, ax = plt.subplots(figsize=dim)
    ax = sns.barplot(x="importance", y="predictor",
                     data=df_variable_importance)
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Variable importance")
    plt.show()


def plot_predictor_quality(df_auc: pd.DataFrame,
                           dim: tuple=(12, 8)):
    """Plot univariate quality of the predictors

    Parameters
    ----------
    df_auc : pd.DatFrame
        Contains for each variable the train auc and selection auc allong with
        a boolean indicating whether or not it is selected based on the
        criteria
    dim : tuple, optional
        tuple with width and lentgh of the plot
    """

    plt.style.use('seaborn-darkgrid')

    df = (df_auc[df_auc["preselection"]]
          .sort_values(by='AUC train', ascending=False))

    df = pd.melt(df, id_vars=["predictor"],
                 value_vars=["AUC train", "AUC selection"],
                 var_name="partition",
                 value_name="AUC")

    # plots
    fig, ax = plt.subplots(figsize=dim)

    ax = sns.barplot(x="AUC", y="predictor", hue="partition", data=df)
    ax.set_title('Univariate Quality of Predictors')
    plt.show()


def plot_correlation_matrix(df_corr: pd.DataFrame,
                            dim: tuple=(12, 8)):
    """Plot correlation matrix amongst the predictors

    Parameters
    ----------
    df_corr : pd.DataFrame
        Correlation matrix
    dim : tuple, optional
        tuple with width and lentgh of the plot
    """
    fig, ax = plt.subplots(figsize=dim)
    ax = sns.heatmap(df_corr, cmap='Blues')
    ax.set_title('Correlation Matrix')
    plt.show()
