# third party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import cobra.utils as utils


def generate_pig_tables(data: pd.DataFrame,
                        id_column_name: str,
                        target_column_name: str,
                        preprocessed_predictors: list) -> pd.DataFrame:
    """Compute PIG tables for all predictors in preprocessed_predictors. The
    output is a DataFrame with columns ``variable``, ``label``, ``pop_size``,
    ``avg_incidence`` and ``incidence``

    Parameters
    ----------
    data : pd.DataFrame
        basetable to compute PIG tables of
    id_column_name : str
        column name of the id (e.g. customernumber)
    target_column_name : str
        column name of the target
    predictors: list
        list of preprocessed predictor names

    Returns
    -------
    pd.DataFrame
        DataFrame containing a PIG table for all predictors
    """

    pigs = [compute_pig_table(data, column_name, target_column_name,
                              id_column_name)
            for column_name in sorted(preprocessed_predictors)
            if column_name not in [id_column_name, target_column_name]]

    output = pd.concat(pigs)

    return output


def compute_pig_table(data: pd.DataFrame,
                      column_name: str,
                      target_column_name: str,
                      id_column_name: str) -> pd.DataFrame:
    """Compute the pig table of a given predictor for a given target

    Parameters
    ----------
    data : pd.DataFrame
        input data from which to compute the pig table
    column_name : str
        predictor name of which to compute the pig table
    target_column_name : str
        name of the target variable
    id_column_name : str
        name of the id column (used to count population size)

    Returns
    -------
    pd.DataFrame
        pig table as a DataFrame
    """
    avg_incidence = data[target_column_name].mean()

    # group by the binned variable, compute the incidence
    # (=mean of the target for the given bin) and compute the bin size
    # (e.g. COUNT(id_column_name)). After that, rename the columns
    res = (data.groupby(column_name)
               .agg({target_column_name: "mean", id_column_name: "size"})
               .reset_index()
               .rename(columns={column_name: "label",
                                target_column_name: "incidence",
                                id_column_name: "pop_size"}))

    # add the column name to a variable column
    # add the average incidence
    # replace population size by a percentage of total population
    res["variable"] = utils.clean_predictor_name(column_name)
    res["avg_incidence"] = avg_incidence
    res["pop_size"] = res["pop_size"]/len(data.index)

    # make sure to always return the data with the proper column order
    column_order = ["variable", "label", "pop_size",
                    "avg_incidence", "incidence"]

    return res[column_order]


def plot_incidence(df: pd.DataFrame, variable: str,
                   column_order: list=None, dim: tuple=(12, 8)):
    """Function plots Predictor Incidence Graphs (PIGs).
    Bins are ordered in descening order of bin incidence
    unless specified otherwise with `column_order` list.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe with cleaned, binned, partitioned and prepared data

    variable: str
        variable for which the incidence plot will be shown

    column_order: list, default=None
        explicit order of variable

    dim: tuple, default=(12, 8)
        tuple with width and lentgh of the plot
    """
    df_plot = df[df['variable'] == variable]

    if column_order is not None:

        if not set(df_plot['label']) == set(column_order):
            raise ValueError(
                'Variables in column_order and dataframe are not equal')

        df_plot['label'] = df_plot['label'].astype('category')
        df_plot['label'].cat.reorder_categories(column_order,
                                                inplace=True)

        df_plot.sort_values(by=['label'], ascending=True, inplace=True)
        df_plot.reset_index(inplace=True)
    else:
        df_plot.sort_values(by=['incidence'], ascending=False, inplace=True)
        df_plot.reset_index(inplace=True)

    with plt.style.context("seaborn-ticks"):
        fig, ax = plt.subplots(figsize=dim)

        # First Axis
        ax.bar(df_plot['label'], df_plot['pop_size'],
               align='center', color="cornflowerblue")
        ax.set_ylabel('population size', fontsize=16)
        ax.set_xlabel('{} bins' ''.format(variable), fontsize=16)
        ax.xaxis.set_tick_params(rotation=45, labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)

        max_inc = max(df_plot['incidence'])

        # Second Axis
        ax2 = ax.twinx()

        plt.plot(df_plot['incidence'], color="darkorange", marker=".",
                 markersize=20, linewidth=3, label='incidence rate per bin')
        plt.plot(df_plot['avg_incidence'], color="dimgrey", linewidth=4,
                 linestyle='--',
                 label='average incidence rate')

        # dummy line to have label on second axis from first
        ax2.plot(np.nan, "cornflowerblue", linewidth=6, label='bin size')
        ax2.set_yticks(np.arange(0, max_inc+0.05, 0.05))
        ax2.set_yticklabels(
            ['{:3.1f}%'.format(x*100) for x in ax2.get_yticks()])
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.set_ylabel('incidence', fontsize=16)

        sns.despine(ax=ax, right=True, left=True)
        sns.despine(ax=ax2, left=True, right=False)
        ax2.spines['right'].set_color('white')

        ax2.grid(False)

        fig.suptitle('Incidence Plot - ' + variable, fontsize=22, y=1.02)
        ax2.legend(frameon=False, bbox_to_anchor=(0., 1.01, 1., .102),
                   loc=3, ncol=1, mode="expand", borderaxespad=0.,
                   prop={"size": 14})
        plt.show()
