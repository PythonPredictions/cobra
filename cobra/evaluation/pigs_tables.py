"""Create Predictor Insight Graph tables."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

import cobra.utils as utils

def generate_pig_tables(basetable: pd.DataFrame,
                        id_column_name: str,
                        target_column_name: str,
                        preprocessed_predictors: list) -> pd.DataFrame:
    """Compute PIG tables for all predictors in preprocessed_predictors.

    The output is a DataFrame with columns ``variable``, ``label``,
    ``pop_size``, ``global_avg_target`` and ``avg_target``.

    Parameters
    ----------
    basetable : pd.DataFrame
        Basetable to compute PIG tables from.
    id_column_name : str
        Name of the basetable column containing the IDs of the basetable rows
        (e.g. customernumber).
    target_column_name : str
        Name of the basetable column containing the target values to predict.
    preprocessed_predictors: list
        List of basetable column names containing preprocessed predictors.

    Returns
    -------
    pd.DataFrame
        DataFrame containing a PIG table for all predictors.
    """
    pigs = [
        compute_pig_table(basetable,
                          column_name,
                          target_column_name,
                          id_column_name)
        for column_name in sorted(preprocessed_predictors)
        if column_name not in [id_column_name, target_column_name]
    ]
    output = pd.concat(pigs)
    return output


def compute_pig_table(basetable: pd.DataFrame,
                      predictor_column_name: str,
                      target_column_name: str,
                      id_column_name: str) -> pd.DataFrame:
    """Compute the PIG table of a given predictor for a given target.

    Parameters
    ----------
    basetable : pd.DataFrame
        Input data from which to compute the pig table.
    predictor_column_name : str
        Predictor name of which to compute the pig table.
    target_column_name : str
        Name of the target variable.
    id_column_name : str
        Name of the id column (used to count population size).

    Returns
    -------
    pd.DataFrame
        PIG table as a DataFrame
    """
    global_avg_target = basetable[target_column_name].mean()

    # group by the binned variable, compute the incidence
    # (=mean of the target for the given bin) and compute the bin size
    # (e.g. COUNT(id_column_name)). After that, rename the columns
    res = (basetable.groupby(predictor_column_name)
           .agg({target_column_name: "mean", id_column_name: "size"})
           .reset_index()
           .rename(columns={predictor_column_name: "label",
                            target_column_name: "avg_target",
                            id_column_name: "pop_size"}))

    # add the column name to a variable column
    # add the average incidence
    # replace population size by a percentage of total population
    res["variable"] = utils.clean_predictor_name(predictor_column_name)
    res["global_avg_target"] = global_avg_target
    res["pop_size"] = res["pop_size"]/len(basetable.index)

    # make sure to always return the data with the proper column order
    column_order = ["variable", "label", "pop_size",
                    "global_avg_target", "avg_target"]

    return res[column_order]


def plot_incidence(pig_tables: pd.DataFrame,
                   variable: str,
                   model_type: str,
                   column_order: list=None,
                   dim: tuple=(12, 8)):
    """Plot a Predictor Insights Graph (PIG).
    
    A PIG is a graph in which the mean
    target value is plotted for a number of bins constructed from a predictor
    variable. When the target is a binary classification target,
    the plotted mean target value is a true incidence rate.

    Bins are ordered in descending order of mean target value
    unless specified otherwise with the `column_order` list.

    Parameters
    ----------
    pig_tables: pd.DataFrame
        Dataframe with cleaned, binned, partitioned and prepared data,
        as created by generate_pig_tables() from this module.
    variable: str
        Name of the predictor variable for which the PIG will be plotted.
    model_type: str
        Type of model (either "classification" or "regression").
    column_order: list, default=None
        Explicit order of the value bins of the predictor variable to be used
        on the PIG.
    dim: tuple, default=(12, 8)
        Optional tuple to configure the width and length of the plot.
    """
    if model_type not in ["classification", "regression"]:
        raise ValueError("An unexpected value was set for the model_type "
                         "parameter. Expected 'classification' or "
                         "'regression'.")

    df_plot = pig_tables[pig_tables['variable'] == variable].copy()

    if column_order is not None:
        if not set(df_plot['label']) == set(column_order):
            raise ValueError(
                'The column_order and pig_tables parameters do not contain '
                'the same set of variables.')

        df_plot['label'] = df_plot['label'].astype('category')
        df_plot['label'].cat.reorder_categories(column_order,
                                                inplace=True)

        df_plot.sort_values(by=['label'], ascending=True, inplace=True)
        df_plot.reset_index(inplace=True)
    else:
        df_plot.sort_values(by=['avg_target'], ascending=False, inplace=True)
        df_plot.reset_index(inplace=True)

    with plt.style.context("seaborn-ticks"):
        fig, ax = plt.subplots(figsize=dim)

        # --------------------------
        # Left axis - average target
        # --------------------------
        ax.plot(df_plot['label'], df_plot['avg_target'],
                color="#00ccff", marker=".",
                markersize=20, linewidth=3,
                label='incidence rate per bin' if model_type == "classification" else "mean target value per bin",
                zorder=10)

        ax.plot(df_plot['label'], df_plot['global_avg_target'],
                color="#022252", linestyle='--', linewidth=4,
                label='average incidence rate' if model_type == "classification" else "global mean target value",
                zorder=10)

        # Dummy line to have label on second axis from first
        ax.plot(np.nan, "#939598", linewidth=6, label='bin size')

        # Set labels & ticks
        ax.set_ylabel('incidence' if model_type == "classification" else "mean target value",
                      fontsize=16)
        ax.set_xlabel('{} bins' ''.format(variable), fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        plt.setp(ax.get_xticklabels(),
                 rotation=45, ha="right", rotation_mode="anchor")
        ax.yaxis.set_tick_params(labelsize=14)

        if model_type == "classification":
            # Mean target values are between 0 and 1 (target incidence rate),
            # so format them as percentages
            ax.set_yticks(np.arange(0, max(df_plot['avg_target'])+0.05, 0.05))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        elif model_type == "regression":
            # If the difference between the highest avg. target of all bins
            # versus the global avg. target AND the difference between the
            # lowest avg. target versus the global avg. target are both smaller
            # than 25% of the global avg. target itself, we increase the
            # y-axis range, to avoid that the minor avg. target differences are
            # spread out over the configured figure height, suggesting
            # incorrectly that there are big differences in avg. target across
            # the bins and versus the global avg. target.
            # (Motivation for the AND above: if on one end there IS enough
            # difference, the effect that we discuss here does not occur.)
            global_avg_target = max(df_plot['global_avg_target'])  # series of same number, for every bin.
            if ((np.abs((max(df_plot['avg_target']) - global_avg_target)) / global_avg_target < 0.25)
                    and (np.abs((min(df_plot['avg_target']) - global_avg_target)) / global_avg_target < 0.25)):
                ax.set_ylim(global_avg_target * 0.75,
                            global_avg_target * 1.25)

        # Remove ticks but keep the labels
        ax.tick_params(axis='both', which='both', length=0)
        ax.tick_params(axis='y', colors="#00ccff")
        ax.yaxis.label.set_color('#00ccff')

        # -----------------
        # Right Axis - bins
        # -----------------
        ax2 = ax.twinx()

        ax2.bar(df_plot['label'], df_plot['pop_size'],
                align='center', color="#939598", zorder=1)

        # Set labels & ticks
        ax2.set_xlabel('{} bins' ''.format(variable), fontsize=16)
        ax2.xaxis.set_tick_params(rotation=45, labelsize=14)

        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax2.set_ylabel('population size', fontsize=16)
        ax2.tick_params(axis='y', colors="#939598")
        ax2.yaxis.label.set_color('#939598')

        # Despine & prettify
        sns.despine(ax=ax, right=True, left=True)
        sns.despine(ax=ax2, left=True, right=False)
        ax2.spines['right'].set_color('white')

        ax2.grid(False)

        # Title & legend
        if model_type == "classification":
            title = "Incidence plot - " + variable
        else:
            title = "Mean target plot - " + variable
        fig.suptitle(title, fontsize=22)
        ax.legend(frameon=False, bbox_to_anchor=(0., 1.01, 1., .102),
                  loc=3, ncol=1, mode="expand", borderaxespad=0.,
                  prop={"size": 14})

        # Set order of layers
        ax.set_zorder(1)
        ax.patch.set_visible(False)

        del df_plot

        plt.tight_layout()
        plt.margins(0.01)

        # Show
        plt.show()
